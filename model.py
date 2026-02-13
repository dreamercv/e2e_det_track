"""
纯 PyTorch 的 BEV + Sparse4D 检测模型组装与前向，不依赖 mmcv/mmdet。
时序在 backbone 内完成（输入时序图像 → 时序 BEV，B*T 合并），head 仅接收 feature_maps，不做跨 forward 的时序 cache/update。
"""
import numpy as np
import torch
import torch.nn as nn

from .instance_bank import InstanceBank
from .detection3d_blocks import (
    SparseBox3DEncoder,
    SparseBox3DRefinementModule,
    SparseBox3DKeyPointsGenerator,
)
from .bev_aggregation import BEVFeatureAggregation
from .head import Sparse4DHead, MHAWrapper, SimpleFFN
from .decoder import SparseBox3DDecoder
from .dn_sampler import DenoisingSampler
from .losses import FocalLoss, SparseBox3DLoss
from .track_head import TrackHead, track_affinity_loss, decode_track

__all__ = [
    "build_sparse4d_bev_head",
    "build_sparse4d_bev_model",
    "build_track_head",
    "TrackHead",
    "track_affinity_loss",
    "decode_track",
]


def build_sparse4d_bev_head(
    num_anchor: int = 900,
    embed_dims: int = 256,
    num_decoder: int = 6,
    num_single_frame_decoder: int = 5,
    num_classes: int = 10,
    bev_bounds=([-80.0, 120.0, 1.0], [-40.0, 40.0, 1.0]),
    anchor_init: np.ndarray = None,
    decouple_attn: bool = True,
    num_heads: int = 8,
    dropout: float = 0.1,
    feedforward_dims: int = 1024,
    use_dn: bool = True,
    num_dn_groups: int = 10,
    dn_noise_scale: float = 0.5,
    max_dn_gt: int = 32,
    add_neg_dn: bool = True,
    reg_weights: list = None,
    use_decoder: bool = False,
    decoder_num_output: int = 300,
    decoder_score_threshold: float = None,
):
    """构建 Sparse4D Head。输入为 feature_maps（如 (B*T, C, H, W)），无跨 forward 时序。use_decoder=True 时构建 SparseBox3DDecoder 用于推理得到 boxes_3d。"""
    if anchor_init is None:
        anchor_init = np.zeros((num_anchor, 11), dtype=np.float32)
    if reg_weights is None:
        reg_weights = [2.0] * 3 + [0.5] * 3 + [0.0] * 5  # 11 dims，与 anchor/box 一致

    # 无时序 cache：num_temp_instances=0，get/update/cache 的时序分支不生效
    instance_bank = InstanceBank(
        num_anchor=num_anchor,
        embed_dims=embed_dims,
        anchor=anchor_init,
        anchor_handler=None,
        num_temp_instances=0,
        default_time_interval=0.5,
        confidence_decay=0.6,
        feat_grad=False,
    )
    anchor_encoder = SparseBox3DEncoder(
        embed_dims=[128, 32, 32, 64] if decouple_attn else 256,
        vel_dims=3,
        mode="cat" if decouple_attn else "add",
        output_fc=not decouple_attn,
        in_loops=1,
        out_loops=4 if decouple_attn else 2,
    )
    gnn_dim = embed_dims * 2 if decouple_attn else embed_dims
    graph_model = MHAWrapper(gnn_dim, num_heads, dropout=dropout, batch_first=True)
    norm_layer = nn.LayerNorm(embed_dims)
    ffn = SimpleFFN(embed_dims, feedforward_dims, dropout=dropout)
    kps = SparseBox3DKeyPointsGenerator(embed_dims=embed_dims, num_learnable_pts=0, fix_scale=((0.0, 0.0, 0.0),))
    bev_agg = BEVFeatureAggregation(
        embed_dims=embed_dims,
        bev_bounds=bev_bounds,
        kps_generator=kps,
        proj_drop=dropout,
        residual_mode="add",
    )
    refine_layer = SparseBox3DRefinementModule(
        embed_dims=embed_dims,
        output_dim=11,
        num_cls=num_classes,
        refine_yaw=True,
        with_cls_branch=True,
        with_quality_estimation=True,
    )

    operation_order = (
        ["gnn", "norm", "deformable", "ffn", "norm", "refine"] * num_single_frame_decoder
        + ["gnn", "norm", "deformable", "ffn", "norm", "refine"] * (num_decoder - num_single_frame_decoder)
    )
    operation_order = operation_order[2:]

    sampler = None
    loss_cls = None
    loss_reg = None
    if use_dn:
        sampler = DenoisingSampler(
            num_dn_groups=num_dn_groups,
            dn_noise_scale=dn_noise_scale,
            max_dn_gt=max_dn_gt,
            add_neg_dn=add_neg_dn,
            reg_weights=reg_weights,
        )
    loss_cls = FocalLoss(alpha=0.25, gamma=2.0, ignore_index=num_classes)
    loss_reg = SparseBox3DLoss(reg_weights=reg_weights, loss_centerness=True, loss_yawness=True)

    decoder = None
    if use_decoder:
        decoder = SparseBox3DDecoder(
            num_output=decoder_num_output,
            score_threshold=decoder_score_threshold,
            sorted=True,
        )

    head = Sparse4DHead(
        instance_bank=instance_bank,
        anchor_encoder=anchor_encoder,
        graph_model=graph_model,
        norm_layer=norm_layer,
        ffn=ffn,
        deformable_model=bev_agg,
        refine_layer=refine_layer,
        operation_order=operation_order,
        num_single_frame_decoder=num_single_frame_decoder,
        decouple_attn=decouple_attn,
        sampler=sampler,
        decoder=decoder,
        loss_cls=loss_cls,
        loss_reg=loss_reg,
        reg_weights=reg_weights,
        gt_cls_key="gt_labels_3d",
        gt_reg_key="gt_bboxes_3d",
        cls_threshold_to_reg=0.05,
    )
    head.init_weights()
    return head


def build_track_head(
    feat_dim: int = 256,
    anchor_dim: int = 11,
    num_heads: int = 8,
    dropout: float = 0.1,
    embed_dim: int = 256,
):
    """构建跟踪头：输出 (B, T, N, N) 亲和矩阵，配合 track_affinity_loss 与 decode_track 使用。"""
    return TrackHead(
        feat_dim=feat_dim,
        anchor_dim=anchor_dim,
        num_heads=num_heads,
        dropout=dropout,
        embed_dim=embed_dim,
    )


def build_sparse4d_bev_model(
    bev_backbone: nn.Module,
    head: nn.Module,
    track_head: nn.Module = None,
):
    """将 BEV backbone、head、可选 track_head 拼成完整模型。"""

    class Sparse4DBEVModel(nn.Module):
        def __init__(self, bev_backbone, head, track_head=None):
            super().__init__()
            self.bev_backbone = bev_backbone
            self.head = head
            self.track_head = track_head

        def forward(self, x, rots, trans, intrins, distorts, post_rot3, post_tran3, theta_mats, T_ego_his2curs, metas=None):
            feature_maps = self.bev_backbone(x, rots, trans, intrins, distorts, post_rot3, post_tran3, theta_mats)
            head_out, seq_features, seq_anchors = self.head(feature_maps, metas)
            track_out = None
            if self.track_head is not None and T_ego_his2curs is not None:
                track_affinity = self.track_head(seq_features, seq_anchors, T_ego_his2curs)
                track_out = {"track_affinity": track_affinity}
                if self.training and metas is not None and metas.get("gt_track_match") is not None:
                    track_out["loss_track"] = track_affinity_loss(
                        track_affinity, metas["gt_track_match"], ignore_index=-1
                    )
                if not self.training:
                    track_ids, positions = decode_track(track_affinity, seq_anchors, use_hungarian=True)
                    track_out["track_ids"] = track_ids
                    track_out["track_positions"] = positions
            return head_out, seq_features, seq_anchors, track_out

    return Sparse4DBEVModel(bev_backbone, head, track_head)


if __name__ == "__main__":
    try:
        from .bev_backbone import *
    except ImportError:
        from bev_backbone import *
    

    grid_conf = {
        'xbound': [-80.0, 120.0, 1],
        'ybound': [-40.0, 40.0, 1],
        'zbound': [-2.0, 4.0, 1.0]
    }
    backbone = rebuild_backbone(grid_conf)
    head = build_sparse4d_bev_head(
        num_anchor=900,
        embed_dims=256,
        num_decoder=2,
        num_single_frame_decoder=2,
    )
    model = build_sparse4d_bev_model(backbone,head)
    T_lidar2camera = np.array([[
                    0.010424562939021826,
                    -0.99962866534536,
                    -0.024994439529313277,
                    -0.059431263039249574
                ],
                [
                    0.006448137058870251,
                    0.0250541141879979,
                    -0.9996632045797847,
                    1.744147092885818
                ],
                [
                    0.9999284955562123,
                    0.010257958549557083,
                    0.006708909083106572,
                    -1.9743571849555366
                ],
                [
                    0,
                    0,
                    0,
                    1
                ]]).reshape(4,4)
    dist_coeffs= np.array([[
                -0.034049232722088534,
                0.008472202135916486,
                0,
                0,
                -0.019044060806897155,
                0.009176172281889064,
                0,
                0
            ]]).reshape(1,8)
    camera_matrix=np.array([
                [
                    1901.8956477873126,
                    0,
                    1922.52081253954
                ],
                [
                    0,
                    1902.2429685532018,
                    1078.2301107170085
                ],
                [
                    0,
                    0,
                    1
                ]
            ]).reshape(3,3)
    ego_poses = [
        {
            "orientation": [
                0.7252925277030919,
                0.006445121526768708,
                0.004589807880389847,
                -0.688395339416375
            ],
            "position": [
                -16.00869568908437,
                45.379399067139,
                -0.543346973087456
            ]
        },
        {
            "orientation": [
                0.7254516385501073,
                0.00675402415715356,
                0.004536041108295809,
                -0.6882250559328055
            ],
            "position": [
                -16.00840644428952,
                45.38183205610659,
                -0.5425536577750744
            ]
        },
        {
            "orientation": [
                0.7256345746113962,
                0.00675949530897834,
                0.004310011176511773,
                -0.6880335724048092
            ],
            "position": [
                -16.007917008977827,
                45.383739493563155,
                -0.5423570234377117
            ]
        },
        {
            "orientation": [
                0.7257478117103694,
                0.006642957922078682,
                0.004081862107264589,
                -0.6879166543277585
            ],
            "position": [
                -16.0081965629427,
                45.38496794351029,
                -0.5429472288690743
            ]
        },
        {
            "orientation": [
                0.7257670195042981,
                0.0066346570824429504,
                0.004051102145859878,
                -0.6878966515270655
            ],
            "position": [
                -16.0086749159006,
                45.3839920241313,
                -0.5433879408122706
            ]
        },
        {
            "orientation": [
                0.7258077113155456,
                0.006781442667119014,
                0.00407377133102134,
                -0.6878521514810235
            ],
            "position": [
                -16.007274032017396,
                45.3843939298178,
                -0.5424020189052948
            ]
        },
        {
            "orientation": [
                0.7259736629900775,
                0.006877674717116843,
                0.004023298068784183,
                -0.6876763417429843
            ],
            "position": [
                -16.007786972618277,
                45.38313008782289,
                -0.5430163365819
            ]
        }
        ]

    bs = 2
    seq_len = 7 # 时序
    img_num = 1 # 环视

    T_ego_his2curs = egopose_alginhistory2current(ego_poses)
    theta_mats = gen_theta_mat(ego_poses)
    rots = torch.from_numpy(T_lidar2camera[:3,:3])[None,None,None].expand(bs,seq_len,img_num,3,3) # 1 1 3 3
    trans = torch.from_numpy(T_lidar2camera[:3,3])[None,None,None,None].expand(bs,seq_len,img_num,1,3) # 1 1 1 3
    intrins = torch.from_numpy(camera_matrix)[None,None,None].expand(bs,seq_len,img_num,3,3) # 1 1 1 3
    distorts = torch.from_numpy(dist_coeffs)[None,None,None].expand(bs,seq_len,img_num,1,8) # 1 1 1 3
    post_tran3 = torch.zeros(3)
    post_rot3 = torch.eye(3)
    post_tran3 = post_tran3[None,None,None,None].expand(bs,seq_len,img_num,1,3)
    post_rot3 = post_rot3[None,None,None].expand(bs,seq_len,img_num,3,3)
    theta_mats = torch.from_numpy(theta_mats[None]).expand(bs,seq_len,2,3)
    T_ego_his2curs = torch.from_numpy(T_ego_his2curs[None]).expand(bs,seq_len,4,4)

    x_input = torch.zeros([bs,seq_len,img_num,3,128,384])
    out,seq_features,seq_anchors = model(x_input,rots, trans, intrins, distorts, post_rot3, post_tran3,theta_mats,T_ego_his2curs,metas={})
    print("seq_features shape:", seq_features.shape)
    print("seq_anchors shape:", seq_anchors.shape)

    # 快速前向测试（请用: cd projects && python -m sparse4d_bev_torch.model）
    # B, S, C, H, W = 2, 7, 256, 200, 80
    
    # bev = torch.randn(B ,S, C, H, W)
    # out = head([bev], metas={})
    print("classification len:", len(out["classification"]))
    print("prediction[0] shape:", out["prediction"][0].shape)
    print("quality[0] shape:", out["quality"][0].shape if out["quality"][0] is not None else None)
