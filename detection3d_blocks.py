"""
SparseBox3D Encoder / Refinement / KeyPointsGenerator 纯 PyTorch 实现。
"""
import torch
import torch.nn as nn

from .box3d import X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, VY, VZ

__all__ = ["SparseBox3DEncoder", "SparseBox3DRefinementModule", "SparseBox3DKeyPointsGenerator"]


def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(nn.Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers


class SparseBox3DEncoder(nn.Module):
    def __init__(
        self,
        embed_dims,
        vel_dims=3,
        mode="add",
        output_fc=True,
        in_loops=1,
        out_loops=2,
    ):
        super().__init__()
        assert mode in ["add", "cat"]
        self.embed_dims = embed_dims
        self.vel_dims = vel_dims
        self.mode = mode

        def emb_layer(idim, odim):
            return nn.Sequential(*linear_relu_ln(odim, in_loops, out_loops, idim))

        if not isinstance(embed_dims, (list, tuple)):
            embed_dims = [embed_dims] * 5
        self.pos_fc = emb_layer(3, embed_dims[0])
        self.size_fc = emb_layer(3, embed_dims[1])
        self.yaw_fc = emb_layer(2, embed_dims[2])
        if vel_dims > 0:
            self.vel_fc = emb_layer(self.vel_dims, embed_dims[3])
        self.output_fc = emb_layer(embed_dims[-1], embed_dims[-1]) if output_fc else None

    def forward(self, box_3d: torch.Tensor):
        pos_feat = self.pos_fc(box_3d[..., [X, Y, Z]])
        size_feat = self.size_fc(box_3d[..., [W, L, H]])
        yaw_feat = self.yaw_fc(box_3d[..., [SIN_YAW, COS_YAW]])
        if self.mode == "add":
            out = pos_feat + size_feat + yaw_feat
        else:
            out = torch.cat([pos_feat, size_feat, yaw_feat], dim=-1)
        if self.vel_dims > 0:
            vel_feat = self.vel_fc(box_3d[..., VX : VX + self.vel_dims])
            out = out + vel_feat if self.mode == "add" else torch.cat([out, vel_feat], dim=-1)
        if self.output_fc is not None:
            out = self.output_fc(out)
        return out


class Scale(nn.Module):
    def __init__(self, scales):
        super().__init__()
        self.scales = nn.Parameter(torch.tensor(scales, dtype=torch.float32))

    def forward(self, x):
        return x * self.scales


def bias_init_with_prob(prior_prob):
    return -torch.log(torch.tensor(1 - prior_prob) / prior_prob).item()


class SparseBox3DRefinementModule(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        output_dim=11,
        num_cls=10,
        normalize_yaw=False,
        refine_yaw=False,
        with_cls_branch=True,
        with_quality_estimation=False,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.output_dim = output_dim
        self.num_cls = num_cls
        self.normalize_yaw = normalize_yaw
        self.refine_yaw = refine_yaw
        self.refine_state = [X, Y, Z, W, L, H]
        if self.refine_yaw:
            self.refine_state += [SIN_YAW, COS_YAW]

        self.layers = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2),
            nn.Linear(embed_dims, self.output_dim),
            Scale([1.0] * self.output_dim),
        )
        self.with_cls_branch = with_cls_branch
        if with_cls_branch:
            self.cls_layers = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 2),
                nn.Linear(embed_dims, self.num_cls),
            )
        self.with_quality_estimation = with_quality_estimation
        if with_quality_estimation:
            self.quality_layers = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 2),
                nn.Linear(embed_dims, 2),
            )

    def init_weight(self):
        if self.with_cls_branch:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.cls_layers[-1].bias, bias_init)

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        time_interval: torch.Tensor = None,
        return_cls=True,
    ):
        feature = instance_feature + anchor_embed
        output = self.layers(feature)
        output[..., self.refine_state] = output[..., self.refine_state] + anchor[..., self.refine_state]
        if self.normalize_yaw:
            output[..., [SIN_YAW, COS_YAW]] = F.normalize(output[..., [SIN_YAW, COS_YAW]], dim=-1)
        if self.output_dim > 8 and time_interval is not None:
            if not isinstance(time_interval, torch.Tensor):
                time_interval = instance_feature.new_tensor(time_interval)
            translation = output[..., VX:].transpose(0, -1)
            velocity = (translation / time_interval).transpose(0, -1)
            output = output.clone()
            output[..., VX:] = velocity + anchor[..., VX:]

        cls = self.cls_layers(instance_feature) if return_cls and self.with_cls_branch else None
        quality = self.quality_layers(feature) if return_cls and self.with_quality_estimation else None
        return output, cls, quality


class SparseBox3DKeyPointsGenerator(nn.Module):
    def __init__(self, embed_dims=256, num_learnable_pts=0, fix_scale=None):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_learnable_pts = num_learnable_pts
        if fix_scale is None:
            fix_scale = ((0.0, 0.0, 0.0),)
        self.fix_scale = nn.Parameter(torch.tensor(fix_scale, dtype=torch.float32), requires_grad=False)
        self.num_pts = len(self.fix_scale) + num_learnable_pts
        if num_learnable_pts > 0:
            self.learnable_fc = nn.Linear(embed_dims, num_learnable_pts * 3)

    def init_weight(self):
        if self.num_learnable_pts > 0:
            nn.init.xavier_uniform_(self.learnable_fc.weight)
            if self.learnable_fc.bias is not None:
                nn.init.zeros_(self.learnable_fc.bias)

    def forward(
        self,
        anchor: torch.Tensor,
        instance_feature=None,
        T_cur2temp_list=None,
        cur_timestamp=None,
        temp_timestamps=None,
    ):
        bs, num_anchor = anchor.shape[:2]
        size = anchor[..., None, [W, L, H]].exp()
        key_points = self.fix_scale.to(anchor.device) * size
        if self.num_learnable_pts > 0 and instance_feature is not None:
            learnable_scale = (
                self.learnable_fc(instance_feature)
                .reshape(bs, num_anchor, self.num_learnable_pts, 3)
                .sigmoid()
                - 0.5
            )
            key_points = torch.cat([key_points, learnable_scale * size], dim=-2)

        rotation_mat = anchor.new_zeros(bs, num_anchor, 3, 3)
        rotation_mat[:, :, 0, 0] = anchor[:, :, COS_YAW]
        rotation_mat[:, :, 0, 1] = -anchor[:, :, SIN_YAW]
        rotation_mat[:, :, 1, 0] = anchor[:, :, SIN_YAW]
        rotation_mat[:, :, 1, 1] = anchor[:, :, COS_YAW]
        rotation_mat[:, :, 2, 2] = 1
        key_points = (rotation_mat[:, :, None] @ key_points[..., None]).squeeze(-1)
        key_points = key_points + anchor[..., None, [X, Y, Z]]
        return key_points

    @staticmethod
    def anchor_projection(anchor, T_src2dst_list, src_timestamp=None, dst_timestamps=None, time_intervals=None):
        dst_anchors = []
        for i in range(len(T_src2dst_list)):
            vel = anchor[..., VX:]
            vel_dim = vel.shape[-1]
            T_src2dst = T_src2dst_list[i].to(dtype=anchor.dtype).unsqueeze(1)
            center = anchor[..., [X, Y, Z]]
            if time_intervals is not None:
                time_interval = time_intervals[i]
            elif src_timestamp is not None and dst_timestamps is not None:
                time_interval = (src_timestamp - dst_timestamps[i]).to(dtype=vel.dtype)
            else:
                time_interval = None
            if time_interval is not None:
                translation = vel.transpose(0, -1) * time_interval
                translation = translation.transpose(0, -1)
                center = center - translation
            center = (T_src2dst[..., :3, :3] @ center[..., None]).squeeze(-1) + T_src2dst[..., :3, 3]
            size = anchor[..., [W, L, H]]
            yaw = (T_src2dst[..., :2, :2] @ anchor[..., [COS_YAW, SIN_YAW]].unsqueeze(-1)).squeeze(-1)
            vel = (T_src2dst[..., :vel_dim, :vel_dim] @ vel.unsqueeze(-1)).squeeze(-1)
            dst_anchor = torch.cat([center, size, yaw, vel], dim=-1)
            dst_anchors.append(dst_anchor)
        return dst_anchors
