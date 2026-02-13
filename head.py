"""
Sparse4DHead 纯 PyTorch 实现：接收 BEV 特征，输出 instance_feature / anchor / classification 等。
不依赖 mmcv/mmdet，便于调试开发。
"""
from typing import List, Optional, Union

import torch
import torch.nn as nn

from .instance_bank import InstanceBank
from .detection3d_blocks import SparseBox3DEncoder, SparseBox3DRefinementModule, SparseBox3DKeyPointsGenerator

__all__ = ["Sparse4DHead", "MHAWrapper", "SimpleFFN"]


class MHAWrapper(nn.Module):
    """包装 nn.MultiheadAttention，batch_first=True，forward 返回单 tensor，便于 head 调用。"""

    def __init__(self, embed_dims: int, num_heads: int, dropout: float = 0.0, batch_first: bool = True):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dims, num_heads, dropout=dropout, batch_first=batch_first)

    def forward(self, query, key=None, value=None, attn_mask=None, key_padding_mask=None):
        if key is None:
            key = query
        if value is None:
            value = query
        out, _ = self.mha(query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return out


class SimpleFFN(nn.Module):
    """两层 Linear + ReLU + residual，与 AsymmetricFFN 简化版一致。"""

    def __init__(self, embed_dims=256, feedforward_dims=1024, dropout=0.0):
        super().__init__()
        self.linear1 = nn.Linear(embed_dims, feedforward_dims)
        self.linear2 = nn.Linear(feedforward_dims, embed_dims)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.linear2(self.dropout(self.act(self.linear1(x))))
        return x + self.dropout(out)


class Sparse4DHead(nn.Module):
    """
    纯 PyTorch Sparse4D Head，接 BEV 特征。
    所有子模块由外部构造后传入，无 mmcv build_from_cfg。
    """

    def __init__(
        self,
        instance_bank: InstanceBank,
        anchor_encoder: nn.Module,
        graph_model: nn.Module,
        norm_layer: nn.Module,
        ffn: nn.Module,
        deformable_model: nn.Module,
        refine_layer: nn.Module,
        operation_order: List[str],
        num_single_frame_decoder: int = 5,
        decouple_attn: bool = True,
        sampler=None,
        decoder=None,
        loss_cls=None,
        loss_reg=None,
        reg_weights: Optional[List[float]] = None,
        gt_cls_key: str = "gt_labels_3d",
        gt_reg_key: str = "gt_bboxes_3d",
        cls_threshold_to_reg: float = -1,
    ):
        super().__init__()
        self.instance_bank = instance_bank
        self.anchor_encoder = anchor_encoder
        self.sampler = sampler
        self.decoder = decoder
        self.loss_cls = loss_cls
        self.loss_reg = loss_reg
        self.reg_weights = reg_weights if reg_weights is not None else [1.0] * 11
        self.gt_cls_key = gt_cls_key
        self.gt_reg_key = gt_reg_key
        self.cls_threshold_to_reg = cls_threshold_to_reg
        self.num_single_frame_decoder = num_single_frame_decoder
        self.decouple_attn = decouple_attn
        self.embed_dims = instance_bank.embed_dims

        if decouple_attn:
            self.fc_before = nn.Linear(self.embed_dims, self.embed_dims * 2, bias=False)
            self.fc_after = nn.Linear(self.embed_dims * 2, self.embed_dims, bias=False)
        else:
            self.fc_before = nn.Identity()
            self.fc_after = nn.Identity()

        self.operation_order = operation_order
        layers = []
        for op in operation_order:
            if op == "gnn":
                layers.append(graph_model)
            elif op == "norm":
                layers.append(norm_layer)
            elif op == "ffn":
                layers.append(ffn)
            elif op == "deformable":
                layers.append(deformable_model)
            elif op == "refine":
                layers.append(refine_layer)
            else:
                raise ValueError(f"Unknown op: {op}")
        self.layers = nn.ModuleList(layers)

    def init_weights(self):
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            if op != "refine":
                for p in self.layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def _graph_forward(self, index, query, key=None, value=None, query_pos=None, key_pos=None, attn_mask=None):
        if value is not None:
            value = self.fc_before(value)
        if key is None:
            key = value
        elif self.decouple_attn and key.shape[-1] != value.shape[-1]:
            key = self.fc_before(key)
        if self.decouple_attn:
            if query_pos is not None:
                query = torch.cat([query, query_pos], dim=-1)
            if key_pos is not None and key is not None:
                key = torch.cat([key, key_pos], dim=-1)
        layer = self.layers[index]
        out = layer(query, key, value, attn_mask=attn_mask)
        if isinstance(out, tuple):
            out = out[0]
        return self.fc_after(out)

    def forward(
        self,
        feature_maps: Union[torch.Tensor, List[torch.Tensor]],
        metas: Optional[dict] = None,
    ):
        if isinstance(feature_maps, torch.Tensor):
            feature_maps = [feature_maps]
        batch_size = feature_maps[0].shape[0]
        if metas is None:
            metas = {}

        dn_metas = getattr(self.sampler, "dn_metas", None)
        if dn_metas is not None and self.sampler.dn_metas["dn_anchor"].shape[0] != batch_size:
            self.sampler.dn_metas = None
            dn_metas = None

        (
            instance_feature,
            anchor,
            _,
            _,
            time_interval,
        ) = self.instance_bank.get(batch_size, metas, dn_metas=dn_metas)

        attn_mask = None
        dn_metas = None
        num_free_instance = instance_feature.shape[1]
        if self.training and self.sampler is not None and hasattr(self.sampler, "get_dn_anchors"):
            gt_cls = metas.get("gt_labels_3d")
            gt_reg = metas.get("gt_bboxes_3d")
            if gt_cls is not None and gt_reg is not None:
                dn_metas = self.sampler.get_dn_anchors(gt_cls, gt_reg, None)
        if dn_metas is not None:
            (dn_anchor, dn_reg_target, dn_cls_target, dn_attn_mask, valid_mask, dn_id_target) = dn_metas
            num_dn_anchor = dn_anchor.shape[1]
            if dn_anchor.shape[-1] != anchor.shape[-1]:
                remain = anchor.shape[-1] - dn_anchor.shape[-1]
                dn_anchor = torch.cat([
                    dn_anchor,
                    dn_anchor.new_zeros(batch_size, num_dn_anchor, remain),
                ], dim=-1)
            anchor = torch.cat([anchor, dn_anchor], dim=1)
            instance_feature = torch.cat([
                instance_feature,
                instance_feature.new_zeros(batch_size, num_dn_anchor, instance_feature.shape[-1]),
            ], dim=1)
            num_free_instance = instance_feature.shape[1] - num_dn_anchor
            attn_mask = anchor.new_ones(instance_feature.shape[1], instance_feature.shape[1], dtype=torch.bool)
            attn_mask[:num_free_instance, :num_free_instance] = False
            attn_mask[num_free_instance:, num_free_instance:] = dn_attn_mask

        anchor_embed = self.anchor_encoder(anchor)

        prediction = []
        classification = []
        quality = []
        
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            if op == "gnn":
                instance_feature = self._graph_forward(
                    i, instance_feature, instance_feature, instance_feature,
                    query_pos=anchor_embed, attn_mask=attn_mask,
                )
            elif op in ("norm", "ffn"):
                instance_feature = self.layers[i](instance_feature)
            elif op == "deformable":
                instance_feature = self.layers[i](
                    instance_feature, anchor, anchor_embed, feature_maps, metas
                )
            elif op == "refine":
                time_int = time_interval if isinstance(time_interval, torch.Tensor) else anchor.new_tensor(time_interval)
                anchor, cls, qt = self.layers[i](
                    instance_feature, anchor, anchor_embed,
                    time_interval=time_int,
                    return_cls=(
                        self.training
                        or len(prediction) == self.num_single_frame_decoder - 1
                        or i == len(self.operation_order) - 1
                    ),
                )
                prediction.append(anchor)
                classification.append(cls)
                quality.append(qt)
                if i != len(self.operation_order) - 1:
                    anchor_embed = self.anchor_encoder(anchor)
            else:
                raise NotImplementedError(op)
        
        
        

        if dn_metas is not None:
            dn_classification = [c[:, num_free_instance:] for c in classification]
            dn_prediction = [p[:, num_free_instance:] for p in prediction]
            dn_quality = [q[:, num_free_instance:] if q is not None else None for q in quality]
            classification = [c[:, :num_free_instance] for c in classification]
            prediction = [p[:, :num_free_instance] for p in prediction]
            quality = [q[:, :num_free_instance] if q is not None else None for q in quality]
            instance_feature = instance_feature[:, :num_free_instance]
            anchor = anchor[:, :num_free_instance]
            cls = cls[:, :num_free_instance]
            out = {
                "classification": classification,
                "prediction": prediction,
                "quality": quality,
                "dn_prediction": dn_prediction,
                "dn_classification": dn_classification,
                "dn_quality": dn_quality,
                "dn_reg_target": dn_reg_target,
                "dn_cls_target": dn_cls_target,
                "dn_valid_mask": valid_mask,
            }
        else:
            out = {"classification": classification, "prediction": prediction, "quality": quality}

        final_feature = instance_feature.reshape(-1, 7, *instance_feature.shape[1:])
        final_anchor = anchor.reshape(-1, 7, *anchor.shape[1:])

        if not self.training and self.decoder is not None and hasattr(self.decoder, "score_threshold"):
            out["instance_id"] = self.instance_bank.get_instance_id(
                cls, anchor, self.decoder.score_threshold
            )
        return out,final_feature,final_anchor

    def _reduce_mean(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum() / max(x.numel(), 1)

    def prepare_for_dn_loss(self, model_outs, prefix=""):
        dn_valid_mask = model_outs[f"{prefix}dn_valid_mask"].flatten(end_dim=1)
        dn_cls_target = model_outs[f"{prefix}dn_cls_target"].flatten(end_dim=1)[dn_valid_mask]
        dn_reg_target = model_outs[f"{prefix}dn_reg_target"].flatten(end_dim=1)[dn_valid_mask][
            ..., : len(self.reg_weights)
        ]
        dn_pos_mask = dn_cls_target >= 0
        dn_reg_target = dn_reg_target[dn_pos_mask]
        reg_weights = dn_reg_target.new_tensor(self.reg_weights)[None].tile(dn_reg_target.shape[0], 1)
        num_dn_pos = max(
            self._reduce_mean(dn_valid_mask.to(dtype=dn_reg_target.dtype)).item(),
            1.0,
        )
        dn_quality_pos = None
        if f"{prefix}dn_quality" in model_outs:
            q_list = model_outs[f"{prefix}dn_quality"]
            dn_quality_pos = []
            for q in q_list:
                if q is None:
                    dn_quality_pos.append(None)
                    continue
                q_flat = q.flatten(end_dim=1)[dn_valid_mask][dn_pos_mask]
                dn_quality_pos.append(q_flat)
        return (
            dn_valid_mask,
            dn_cls_target,
            dn_reg_target,
            dn_pos_mask,
            reg_weights,
            num_dn_pos,
            dn_quality_pos,
        )

    def loss(self, model_outs, data, feature_maps=None):
        if self.loss_cls is None or self.loss_reg is None or self.sampler is None:
            return {}
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        quality = model_outs["quality"]
        output = {}
        gt_cls = data.get(self.gt_cls_key)
        gt_reg = data.get(self.gt_reg_key)
        for decoder_idx, (cls, reg, qt) in enumerate(zip(cls_scores, reg_preds, quality)):
            reg = reg[..., : len(self.reg_weights)]
            cls_target, reg_target, reg_weights = self.sampler.sample(
                cls, reg, gt_cls, gt_reg
            )
            reg_target = reg_target[..., : len(self.reg_weights)]
            mask = torch.logical_not(torch.all(reg_target == 0, dim=-1))
            num_pos = max(self._reduce_mean(torch.sum(mask).to(dtype=reg.dtype)), 1.0)
            if self.cls_threshold_to_reg > 0:
                mask = torch.logical_and(
                    mask, cls.max(dim=-1).values.sigmoid() > self.cls_threshold_to_reg
                )
            cls_flat = cls.flatten(end_dim=1)
            cls_target_flat = cls_target.flatten(end_dim=1)
            cls_loss = self.loss_cls(cls_flat, cls_target_flat, avg_factor=num_pos)
            mask_flat = mask.reshape(-1)
            reg_weights_flat = (reg_weights * reg.new_tensor(self.reg_weights)).flatten(end_dim=1)[mask_flat]
            reg_target_flat = reg_target.flatten(end_dim=1)[mask_flat]
            reg_flat = reg.flatten(end_dim=1)[mask_flat]
            reg_target_flat = torch.where(reg_target_flat.isnan(), reg.new_tensor(0.0), reg_target_flat)
            cls_target_masked = cls_target_flat[mask_flat]
            qt_masked = qt.flatten(end_dim=1)[mask_flat] if qt is not None else None
            reg_loss_dict = self.loss_reg(
                reg_flat,
                reg_target_flat,
                weight=reg_weights_flat,
                avg_factor=num_pos,
                suffix=f"_{decoder_idx}",
                quality=qt_masked,
                cls_target=cls_target_masked,
            )
            output[f"loss_cls_{decoder_idx}"] = cls_loss
            output.update(reg_loss_dict)

        if "dn_prediction" not in model_outs:
            return output

        (
            dn_valid_mask,
            dn_cls_target,
            dn_reg_target,
            dn_pos_mask,
            reg_weights,
            num_dn_pos,
            dn_quality_pos,
        ) = self.prepare_for_dn_loss(model_outs)
        dn_cls_scores = model_outs["dn_classification"]
        dn_reg_preds = model_outs["dn_prediction"]
        for decoder_idx, (cls, reg) in enumerate(zip(dn_cls_scores, dn_reg_preds)):
            reg = reg[..., : len(self.reg_weights)]
            cls_loss_dn = self.loss_cls(
                cls.flatten(end_dim=1)[dn_valid_mask],
                dn_cls_target,
                avg_factor=num_dn_pos,
            )
            qt_dn = None
            if dn_quality_pos is not None and decoder_idx < len(dn_quality_pos):
                qt_dn = dn_quality_pos[decoder_idx]
            reg_loss_dict = self.loss_reg(
                reg.flatten(end_dim=1)[dn_valid_mask][dn_pos_mask],
                dn_reg_target,
                avg_factor=num_dn_pos,
                weight=reg_weights,
                suffix=f"_dn_{decoder_idx}",
                quality=qt_dn,
            )
            output[f"loss_cls_dn_{decoder_idx}"] = cls_loss_dn
            output.update(reg_loss_dict)
        return output
