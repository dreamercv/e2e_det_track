"""
去噪训练：DenoisingSampler 纯 PyTorch 实现。
提供 get_dn_anchors（生成噪声 anchor + target）和 sample（正常预测与 GT 的匈牙利匹配）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment

from .box3d import X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX

__all__ = ["DenoisingSampler"]


class DenoisingSampler(nn.Module):
    """去噪采样器：生成 dn_metas（get_dn_anchors）与正常匹配（sample）。"""

    def __init__(
        self,
        num_dn_groups: int = 10,
        dn_noise_scale: float = 0.5,
        max_dn_gt: int = 32,
        add_neg_dn: bool = True,
        num_temp_dn_groups: int = 0,
        reg_weights=None,
        cls_weight: float = 2.0,
        box_weight: float = 0.25,
        alpha: float = 0.25,
        gamma: float = 2.0,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.num_dn_groups = num_dn_groups
        self.dn_noise_scale = dn_noise_scale
        self.max_dn_gt = max_dn_gt
        self.add_neg_dn = add_neg_dn
        self.num_temp_dn_groups = num_temp_dn_groups
        self.dn_metas = None
        self.reg_weights = reg_weights if reg_weights is not None else [1.0] * 8 + [0.0] * 3  # 11 dims
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def encode_reg_target(self, box_target, device=None):
        """将 GT 框 (x,y,z,w,l,h,yaw,vx,vy,vz) 编码为 11 维：(x,y,z,log(w),log(l),log(h),sin(yaw),cos(yaw),vx,vy,vz)。"""
        state_dims = 11
        outputs = []
        for box in box_target:
            if box.numel() == 0:
                outputs.append(box.new_zeros(0, state_dims))
                continue
            if box.shape[-1] >= 10:
                xyz = box[..., :3]
                whl = box[..., 3:6].clamp(min=1e-4).log()
                yaw = box[..., 6]
                sin_yaw = torch.sin(yaw).unsqueeze(-1)
                cos_yaw = torch.cos(yaw).unsqueeze(-1)
                vel = box[..., 7:10] if box.shape[-1] >= 10 else box.new_zeros(*box.shape[:-1], 3)
                out = torch.cat([xyz, whl, sin_yaw, cos_yaw, vel], dim=-1)
            else:
                out = box
            if out.shape[-1] < state_dims:
                out = F.pad(out, (0, state_dims - out.shape[-1]), value=0)
            if device is not None:
                out = out.to(device=device)
            outputs.append(out)
        return outputs

    def _cls_cost(self, cls_pred, cls_target):
        """Focal-style cost: pos_cost - neg_cost per class."""
        bs = cls_pred.shape[0]
        cls_pred = cls_pred.sigmoid()
        cost = []
        for i in range(bs):
            if len(cls_target[i]) > 0:
                neg = (
                    -(1 - cls_pred[i] + self.eps).log()
                    * (1 - self.alpha)
                    * cls_pred[i].pow(self.gamma)
                )
                pos = (
                    -(cls_pred[i] + self.eps).log()
                    * self.alpha
                    * (1 - cls_pred[i]).pow(self.gamma)
                )
                cost.append((pos[:, cls_target[i]] - neg[:, cls_target[i]]) * self.cls_weight)
            else:
                cost.append(None)
        return cost

    def _box_cost(self, box_pred, box_target, reg_weights):
        """
        L1 cost with reg_weights. 与官方 target.py 一致。
        box_target: list of (N_i, D)（sample 用）或 3D tensor (bs, num_gt, D)（get_dn_anchors 用）。
        统一按最后一维 D = min(pred_dim, tgt_dim) 切片，避免 10/11 混用报错。
        """
        bs = box_pred.shape[0]
        is_tensor = isinstance(box_target, torch.Tensor)
        cost = []
        for i in range(bs):
            tgt = box_target[i]
            n_tgt = tgt.shape[0] if is_tensor else len(tgt)
            if n_tgt > 0:
                D = min(int(box_pred.shape[-1]), int(tgt.shape[-1]))
                pred_i = box_pred[i, :, :D].contiguous()
                tgt_i = tgt[:, :D].contiguous()
                diff = torch.abs(pred_i[:, None, :] - tgt_i[None, :, :])
                w = reg_weights[i] if isinstance(reg_weights, (list, tuple)) else reg_weights[i]
                w = w[..., :D]
                if w.dim() == 1:
                    w = w.unsqueeze(0).unsqueeze(0).expand(1, n_tgt, D)
                else:
                    w = w.unsqueeze(0)
                if w.shape[-1] < D:
                    w = F.pad(w, (0, D - w.shape[-1]), value=0)
                reg_w = box_pred.new_tensor(self.reg_weights[:D])
                if reg_w.shape[-1] < D:
                    reg_w = F.pad(reg_w, (0, D - reg_w.shape[-1]), value=0)
                reg_w = reg_w[None, None, :]
                c = (diff * w * reg_w).sum(dim=-1) * self.box_weight
                cost.append(c)
            else:
                cost.append(None)
        return cost

    def sample(self, cls_pred, box_pred, cls_target, box_target):
        """
        正常预测与 GT 的匈牙利匹配。
        cls_target / box_target: list of tensor，每个样本的 GT 类别与框（已编码 11 维）。
        Returns: output_cls_target (bs, num_pred), output_box_target (bs, num_pred, 11), output_reg_weights (bs, num_pred, 11)
        """
        bs, num_pred, num_cls = cls_pred.shape
        box_target = self.encode_reg_target(box_target, cls_pred.device)
        cls_cost = self._cls_cost(cls_pred, cls_target)
        reg_weights = [
            torch.logical_not(box_target[i].isnan()).to(dtype=box_target[i].dtype)
            for i in range(len(box_target))
        ]
        box_cost = self._box_cost(box_pred, box_target, reg_weights)

        indices = []
        for i in range(bs):
            if cls_cost[i] is not None and box_cost[i] is not None:
                cost = (cls_cost[i] + box_cost[i]).detach().cpu().numpy()
                cost = np.where(np.isneginf(cost) | np.isnan(cost), 1e8, cost)
                pred_idx, target_idx = linear_sum_assignment(cost)
                indices.append((
                    cls_pred.new_tensor(pred_idx, dtype=torch.int64),
                    cls_pred.new_tensor(target_idx, dtype=torch.int64),
                ))
            else:
                indices.append((None, None))

        output_cls_target = cls_target[0].new_ones(bs, num_pred, dtype=torch.long) * num_cls
        output_box_target = box_pred.new_zeros(box_pred.shape)
        output_reg_weights = box_pred.new_zeros(box_pred.shape)
        for i, (pred_idx, target_idx) in enumerate(indices):
            if pred_idx is None or len(cls_target[i]) == 0:
                continue
            output_cls_target[i, pred_idx] = cls_target[i][target_idx]
            output_box_target[i, pred_idx] = box_target[i][target_idx]
            output_reg_weights[i, pred_idx] = reg_weights[i][target_idx]
        return output_cls_target, output_box_target, output_reg_weights

    def get_dn_anchors(self, cls_target, box_target, gt_instance_id=None):
        """
        生成去噪 anchor 与 target。
        cls_target: list of (N_gt,) 类别；box_target: list of (N_gt, 10) 或 (N_gt, 11)  decoded 框。
        Returns: (dn_anchor, dn_reg_target, dn_cls_target, dn_attn_mask, valid_mask, dn_id_target)
        """
        if self.num_dn_groups <= 0:
            return None
        if self.num_temp_dn_groups <= 0:
            gt_instance_id = None

        if self.max_dn_gt > 0:
            cls_target = [x[: self.max_dn_gt] for x in cls_target]
            box_target = [x[: self.max_dn_gt] for x in box_target]
            if gt_instance_id is not None:
                gt_instance_id = [x[: self.max_dn_gt] for x in gt_instance_id]

        max_dn_gt = max(len(x) for x in cls_target)
        if max_dn_gt == 0:
            return None

        cls_target = torch.stack([
            F.pad(x, (0, max_dn_gt - x.shape[0]), value=-1)
            for x in cls_target
        ])
        box_target = self.encode_reg_target(box_target, cls_target.device)
        state_dims = 11
        box_target = [
            F.pad(x, (0, max(0, state_dims - x.shape[-1])), value=0) if x.numel() > 0 else x.new_zeros(0, state_dims)
            for x in box_target
        ]
        box_target = torch.stack([
            F.pad(x, (0, 0, 0, max_dn_gt - x.shape[0]))
            for x in box_target
        ])
        state_dims = 11
        if box_target.shape[-1] < state_dims:
            box_target = F.pad(box_target, (0, state_dims - box_target.shape[-1]), value=0)
        elif box_target.shape[-1] > state_dims:
            box_target = box_target[..., :state_dims]
        box_target = torch.where(
            cls_target[..., None] == -1,
            box_target.new_tensor(0.0),
            box_target,
        )
        if gt_instance_id is not None:
            gt_instance_id = torch.stack([
                F.pad(x, (0, max_dn_gt - x.shape[0]), value=-1)
                for x in gt_instance_id
            ])

        bs, num_gt, _ = box_target.shape
        if self.num_dn_groups > 1:
            cls_target = cls_target.repeat(self.num_dn_groups, 1)
            box_target = box_target.repeat(self.num_dn_groups, 1, 1)
            if gt_instance_id is not None:
                gt_instance_id = gt_instance_id.repeat(self.num_dn_groups, 1)

        noise = (torch.rand_like(box_target) * 2 - 1) * box_target.new_tensor(self.dn_noise_scale)
        dn_anchor = box_target + noise
        if self.add_neg_dn:
            noise_neg = (torch.rand_like(box_target) + 1) * torch.where(
                torch.rand_like(box_target) > 0.5,
                box_target.new_tensor(1.0),
                box_target.new_tensor(-1.0),
            ) * box_target.new_tensor(self.dn_noise_scale)
            dn_anchor = torch.cat([dn_anchor, box_target + noise_neg], dim=1)
            num_gt *= 2
            cls_target = torch.cat([cls_target, cls_target], dim=1)
            box_target = torch.cat([box_target, box_target], dim=1)
            if gt_instance_id is not None:
                gt_instance_id = torch.cat([gt_instance_id, gt_instance_id], dim=1)

        # 与官方一致：dn_anchor 与 box_target 同形，直接传 3D tensor 给 _box_cost
        D = max(dn_anchor.shape[-1], box_target.shape[-1], state_dims)
        if dn_anchor.shape[-1] < D:
            dn_anchor = F.pad(dn_anchor, (0, D - dn_anchor.shape[-1]), value=0)
        else:
            dn_anchor = dn_anchor[..., :D]
        if box_target.shape[-1] < D:
            box_target = F.pad(box_target, (0, D - box_target.shape[-1]), value=0)
        else:
            box_target = box_target[..., :D]
        dn_anchor = dn_anchor.contiguous()
        box_target = box_target.contiguous()
        box_cost = self._box_cost(
            dn_anchor,
            box_target,
            torch.ones_like(box_target),
        )
        dn_box_target = torch.zeros_like(dn_anchor)
        dn_cls_target = (-torch.ones_like(cls_target) * 3).long()
        if gt_instance_id is not None:
            dn_id_target = -torch.ones_like(gt_instance_id).long()
        else:
            dn_id_target = None

        for i in range(dn_anchor.shape[0]):
            cost = box_cost[i].cpu().numpy()
            anchor_idx, gt_idx = linear_sum_assignment(cost)
            anchor_idx = dn_anchor.new_tensor(anchor_idx, dtype=torch.int64)
            gt_idx = dn_anchor.new_tensor(gt_idx, dtype=torch.int64)
            dn_box_target[i, anchor_idx] = box_target[i, gt_idx]
            dn_cls_target[i, anchor_idx] = cls_target[i, gt_idx]
            if gt_instance_id is not None:
                dn_id_target[i, anchor_idx] = gt_instance_id[i, gt_idx]

        dn_anchor = dn_anchor.reshape(
            self.num_dn_groups, bs, num_gt, state_dims
        ).permute(1, 0, 2, 3).reshape(bs, -1, state_dims)
        dn_box_target = dn_box_target.reshape(
            self.num_dn_groups, bs, num_gt, state_dims
        ).permute(1, 0, 2, 3).reshape(bs, -1, state_dims)
        dn_cls_target = dn_cls_target.reshape(
            self.num_dn_groups, bs, num_gt
        ).permute(1, 0, 2).reshape(bs, -1)
        if dn_id_target is not None:
            dn_id_target = dn_id_target.reshape(
                self.num_dn_groups, bs, num_gt
            ).permute(1, 0, 2).reshape(bs, -1)

        valid_mask = dn_cls_target >= 0
        num_gt_per_group = num_gt
        total_dn = num_gt_per_group * self.num_dn_groups
        attn_mask = dn_box_target.new_ones(total_dn, total_dn).bool()
        for i in range(self.num_dn_groups):
            start = num_gt_per_group * i
            end = start + num_gt_per_group
            attn_mask[start:end, start:end] = False

        return (
            dn_anchor,
            dn_box_target,
            dn_cls_target,
            attn_mask,
            valid_mask,
            dn_id_target,
        )

    def cache_dn(self, dn_instance_feature, dn_anchor, dn_cls_target, valid_mask, dn_id_target):
        """可选：缓存当前 dn 用于时序去噪，本实现暂不启用。"""
        pass
