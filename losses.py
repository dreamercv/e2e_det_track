"""
纯 PyTorch 检测 loss：分类（Focal/BCE）+ 回归（L1，可选 quality）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .box3d import X, Y, Z, SIN_YAW, COS_YAW, CNS, YNS

__all__ = ["SparseBox3DLoss", "FocalLoss", "L1Loss"]


def reduce_mean(x: torch.Tensor) -> torch.Tensor:
    return x.sum() / max(x.numel(), 1)


class FocalLoss(nn.Module):
    """简化 Focal：对多类 one-hot 或 index target，ignore_index 的样本不参与。"""

    def __init__(self, alpha=0.25, gamma=2.0, reduction="sum", ignore_index=-1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, pred, target, avg_factor=None):
        """
        pred: (N, num_cls) logits
        target: (N,) class index，ignore_index 表示忽略
        """
        valid = (target >= 0) & (target != self.ignore_index)
        if valid.sum() == 0:
            return pred.new_tensor(0.0)
        pred = pred[valid]
        target = target[valid]
        num_cls = pred.shape[-1]
        target_one_hot = F.one_hot(target.clamp(0, num_cls - 1), num_cls).float()
        pred_sigmoid = pred.sigmoid()
        pt = (pred_sigmoid * target_one_hot + (1 - pred_sigmoid) * (1 - target_one_hot)).sum(dim=-1)
        weight = (1 - pt).pow(self.gamma)
        if self.alpha is not None:
            alpha_t = self.alpha * target_one_hot + (1 - self.alpha) * (1 - target_one_hot)
            weight = weight * alpha_t.sum(dim=-1)
        loss = F.binary_cross_entropy_with_logits(
            pred, target_one_hot, reduction="none"
        ).sum(dim=-1) * weight
        loss = loss.sum()
        if avg_factor is not None:
            loss = loss / max(avg_factor, 1.0)
        elif self.reduction == "mean":
            loss = loss / max(valid.sum().float().item(), 1.0)
        return loss


class L1Loss(nn.Module):
    def __init__(self, reduction="sum"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target, weight=None, avg_factor=None):
        if weight is not None:
            loss = (torch.abs(pred - target) * weight).sum(dim=-1)
        else:
            loss = torch.abs(pred - target).sum(dim=-1)
        if self.reduction == "sum":
            loss = loss.sum()
        else:
            loss = loss.mean()
        if avg_factor is not None:
            loss = loss / max(avg_factor, 1.0)
        return loss


class SparseBox3DLoss(nn.Module):
    """框回归 L1 + 可选 centerness / yawness（quality）。"""

    def __init__(
        self,
        reg_weights=None,
        loss_centerness=False,
        loss_yawness=False,
    ):
        super().__init__()
        self.reg_weights = reg_weights if reg_weights is not None else [1.0] * 11
        self.loss_centerness = loss_centerness
        self.loss_yawness = loss_yawness

    def forward(
        self,
        box,
        box_target,
        weight=None,
        avg_factor=None,
        suffix="",
        quality=None,
        cls_target=None,
    ):
        out = {}
        ndim = min(len(self.reg_weights), box.shape[-1], box_target.shape[-1])
        box = box[..., :ndim]
        box_target = box_target[..., :ndim]
        w = weight
        if w is None:
            w = box.new_ones(box.shape)
        else:
            w = w[..., :ndim]
        w = w * box.new_tensor(self.reg_weights[:ndim])[None, None, :]
        loss_box = (torch.abs(box - box_target) * w).sum() / max(avg_factor or 1.0, 1.0)
        out[f"loss_box{suffix}"] = loss_box

        if quality is not None and (self.loss_centerness or self.loss_yawness):
            cns = quality[..., CNS]
            yns = quality[..., YNS].sigmoid()
            cns_target = torch.exp(-torch.norm(box_target[..., [X, Y, Z]] - box[..., [X, Y, Z]], p=2, dim=-1))
            cns_loss = F.l1_loss(cns, cns_target, reduction="sum") / max(avg_factor or 1.0, 1.0)
            out[f"loss_cns{suffix}"] = cns_loss
            yns_target = (F.cosine_similarity(
                box_target[..., [SIN_YAW, COS_YAW]],
                box[..., [SIN_YAW, COS_YAW]],
                dim=-1,
            ) > 0).float()
            yns_loss = F.binary_cross_entropy(yns, yns_target, reduction="sum") / max(avg_factor or 1.0, 1.0)
            out[f"loss_yns{suffix}"] = yns_loss
        return out
