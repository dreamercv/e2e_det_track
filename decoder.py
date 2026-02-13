"""
3D 检测解码：将 head 输出的 classification / prediction（编码格式）解码为
boxes_3d（x,y,z,w,l,h,yaw,vx,vy,vz）、scores_3d、labels_3d。
参考官方 mmdet3d_plugin SparseBox3DDecoder。
"""
from typing import Optional, List, Union

import torch

from .box3d import decode_box, X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, CNS

__all__ = ["SparseBox3DDecoder", "decode_box"]


class SparseBox3DDecoder:
    """
    纯 PyTorch 实现，不依赖 mmdet。
    num_output: 每张图保留的 top-k 预测数
    score_threshold: 低于此分数的框过滤掉（None 表示不过滤）
    sorted: topk 是否按分数排序
    """

    def __init__(
        self,
        num_output: int = 300,
        score_threshold: Optional[float] = None,
        sorted: bool = True,
    ):
        self.num_output = num_output
        self.score_threshold = score_threshold
        self.sorted = sorted

    def decode_box(self, box: torch.Tensor) -> torch.Tensor:
        """单框解码：11 维编码 -> 10 维 (x,y,z,w,l,h,yaw,vx,vy,vz)。"""
        if box.shape[-1] > 11:
            box = box[..., :11]
        return decode_box(box)

    def decode(
        self,
        cls_scores: Union[torch.Tensor, List[torch.Tensor]],
        box_preds: Union[torch.Tensor, List[torch.Tensor]],
        instance_id: Optional[torch.Tensor] = None,
        quality: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        output_idx: int = -1,
    ) -> List[dict]:
        """
        对 head 输出做 topk、阈值过滤、decode_box，得到每张图的检测结果。
        cls_scores: (bs, num_pred, num_cls) 或 list of，取 [output_idx]
        box_preds: (bs, num_pred, 11) 或 list of，取 [output_idx]
        instance_id: (bs, num_pred) 可选，用于跟踪
        quality: (bs, num_pred, 2) 可选 [centerness, yawness]，用于 score * centerness
        output_idx: 使用第几个 decoder 的输出，-1 表示最后一个
        Returns: list of dict，每项含 "boxes_3d", "scores_3d", "labels_3d"；
                 若有 quality 则含 "cls_scores"；若有 instance_id 则含 "instance_ids"。
        """
        if isinstance(cls_scores, (list, tuple)):
            cls_scores = cls_scores[output_idx]
        if isinstance(box_preds, (list, tuple)):
            box_preds = box_preds[output_idx]
        cls_scores = cls_scores.sigmoid()
        squeeze_cls = instance_id is not None

        bs, num_pred, num_cls = cls_scores.shape
        if squeeze_cls:
            cls_scores_max, _ = cls_scores.max(dim=-1)
            cls_scores_max = cls_scores_max.unsqueeze(dim=-1)

        cls_flat = cls_scores.flatten(start_dim=1)
        cls_scores_topk, indices = cls_flat.topk(
            min(self.num_output, cls_flat.shape[1]), dim=1, sorted=self.sorted
        )
        cls_ids = indices % num_cls
        if self.score_threshold is not None:
            mask = cls_scores_topk >= self.score_threshold

        if quality is not None:
            if isinstance(quality, (list, tuple)):
                quality = quality[output_idx]
            centerness = quality[..., CNS]
            centerness = torch.gather(centerness, 1, torch.div(indices, num_cls, rounding_mode="trunc"))
            cls_scores_origin = cls_scores_topk.clone()
            cls_scores_topk = cls_scores_topk * centerness.sigmoid()
            cls_scores_topk, idx = torch.sort(cls_scores_topk, dim=1, descending=True)
            cls_ids = torch.gather(cls_ids, 1, idx)
            if self.score_threshold is not None:
                mask = torch.gather(mask, 1, idx)
            indices = torch.gather(indices, 1, idx)

        output = []
        for i in range(bs):
            category_ids = cls_ids[i]
            scores = cls_scores_topk[i]
            box = box_preds[i, torch.div(indices[i], num_cls, rounding_mode="trunc")]
            if self.score_threshold is not None:
                category_ids = category_ids[mask[i]]
                scores = scores[mask[i]]
                box = box[mask[i]]
            box = self.decode_box(box)
            out_i = {
                "boxes_3d": box.cpu(),
                "scores_3d": scores.cpu(),
                "labels_3d": category_ids.cpu(),
            }
            if quality is not None:
                scores_origin = cls_scores_origin[i]
                if self.score_threshold is not None:
                    scores_origin = scores_origin[mask[i]]
                out_i["cls_scores"] = scores_origin.cpu()
            if instance_id is not None:
                ids = instance_id[i, indices[i]]
                if self.score_threshold is not None:
                    ids = ids[mask[i]]
                out_i["instance_ids"] = ids
            output.append(out_i)
        return output
