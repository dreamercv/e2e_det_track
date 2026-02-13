"""
BEV 特征聚合：从 BEV 特征图按 anchor (x,y) 采样，纯 PyTorch。
"""
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .detection3d_blocks import SparseBox3DKeyPointsGenerator
from .box3d import X, Y, Z, W, L, H, SIN_YAW, COS_YAW

__all__ = ["BEVFeatureAggregation"]


class BEVFeatureAggregation(nn.Module):
    """从 BEV 特征图 (B, C, H_bev, W_bev) 按 key_points 的 x,y 采样并更新 instance_feature。"""

    def __init__(
        self,
        embed_dims: int = 256,
        bev_bounds: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None,
        kps_generator: Optional[nn.Module] = None,
        num_pts: Optional[int] = None,
        proj_drop: float = 0.0,
        residual_mode: str = "add",
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.proj_drop = nn.Dropout(proj_drop)
        self.residual_mode = residual_mode
        if bev_bounds is None:
            bev_bounds = ([-80.0, 120.0, 1.0], [-40.0, 40.0, 1.0])
        self.register_buffer(
            "bev_bounds",
            torch.tensor(
                [
                    [bev_bounds[0][0], bev_bounds[0][1], bev_bounds[0][2]],
                    [bev_bounds[1][0], bev_bounds[1][1], bev_bounds[1][2]],
                ],
                dtype=torch.float32,
            ),
        )
        if kps_generator is not None:
            self.kps_generator = kps_generator
            self.num_pts = self.kps_generator.num_pts
        else:
            self.kps_generator = None
            self.num_pts = num_pts if num_pts is not None else 1
        self.output_proj = nn.Linear(embed_dims, embed_dims)

    def init_weight(self):
        nn.init.xavier_uniform_(self.output_proj.weight)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)

    def _key_points_to_bev_grid(self, key_points: torch.Tensor) -> torch.Tensor:
        bs, num_anchor, num_pts, _ = key_points.shape
        xy = key_points[..., :2]
        xmin = self.bev_bounds[0, 0].item()
        xmax = self.bev_bounds[0, 1].item()
        ymin = self.bev_bounds[1, 0].item()
        ymax = self.bev_bounds[1, 1].item()
        grid_x = (xy[..., 0] - xmin) / (xmax - xmin + 1e-6) * 2.0 - 1.0
        grid_y = (xy[..., 1] - ymin) / (ymax - ymin + 1e-6) * 2.0 - 1.0
        grid = torch.stack([grid_y, grid_x], dim=-1)
        return grid.reshape(bs, num_anchor * num_pts, 2)

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        feature_maps: List[torch.Tensor],
        metas: dict = None,
        **kwargs,
    ):
        bs, num_anchor = instance_feature.shape[:2]
        if self.kps_generator is not None:
            key_points = self.kps_generator(anchor, instance_feature)
        else:
            key_points = anchor[..., [X, Y, Z]].unsqueeze(2)
        bev_map = feature_maps[0]
        if bev_map.dim() != 4:
            bev_map = bev_map.flatten(end_dim=1)
        grid = self._key_points_to_bev_grid(key_points)
        grid = grid.to(device=bev_map.device, dtype=bev_map.dtype)
        grid = grid.reshape(bs, 1, num_anchor * self.num_pts, 2)
        sampled = torch.nn.functional.grid_sample(
            bev_map, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        C = sampled.shape[1]
        sampled = sampled.squeeze(2).permute(0, 2, 1).reshape(bs, num_anchor, self.num_pts, C)
        features = sampled.sum(dim=2)
        output = self.proj_drop(self.output_proj(features))
        if self.residual_mode == "add":
            output = output + instance_feature
        elif self.residual_mode == "cat":
            output = torch.cat([output, instance_feature], dim=-1)
        return output
