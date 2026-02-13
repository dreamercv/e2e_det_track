"""
跟踪头：用 ego 位姿将历史 anchor 对齐到当前帧，instance + 对齐后的 anchor 编码后做 cross-attention，
输出 (B, T, N, N) 亲和矩阵，与 anchor 顺序一致，用于匹配得到 track_id 与位置。
含 loss 与 decode。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment

from .box3d import X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, VY, VZ

__all__ = ["TrackHead", "track_affinity_loss", "decode_track", "align_anchors_to_frame"]


def align_anchors_to_frame(anchor: torch.Tensor, T_src2dst: torch.Tensor) -> torch.Tensor:
    """
    将 anchor 从源帧变换到目标帧。
    anchor: (B, N, 11) 编码格式 [x,y,z,log(w),log(l),log(h),sin(yaw),cos(yaw),vx,vy,vz]
    T_src2dst: (B, 4, 4) 源帧到目标帧的变换矩阵
    returns: (B, N, 11) 目标帧下的 anchor（编码格式不变）
    """
    B, N, _ = anchor.shape
    R = T_src2dst[:, :3, :3]
    t = T_src2dst[:, :3, 3]
    center = anchor[..., [X, Y, Z]]
    center = (R.unsqueeze(1) @ center.unsqueeze(-1)).squeeze(-1) + t.unsqueeze(1)
    size = anchor[..., [W, L, H]]
    sin_cos = anchor[..., [SIN_YAW, COS_YAW]]
    yaw_vec = (R[:, :2, :2].unsqueeze(1) @ sin_cos.unsqueeze(-1)).squeeze(-1)
    vel = anchor[..., VX:]
    vel = (R.unsqueeze(1) @ vel.unsqueeze(-1)).squeeze(-1)
    return torch.cat([center, size, yaw_vec, vel], dim=-1)


class TrackHead(nn.Module):
    """
    输入 seq_features (B,T,N,C)、seq_anchors (B,T,N,11)、T_ego_his2cur (B,T,4,4)。
    T_ego_his2cur[b,t] 表示从第 t-1 帧到第 t 帧的 ego 变换（t=0 时不使用）。
    输出 affinity (B,T,N,N)：affinity[b,t,i,j] 表示当前帧 t 的第 i 个与上一帧第 j 个的匹配度。
    """

    def __init__(
        self,
        feat_dim: int = 256,
        anchor_dim: int = 11,
        num_heads: int = 8,
        dropout: float = 0.1,
        embed_dim: int = 256,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.feat_proj = nn.Linear(feat_dim, embed_dim)
        self.anchor_proj = nn.Linear(anchor_dim, embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        seq_features: torch.Tensor,
        seq_anchors: torch.Tensor,
        T_ego_his2cur: torch.Tensor,
    ) -> torch.Tensor:
        """
        seq_features: (B, T, N, C)
        seq_anchors: (B, T, N, 11)
        T_ego_his2cur: (B, T, 4, 4)，[b,t] 为从 t-1 帧到 t 帧的变换
        returns: affinity (B, T, N, N)，与 anchor 顺序一致
        """
        B, T, N, C = seq_features.shape
        device = seq_features.device
        affinity_list = []

        for t in range(T):
            if t == 0:
                aff = torch.eye(N, device=device, dtype=seq_features.dtype).unsqueeze(0).expand(B, N, N)
                affinity_list.append(aff)
                continue
            feat_cur = seq_features[:, t]
            anchor_cur = seq_anchors[:, t]
            feat_prev = seq_features[:, t - 1]
            anchor_prev = seq_anchors[:, t - 1]
            T_prev2cur = T_ego_his2cur[:, t]
            anchor_prev_aligned = align_anchors_to_frame(anchor_prev, T_prev2cur)

            token_cur = self.norm(self.feat_proj(feat_cur) + self.anchor_proj(anchor_cur))
            token_prev = self.norm(self.feat_proj(feat_prev) + self.anchor_proj(anchor_prev_aligned))

            attn_out, attn_weights = self.cross_attn(
                token_cur, token_prev, token_prev, need_weights=True
            )
            affinity_list.append(attn_weights)

        return torch.stack(affinity_list, dim=1)


def track_affinity_loss(
    affinity: torch.Tensor,
    gt_match: torch.Tensor,
    ignore_index: int = -1,
) -> torch.Tensor:
    """
    亲和矩阵与 GT 匹配的负对数似然损失（affinity 为 attention 概率，每行和为 1）。
    affinity: (B, T, N, N)
    gt_match: (B, T, N)，gt_match[b,t,i] = 上一帧中匹配的 index in [0, N)，或 ignore_index 表示新轨迹/忽略
    """
    B, T, N, _ = affinity.shape
    aff_flat = (affinity.reshape(-1, N) + 1e-8).log()
    gt_flat = gt_match.reshape(-1).long()
    valid = gt_flat != ignore_index
    if valid.sum() == 0:
        return affinity.new_tensor(0.0)
    return F.nll_loss(aff_flat[valid], gt_flat[valid], reduction="mean")


def decode_track(
    affinity: torch.Tensor,
    seq_anchors: torch.Tensor,
    use_hungarian: bool = True,
    score_threshold: float = None,
) -> tuple:
    """
    从亲和矩阵解析出 track_id 与对应位置。
    affinity: (B, T, N, N)
    seq_anchors: (B, T, N, 11)，与 affinity 的 instance 顺序一致
    use_hungarian: True 用匈牙利匹配，False 用每行 argmax
    score_threshold: 若给出，则低于此的匹配视为新轨迹（可选）
    returns:
        track_ids: (B, T, N) int64，每个 instance 的 track id
        positions: (B, T, N, 3) 或直接用 seq_anchors 的 xyz，这里返回 seq_anchors[..., :3]
    """
    B, T, N, _ = affinity.shape
    device = affinity.device
    track_ids = torch.zeros(B, T, N, dtype=torch.int64, device=device)
    positions = seq_anchors[..., :3].clone()

    for b in range(B):
        track_ids[b, 0] = torch.arange(N, device=device)
        next_id = N
        for t in range(1, T):
            cost = -affinity[b, t].float().cpu().numpy()
            if use_hungarian:
                row_idx, col_idx = linear_sum_assignment(cost)
                if score_threshold is not None:
                    aff = affinity[b, t].cpu().numpy()
                    for i, j in zip(row_idx, col_idx):
                        if aff[i, j] < score_threshold:
                            track_ids[b, t, i] = next_id
                            next_id += 1
                        else:
                            track_ids[b, t, i] = track_ids[b, t - 1, j].item()
                else:
                    for i, j in zip(row_idx, col_idx):
                        track_ids[b, t, i] = track_ids[b, t - 1, j]
            else:
                best_j = affinity[b, t].argmax(dim=-1)
                for i in range(N):
                    j = best_j[i].item()
                    if score_threshold is not None and affinity[b, t, i, j].item() < score_threshold:
                        track_ids[b, t, i] = next_id
                        next_id += 1
                    else:
                        track_ids[b, t, i] = track_ids[b, t - 1, j]

    return track_ids, positions
