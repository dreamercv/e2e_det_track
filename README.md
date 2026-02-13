# Sparse4D + BEV + TrackHead（纯 PyTorch）

基于 Sparse4D 的 3D 检测与多目标跟踪实现，**不依赖 mmcv/mmdet**，便于调试与二次开发。支持**单次 forward 内多帧时序输入**，同时输出检测结果与跟踪亲和矩阵 / track_id。

---

## 一、方案概览

### 1.1 整体流程

```
时序图像 (B, T, N_cam, 3, H, W)
    ↓
BEV Backbone（含时序对齐，输出 B*T 合并）
    ↓
BEV 特征 (B*T, C, H_bev, W_bev)
    ↓
Sparse4D Head（逐帧 decoder，无跨 forward 的 cache）
    ↓
检测输出：classification / prediction / quality
时序输出：seq_features (B,T,N,256)、seq_anchors (B,T,N,11)
    ↓
TrackHead（可选）
    ↓
track_affinity (B,T,N,N) → 训练时 loss_track，推理时 decode_track → track_ids / positions
```

- **时序方式**：时序在 backbone 内完成（多帧图 → 多帧 BEV，batch 维为 `B*T`）。Head 只接收 `feature_maps`，**不做跨 forward 的 instance/ anchor 缓存与更新**（与原论文的 cache-based 时序不同）。
- **检测**：Sparse4D 稀疏 anchor + 多 decoder 层 + 可选去噪训练（DenoisingSampler）。
- **跟踪**：在 head 输出的 `seq_features`、`seq_anchors` 上，用 ego 位姿对齐历史帧 anchor，经 TrackHead 的 cross-attention 得到帧间亲和矩阵，再通过匈牙利匹配或 argmax 解析出 track_id 与位置。

### 1.2 3D 框表示

- **编码（11 维）**：`x, y, z, log(w), log(l), log(h), sin(yaw), cos(yaw), vx, vy, vz`
- **解码（10 维）**：`x, y, z, w, l, h, yaw, vx, vy, vz`  
解码接口：`from sparse4d_bev_torch import decode_box` 或 `SparseBox3DDecoder.decode_box()`。

### 1.3 模块结构

| 模块 | 作用 |
|------|------|
| `box3d` | 11/10 维索引、`decode_box` |
| `instance_bank` | 可学习 anchor 管理（本方案中 `num_temp_instances=0`，无时序 cache） |
| `detection3d_blocks` | SparseBox3D Encoder / Refinement / KeyPointsGenerator |
| `bev_aggregation` | 从 BEV 特征图按 anchor 关键点采样，更新 instance 特征 |
| `head` | Sparse4DHead：gnn → norm → deformable(bev_agg) → ffn → refine 多轮，输出检测 + `seq_features`/`seq_anchors` |
| `dn_sampler` | 去噪采样：生成 dn_anchors、匈牙利匹配得到正常/dn 的 target |
| `losses` | FocalLoss（分类）、SparseBox3DLoss（回归 + centerness/yawness quality） |
| `decoder` | SparseBox3DDecoder：topk、阈值、decode_box → boxes_3d / scores_3d / labels_3d |
| `track_head` | TrackHead、`align_anchors_to_frame`、`track_affinity_loss`、`decode_track` |
| `model` | `build_sparse4d_bev_head`、`build_track_head`、`build_sparse4d_bev_model` 组装完整模型 |

---

## 二、环境与依赖

- Python 3.7+
- PyTorch（建议 1.9+）
- NumPy、SciPy（`scipy.optimize.linear_sum_assignment` 用于匈牙利匹配）

无需 mmcv、mmdet、mmdet3d。

---

## 三、使用方法

### 3.1 包内导入（推荐）

在 `projects` 或已把 `sparse4d_bev_torch` 加入 `PYTHONPATH` 的前提下：

```python
from sparse4d_bev_torch import (
    build_sparse4d_bev_head,
    build_sparse4d_bev_model,
    build_track_head,
    SparseBox3DDecoder,
    decode_box,
)
```

### 3.2 仅检测（无跟踪）

```python
import torch
from sparse4d_bev_torch import build_sparse4d_bev_head, build_sparse4d_bev_model

# 1. 构建 BEV backbone（需自行实现或使用项目内 bev_backbone）
# 输入：多相机图像 + 内外参；输出：feature_maps (B*T, 256, H_bev, W_bev)
backbone = YourBEVBackbone(...)

# 2. 构建 head（输入为 feature_maps，无 T_ego）
head = build_sparse4d_bev_head(
    num_anchor=900,
    embed_dims=256,
    num_decoder=6,
    num_single_frame_decoder=5,
    num_classes=10,
    use_dn=True,
    use_decoder=True,   # 推理时用 SparseBox3DDecoder 得到 boxes_3d
    decoder_num_output=300,
    decoder_score_threshold=0.3,
)

# 3. 组装模型（不装 track_head 则 T_ego_his2curs 可传 None）
model = build_sparse4d_bev_model(backbone, head, track_head=None)

# 4. 前向
# x: (B, T, N_cam, 3, H, W)，rots/trans/intrins 等与 backbone 约定一致
feature_maps = model.bev_backbone(x, rots, trans, intrins, distorts, post_rot3, post_tran3, theta_mats)
head_out, seq_features, seq_anchors, track_out = model(
    x, rots, trans, intrins, distorts, post_rot3, post_tran3,
    theta_mats=theta_mats,
    T_ego_his2curs=None,
    metas=metas,
)
# head_out: {"classification", "prediction", "quality", ...}
# 若 use_decoder=True，可用 head.decoder.decode(...) 得到 list of {"boxes_3d","scores_3d","labels_3d"}
```

### 3.3 检测 + 跟踪（一次 forward）

```python
from sparse4d_bev_torch import (
    build_sparse4d_bev_head,
    build_sparse4d_bev_model,
    build_track_head,
)

backbone = YourBEVBackbone(...)
head = build_sparse4d_bev_head(
    num_anchor=900,
    embed_dims=256,
    num_decoder=6,
    num_single_frame_decoder=5,
    num_classes=10,
    use_dn=True,
)
track_head = build_track_head(feat_dim=256, anchor_dim=11)
model = build_sparse4d_bev_model(backbone, head, track_head)

# 输入需包含 T_ego_his2curs: (B, T, 4, 4)，表示从 t-1 到 t 的 ego 变换
head_out, seq_features, seq_anchors, track_out = model(
    x, rots, trans, intrins, distorts, post_rot3, post_tran3,
    theta_mats=theta_mats,
    T_ego_his2curs=T_ego_his2curs,  # (B, T, 4, 4)
    metas=metas,
)

# 训练：metas 中提供 gt_track_match (B, T, N)，t=0 全 -1，t>0 为上一帧匹配 index 或 -1
# track_out["loss_track"] 为 NLL loss
if model.training and track_out is not None and "loss_track" in track_out:
    loss_track = track_out["loss_track"]

# 推理：track_out 含 track_affinity (B,T,N,N)、track_ids (B,T,N)、track_positions (B,T,N,3)
if not model.training and track_out is not None:
    track_ids = track_out["track_ids"]       # (B, T, N) int64
    track_positions = track_out["track_positions"]  # (B, T, N, 3)
```

### 3.4 metas 约定（训练）

- **检测**  
  - `gt_labels_3d`: list of (N_i,) 类别下标。  
  - `gt_bboxes_3d`: list of (N_i, 10)，解码格式 `[x,y,z,w,l,h,yaw,vx,vy,vz]`。  
  list 长度为 `B*T`（与 head 的 batch 一致）。

- **跟踪（仅当使用 TrackHead 且训练时）**  
  - `gt_track_match`: `(B, T, N)` long。  
  - `gt_track_match[b,t,i]` = 当前帧第 i 个 instance 在**上一帧**中匹配的 instance 下标 `[0, N)`，若无匹配或新轨迹则为 `-1`。  
  - `t=0` 一般全为 `-1`。

### 3.5 检测解码得到 boxes_3d

```python
from sparse4d_bev_torch import SparseBox3DDecoder, decode_box

decoder = SparseBox3DDecoder(num_output=300, score_threshold=0.3, sorted=True)
# 取最后一层 decoder 输出
cls_list = head_out["classification"]
pred_list = head_out["prediction"]
qt_list = head_out.get("quality")
results = decoder.decode(
    cls_list, pred_list,
    instance_id=head_out.get("instance_id"),
    quality=qt_list,
    output_idx=-1,
)
# results: list of dict，每项 "boxes_3d"(N,10), "scores_3d"(N,), "labels_3d"(N,)
```

---

## 四、Head 输入与 reshape 说明

- Head 的输入是 **单一张量** `feature_maps`（或 list 且第 0 维为 batch）。  
- 本实现中，**batch 维 = B*T**（例如 2×7=14）。Head 内部在输出时将 `instance_feature` / `anchor` 再 reshape 成 `(B, T, N, C)` / `(B, T, N, 11)`，因此 **T 需固定**（代码中默认按 T=7 reshape，若 T 变化需改 head 内 `reshape(-1, 7, ...)` 为 `reshape(B, T, ...)` 并传入 T）。

---

## 五、跟踪相关接口

- **`align_anchors_to_frame(anchor, T_src2dst)`**  
  将编码格式的 `(B, N, 11)` anchor 从源帧变换到目标帧。

- **`TrackHead.forward(seq_features, seq_anchors, T_ego_his2cur)`**  
  输出 `affinity (B, T, N, N)`；t=0 为恒等矩阵。

- **`track_affinity_loss(affinity, gt_match, ignore_index=-1)`**  
  亲和矩阵与 GT 匹配的 NLL 损失。

- **`decode_track(affinity, seq_anchors, use_hungarian=True, score_threshold=None)`**  
  返回 `(track_ids, positions)`，用于推理时得到每帧每个 instance 的 track_id 与 (x,y,z)。

---

## 六、文件一览

```
sparse4d_bev_torch/
├── __init__.py           # 包导出
├── box3d.py              # 11/10 维定义、decode_box
├── instance_bank.py      # 可学习 anchor
├── detection3d_blocks.py # Encoder / Refinement / KeyPointsGenerator
├── bev_aggregation.py    # BEV 特征采样
├── head.py               # Sparse4DHead
├── dn_sampler.py         # 去噪采样与匈牙利匹配
├── losses.py             # FocalLoss、SparseBox3DLoss
├── decoder.py            # SparseBox3DDecoder
├── track_head.py         # TrackHead、track_affinity_loss、decode_track、align_anchors_to_frame
├── model.py              # build_* 与 Sparse4DBEVModel
├── bev_backbone.py       # 可选 BEV backbone 参考实现
└── README.md             # 本说明
```

---

## 七、快速自测（检测 + 跟踪 + loss + decode）

在项目根目录（能 `import sparse4d_bev_torch`）下运行下面脚本，可验证一次 forward 同时得到检测/跟踪输出、检测 loss、跟踪 loss 以及推理时的 track_ids / positions（使用 dummy backbone 与假 GT）：

```python
import torch
import torch.nn as nn
from sparse4d_bev_torch import build_sparse4d_bev_head, build_sparse4d_bev_model, build_track_head

class DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, rots, trans, intrins, distorts, post_rot3, post_tran3, theta_mats):
        B, T = x.shape[:2]
        return torch.randn(B * T, 256, 200, 80, device=x.device)

B, T, N = 2, 7, 50
head = build_sparse4d_bev_head(num_anchor=N, embed_dims=256, num_decoder=2, num_single_frame_decoder=2, num_classes=3, use_dn=True)
track_head = build_track_head(feat_dim=256, anchor_dim=11)
model = build_sparse4d_bev_model(DummyBackbone(), head, track_head)

x = torch.zeros(B, T, 1, 3, 128, 384)
rots = torch.eye(3).view(1,1,1,3,3).expand(B,T,1,3,3)
trans = torch.zeros(B,T,1,1,3)
intrins = torch.eye(3).view(1,1,1,3,3).expand(B,T,1,3,3)
distorts = torch.zeros(B,T,1,1,8)
post_rot3 = torch.eye(3).view(1,1,1,3,3).expand(B,T,1,3,3)
post_tran3 = torch.zeros(B,T,1,1,3)
T_ego_his2curs = torch.eye(4).view(1,1,4,4).expand(B,T,4,4).clone()

metas = {
    "gt_labels_3d": [torch.randint(0, 3, (2,)) for _ in range(B * T)],
    "gt_bboxes_3d": [torch.zeros(2, 10) for _ in range(B * T)],
    "gt_track_match": torch.randint(-1, N, (B, T, N)),
}
metas["gt_track_match"][:, 0, :] = -1

# 训练
model.train()
head_out, seq_features, seq_anchors, track_out = model(x, rots, trans, intrins, distorts, post_rot3, post_tran3, theta_mats=torch.zeros(B,T,2,3), T_ego_his2curs=T_ego_his2curs, metas=metas)
det_losses = head.loss(head_out, metas)
print("train ok", list(det_losses.keys()), "loss_track", track_out["loss_track"].item())

# 推理
model.eval()
with torch.no_grad():
    head_out, _, _, track_out = model(x, rots, trans, intrins, distorts, post_rot3, post_tran3, theta_mats=torch.zeros(B,T,2,3), T_ego_his2curs=T_ego_his2curs, metas=None)
print("eval ok", track_out["track_ids"].shape, track_out["track_positions"].shape)
```

若上述脚本无报错并打印 loss 与 shape，则检测 + 跟踪 + loss + decode 链路通过。
