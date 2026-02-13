# 真值格式与 DataLoader 输出约定

本文档说明：**真值应包含哪些内容**、**各字段的精确格式**，以及 **DataLoader 输出的 batch 如何直接送入 model 训练**。

---

## 一、模型 forward 入参一览

```python
head_out, seq_features, seq_anchors, track_out = model(
    x,                    # 图像
    rots, trans, intrins, distorts, post_rot3, post_tran3,  # 相机/图像增强参数
    theta_mats,           # 与 backbone 约定一致
    T_ego_his2curs,       # ego 位姿，跟踪用
    metas=None,           # 训练时放真值
)
```

| 参数 | 形状/类型 | 说明 |
|------|-----------|------|
| `x` | `(B, T, N_cam, 3, H, W)` | 时序多相机图像，B=batch 样本数，T=每样本帧数 |
| `rots` | 与 backbone 一致 | 每相机旋转 3x3，通常 `(B, T, N_cam, 3, 3)` |
| `trans` | 与 backbone 一致 | 每相机平移，如 `(B, T, N_cam, 1, 3)` |
| `intrins` | 与 backbone 一致 | 内参 3x3，如 `(B, T, N_cam, 3, 3)` |
| `distorts` | 与 backbone 一致 | 畸变参数，如 `(B, T, N_cam, 1, 8)` |
| `post_rot3` / `post_tran3` | 与 backbone 一致 | 图像增强的旋转/平移 |
| `theta_mats` | 如 `(B, T, 2, 3)` | 与 backbone 约定一致 |
| `T_ego_his2curs` | `(B, T, 4, 4)` 或 `None` | 从**上一帧到当前帧**的 ego 变换；不做跟踪可传 `None` |
| `metas` | `dict` 或 `None` | 训练时必须包含检测真值；做跟踪时需额外 `gt_track_match` |

**重要**：Head 内部把 batch 维当作 `B*T`（即把所有帧展平）。因此 `gt_labels_3d` / `gt_bboxes_3d` 的 list 长度必须为 **B*T**，且顺序与 `x` 的展平顺序一致：先样本 0 的 T 帧，再样本 1 的 T 帧……

---

## 二、真值内容与格式（metas）

### 2.1 检测真值（必选，训练时）

| 键 | 类型 | 格式说明 |
|----|------|----------|
| `gt_labels_3d` | `list` of `torch.Tensor` | **长度 = B*T**。第 `k` 个元素对应第 `k` 帧（展平后），形状 `(N_k,)`，dtype `long`，值为**类别下标**（0 到 num_classes-1）。该帧无框时可为 `(0,)` 的空 tensor。 |
| `gt_bboxes_3d` | `list` of `torch.Tensor` | **长度 = B*T**。第 `k` 个元素对应第 `k` 帧，形状 `(N_k, 10)`，**解码格式**：`[x, y, z, w, l, h, yaw, vx, vy, vz]`，与 `box3d` 约定一致。单位与你的场景一致（如米、度）。无框时 `(0, 10)`。 |

- 坐标系：与 backbone/数据集约定一致（通常为车体/ego，或统一世界系）。
- `yaw`：弧度。
- 速度 `vx, vy, vz`：若数据集无速度，可填 0。

### 2.2 跟踪真值（可选，仅训练且用 TrackHead 时需要）

| 键 | 类型 | 格式说明 |
|----|------|----------|
| `gt_track_match` | `torch.Tensor` | 形状 **`(B, T, N)`**，dtype `long`。`N` = head 的 `num_anchor`（如 900）。<br>**含义**：`gt_track_match[b, t, i]` = 当前帧（第 t 帧）第 i 个**预测 slot** 在**上一帧（t-1）**中匹配的**预测 slot 下标** j ∈ [0, N)，若无匹配或新轨迹则填 **-1**。即“当前帧 slot i 对应上一帧的 slot j”。<br>若标注是“每帧 GT 框 + track_id”，需先对每帧做“预测 slot ↔ GT”的匹配，再根据两帧 GT 的 track_id 推出“当前 slot i 对应上一帧的 slot j”。见下文第三节。 |

- **t=0**：没有“上一帧”，一般整帧填 **-1**，即 `gt_track_match[:, 0, :] = -1`。

---

## 三、从“每帧 GT 框 + track_id”构造 gt_track_match

常见标注是：每帧有 `boxes_3d`、`labels_3d`、`track_ids`（每条 GT 一个 track_id）。模型需要的是 **slot 对 slot**：`gt_track_match[b,t,i] = 上一帧的 slot 下标 j`。

**步骤**（对每个样本 b、每一帧 t≥1）：

1. **当前帧 t**：用匈牙利匹配（IoU 或中心距离）把**当前帧 N 个预测 slot** 与**当前帧 GT 框**匹配，得到 `slot_to_gt_cur[i]` = 当前帧 slot i 匹配到的 GT 下标（-1 表示未匹配）。
2. **上一帧 t-1**：同样得到 `slot_to_gt_prev[j]` = 上一帧 slot j 匹配到的 GT 下标。
3. 对当前帧每个 slot i：
   - 若 `slot_to_gt_cur[i] == -1`，则 `gt_track_match[b,t,i] = -1`。
   - 否则取当前帧该 GT 的 `track_id = track_ids_cur[slot_to_gt_cur[i]]`，在上一帧 GT 里找 `track_id_prev[k] == track_id` 的 k，再找**上一帧中匹配到该 GT k 的 slot**：即满足 `slot_to_gt_prev[j] == k` 的 j，令 `gt_track_match[b,t,i] = j`；若上一帧没有同 track_id 的 GT，则填 -1。

这样得到的就是“当前帧 slot i → 上一帧 slot j”的矩阵，可直接用于 `track_affinity_loss`。

---

## 四、DataLoader 输出格式（直接喂给 model）

下面是一个**最小可用的 batch 结构**，保证和 `model(..., metas=metas)` 一致。所有张量都在同一 device 上（或训练时由训练循环 to(device)）。

```python
batch = {
    # ----- 图像与相机（与 backbone 约定一致） -----
    "x": torch.Tensor,              # (B, T, N_cam, 3, H, W)
    "rots": torch.Tensor,            # 如 (B, T, N_cam, 3, 3)
    "trans": torch.Tensor,
    "intrins": torch.Tensor,
    "distorts": torch.Tensor,
    "post_rot3": torch.Tensor,
    "post_tran3": torch.Tensor,
    "theta_mats": torch.Tensor,      # 如 (B, T, 2, 3)

    # ----- 跟踪用（若不用 TrackHead 可省略） -----
    "T_ego_his2curs": torch.Tensor,  # (B, T, 4, 4)，从 t-1 到 t 的 ego 变换

    # ----- 真值（训练时必填） -----
    "metas": {
        "gt_labels_3d": list of Tensor,   # 长度 B*T，每元素 (N_k,)
        "gt_bboxes_3d": list of Tensor,    # 长度 B*T，每元素 (N_k, 10) 解码格式
        # 可选，训练跟踪时：
        "gt_track_match": torch.Tensor,   # (B, T, N)，long，-1 表示无匹配/新轨迹
    },
}
```

**调用方式**（示例）：

```python
head_out, seq_f, seq_a, track_out = model(
    batch["x"],
    batch["rots"],
    batch["trans"],
    batch["intrins"],
    batch["distorts"],
    batch["post_rot3"],
    batch["post_tran3"],
    batch["theta_mats"],
    batch.get("T_ego_his2curs"),
    metas=batch["metas"],
)
det_losses = head.loss(head_out, batch["metas"])
# 若用 TrackHead 且 metas 中有 gt_track_match：
# track_out["loss_track"]
```

---

## 五、示例：Dataset + collate_fn

下面给出一个**最小示例**：按“每样本 T 帧、每帧有 GT 框+类别”组 batch，并保证 `metas` 的 list 长度和顺序为 B*T。

```python
import torch
from torch.utils.data import Dataset, DataLoader

# 假设单条样本已经包含：
#   images: (T, N_cam, 3, H, W)
#   rots, trans, ...: 与 backbone 一致
#   gt_labels_per_frame: list of T tensors, 每帧 (N_t,)
#   gt_bboxes_per_frame: list of T tensors, 每帧 (N_t, 10) 解码格式
#   T_ego_hist2cur: (T, 4, 4)，t=0 可单位阵

class YourSparse4DDataset(Dataset):
    def __init__(self, ...):
        # 你的数据路径、T、N_cam 等
        pass

    def __getitem__(self, idx):
        # 读一条样本：B=1 的“一个片段”，含 T 帧
        return {
            "x": torch.randn(1, T, N_cam, 3, H, W),  # 示例
            "rots": ...,
            "trans": ...,
            "intrins": ...,
            "distorts": ...,
            "post_rot3": ...,
            "post_tran3": ...,
            "theta_mats": ...,
            "T_ego_his2curs": ...,   # (1, T, 4, 4)
            "gt_labels_3d": [t1, t2, ...],   # list 长度 T，每元素 (N_t,)
            "gt_bboxes_3d": [b1, b2, ...],   # list 长度 T，每元素 (N_t, 10)
            "gt_track_match": None,  # 或 (1, T, N) 若做跟踪
        }

def collate_fn(batch_list):
    """将 list of dict 合并成一个 batch；保证 metas 里 list 长度为 B*T。"""
    B = len(batch_list)
    out = {}
    # 图像与相机：按 key 做 stack 或 按约定 concat
    for key in ["x", "rots", "trans", "intrins", "distorts", "post_rot3", "post_tran3", "theta_mats", "T_ego_his2curs"]:
        if key not in batch_list[0]:
            continue
        out[key] = torch.cat([b[key] for b in batch_list], dim=0)

    # metas：list 要展平为 B*T
    gt_labels_3d = []
    gt_bboxes_3d = []
    gt_track_match_list = []
    for b in batch_list:
        gt_labels_3d.extend(b["gt_labels_3d"])
        gt_bboxes_3d.extend(b["gt_bboxes_3d"])
        if b.get("gt_track_match") is not None:
            gt_track_match_list.append(b["gt_track_match"])

    out["metas"] = {
        "gt_labels_3d": gt_labels_3d,
        "gt_bboxes_3d": gt_bboxes_3d,
    }
    if gt_track_match_list:
        out["metas"]["gt_track_match"] = torch.cat(gt_track_match_list, dim=0)

    return out

# 使用
loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, shuffle=True)
for batch in loader:
    # 若在 GPU 上训练，先把 batch 整体 to(device)
    head_out, seq_f, seq_a, track_out = model(
        batch["x"], batch["rots"], batch["trans"],
        batch["intrins"], batch["distorts"],
        batch["post_rot3"], batch["post_tran3"],
        batch["theta_mats"],
        batch.get("T_ego_his2curs"),
        metas=batch["metas"],
    )
    losses = head.loss(head_out, batch["metas"])
    if track_out and "loss_track" in track_out:
        losses["loss_track"] = track_out["loss_track"]
    # 反传 ...
```

**要点**：

- 每个 `__getitem__` 返回的是“一个片段”（含 T 帧），即 batch 里一条样本。
- `collate_fn` 里把 `gt_labels_3d` / `gt_bboxes_3d` 用 `list.extend` 拼成 **B*T 长的 list**，顺序为：样本0帧0, 样本0帧1, …, 样本0帧T-1, 样本1帧0, …。
- `T_ego_his2curs` 若每个样本是 `(1, T, 4, 4)`，则 `torch.cat(..., dim=0)` 得到 `(B, T, 4, 4)`。

---

## 六、head 内 T 的约定

当前 head 里对 `seq_features` / `seq_anchors` 的 reshape 写死为 **T=7**：

```python
final_feature = instance_feature.reshape(-1, 7, *instance_feature.shape[1:])
final_anchor = anchor.reshape(-1, 7, *anchor.shape[1:])
```

因此：

- 若你的 `T != 7`，需要改 head 中这两行为 `reshape(B, T, ...)`，并保证 `feature_maps.shape[0] == B*T`（即 backbone 输出的 batch 维就是 B*T）。
- DataLoader 的 `batch["x"].shape[0]` = B，`batch["x"].shape[1]` = T，backbone 输入 `x` 后应输出 `(B*T, C, H, W)`，这样 head 收到的 batch_size 就是 B*T。

---

## 七、小结表

| 真值/输入 | 形状/类型 | 说明 |
|-----------|-----------|------|
| 检测类别 | `list` of `(N_k,)`，长 B*T | 每帧的类别下标 |
| 检测框 | `list` of `(N_k, 10)`，长 B*T | 解码格式 x,y,z,w,l,h,yaw,vx,vy,vz |
| 跟踪匹配 | `(B, T, N)` long | 当前帧 slot i 匹配上一帧的 slot j，-1 表示无/新轨迹 |
| T_ego_his2curs | `(B, T, 4, 4)` | 上一帧→当前帧 ego 变换 |
| metas 长度 | gt_* 的 list 长 B*T | 与 backbone 输出 (B*T, C, H, W) 一一对应 |

按上述格式制作真值并实现 Dataset + collate_fn，DataLoader 的输出即可直接接到 model 进行训练。
