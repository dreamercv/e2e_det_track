"""3D box 维度索引与 decode，与 Sparse4D 官方一致。"""
import torch

# 编码格式 (11 维): x, y, z, log(w), log(l), log(h), sin(yaw), cos(yaw), vx, vy, vz
# 解码格式 (10 维): x, y, z, w, l, h, yaw, vx, vy, vz
X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, VY, VZ = list(range(11))
CNS, YNS = 0, 1
YAW = 6  # decoded 中 yaw 的索引 (第 7 维，0-based=6)


def decode_box(box):
    """
    将编码后的 3D 框 (11 维) 解码为 (x,y,z,w,l,h,yaw,vx,vy,vz) 10 维。
    与官方 mmdet3d_plugin SparseBox3DDecoder.decode_box 一致。
    box: (..., 11) 最后一维为 [x,y,z,log(w),log(l),log(h),sin(yaw),cos(yaw),vx,vy,vz]
    returns: (..., 10) [x,y,z,w,l,h,yaw,vx,vy,vz]
    """
    yaw = torch.atan2(box[..., SIN_YAW], box[..., COS_YAW])
    out = torch.cat(
        [
            box[..., [X, Y, Z]],
            box[..., [W, L, H]].exp(),
            yaw.unsqueeze(-1),
            box[..., VX:],
        ],
        dim=-1,
    )
    return out
