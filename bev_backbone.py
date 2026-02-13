# -*- encoding: utf-8 -*-
'''
@File         :bev_backbone_1.py
@Date         :2026/02/12 16:14:32
@Author       :Binge.Van
@E-mail       :afb5szh@bosch.com
@Version      :V1.0.0
@Description  :

'''


import os,sys

import json
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import colorsys
import torch.nn.functional as F
import torch

from torch import nn

def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
    return dx, bx, nx

class Splat(nn.Module):
    def __init__(self,grid_conf,input_size=(128,384)):
        super(Splat, self).__init__()
        self.input_size = input_size
        self.grid_conf = grid_conf

        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'], self.grid_conf['ybound'], self.grid_conf['zbound'])
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)
        self.voxels = self.create_voxels()

    def create_voxels(self):
        xs = torch.linspace(self.bx[0] - self.dx[0] / 2, self.bx[0] - self.dx[0] / 2 + self.dx[0] * (self.nx[0] - 1),
                            self.nx[0], dtype=torch.float).view(1, 1, self.nx[0]).expand(self.nx[2], self.nx[1],
                                                                                         self.nx[0])
        ys = torch.linspace(self.bx[1] - self.dx[1] / 2, self.bx[1] - self.dx[1] / 2 + self.dx[1] * (self.nx[1] - 1),
                            self.nx[1], dtype=torch.float).view(1, self.nx[1], 1).expand(self.nx[2], self.nx[1],
                                                                                         self.nx[0])
        zs = torch.linspace(self.bx[2] - self.dx[2] / 2, self.bx[2] - self.dx[2] / 2 + self.dx[2] * (self.nx[2] - 1),
                            self.nx[2], dtype=torch.float).view(self.nx[2], 1, 1).expand(self.nx[2], self.nx[1],
                                                                                         self.nx[0])
        voxels = torch.stack((xs, ys, zs), -1) 
        return nn.Parameter(voxels, requires_grad=False)

    def bev2eachroi(self,  points, rots, trans, intrins, distorts, post_rots, post_trans):
        B, N, _, _ = rots.shape  # [bs*sq,roi,3,3]
        Z, Y, X, _ = points.shape  
        P = Z * Y * X
        points = points.view(1, 1, Z * Y * X, 3).expand(B, N, P, 3).clone()
        
        points = rots.view(B, N, 1, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)
        points = points + trans.view(B, N, 1, 3)
        # points = points.unsqueeze(-2).matmul(torch.inverse(Rs).view(B, N, 1, 3, 3)).squeeze(-2)

        x, y, z = torch.where(points[:, :, :, 2] < 0)
        points[x, y, z, 0] = 9999 * torch.ones_like(x, dtype=points.dtype)
        points[x, y, z, 1] = 9999 * torch.ones_like(x, dtype=points.dtype)

        depths = points[..., 2:]
        points = torch.cat((points[..., :2] / depths, torch.ones_like(depths)), -1)

        points1 = points[:, 0:1]
        intrins1 = intrins[:, 0:1].view(B, 1, 1, 3, 3)
        distorts1 = distorts[:, 0:1].view(B, 1, 1, 8)
        points[:, 0:1] = self.projectPoints_fisheye(points1, distorts1, intrins1) 

        points = post_rots.view(B, N, 1, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)
        points = points + post_trans.view(B, N, 1, 3)
        points = points.view(B, N, Z, Y, X, 3).permute(0, 1, 2, 4, 3, 5)[..., :2]

        # # 可视化验证
        # image_all_000 = feature.cpu().numpy()[0].transpose(1,2,0)
        # alll = []
        # for i in range(6):
        #     pts  = points[0,0,i].reshape(-1, 2).cpu().numpy()
        #     image_all = image_all_000.copy()
        #     for m in range(pts.shape[0]):
                
        #         try:
        #             cv2.circle(image_all, (int(pts[m, 0]), int(pts[m, 1])), 1, (255, 255, 255), 2)
        #         except:
        #             pass
        #     alll.append(image_all)
        # cv2.imwrite("local_map.jpg", np.concatenate(alll,0))




        features_shape = torch.Tensor([self.input_size[0], self.input_size[1]]).to(points.device)
        points = self.normalize_coords(coords=points, shape=features_shape)
        depths = depths.view(B, N, Z, Y, X, 1).permute(0, 1, 2, 5, 4, 3).squeeze().view(B * N, Z, X, Y)
        points = points.view(B * N * Z, X, Y, 2)

        return points


    def projectPoints_fisheye(self, proj_points, dist_coeffs, ori_intrin):
        """
        :param data: (torch.tensor, shape=[N, cams, C, 3]) 3D points in camera coordinates.
        :param K: (torch.tensor, shape=[N, cams, 3, 3]) Camera matrix.
        :param dist_coeffs: (torch.tensor, shape=[N, cams, 1, 8]) Distortion coefficients.
        : 径向畸变系数(k1,k2),切向畸变系数(p1,p2),径向畸变系数(k3,k4,k5,k6)
        :return: (torch.tensor, shape=[N, 2]) Projected 2D points.
        """
        # Apply dist_coeffs
        import math
        r = torch.sqrt(torch.sum(proj_points[..., :2] ** 2, dim=-1, keepdim=True)).squeeze(-1)
        k1, k2, p1, p2, k3, k4, k5, k6 = dist_coeffs.unbind(-1)  # 
        t = torch.atan2(r, torch.ones_like(r))
        radial = t * (1 + k1 * t ** 2 + k2 * t ** 4 + k3 * t ** 6 + k4 * t ** 8) / r  # torch.Size([5, 1, 76800])

        proj_points[..., 0] = proj_points[..., 0] * radial
        proj_points[..., 1] = proj_points[..., 1] * radial

        # Apply camera matrix
        proj_points = ori_intrin.matmul(proj_points.unsqueeze(-1)).squeeze(-1)

        return proj_points

    

    def normalize_coords(self, coords, shape):
        """
        Normalize coordinates of a grid between [-1, 1]
        Args:
            coords [torch.Tensor(..., 2)]: Coordinates in grid
            shape [torch.Tensor(2)]: Grid shape [H, W]
        Returns:
            norm_coords [torch.Tensor(.., 2)]: Normalized coordinates in grid
        """
        min_n = -1
        max_n = 1
        shape = torch.flip(shape, dims=[0])  # Reverse ordering of shape 512,128

        # Subtract 1 since pixel indexing from [0, shape - 1]
        norm_coords = coords / (shape - 1) * (max_n - min_n) + min_n
        return norm_coords  # [-1,1]


class BEVBackbone(nn.Module):
    def __init__(self, grid_conf, input_size=(128,384)):
        super(BEVBackbone, self).__init__()
        self.grid_conf = grid_conf
        self.input_size = input_size
        self.splat = Splat(grid_conf, input_size)
        self.bev_backbone = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2)

        )
        self.layer = nn.Conv2d(256*6, 256, kernel_size=3, stride=1, padding=1)
        self.algin_fusion = nn.Conv2d(256*2, 256, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x,rots, trans, intrins, distorts, post_rot3, post_tran3,theta_mats):

        """
        # B,5*6,3,128,384
        # B*5*6,3,128,384
        B, img_num 6,  c h w
        x_img.reshape(6, B, img_num, *x_img.shape[2:])
        """
        B,S,N,C,H,W = x.shape
        rots = rots.flatten(0,1)
        trans = trans.flatten(0,1)
        intrins = intrins.flatten(0,1)
        distorts = distorts.flatten(0,1)
        post_rot3 = post_rot3.flatten(0,1)
        post_tran3 = post_tran3.flatten(0,1)
        x = x.view(B*S*N,C,H,W)
        x = self.bev_backbone(x)
        points = self.splat.create_voxels()
        grid = self.splat.bev2eachroi(points,rots.to(points.dtype), trans.to(points.dtype), intrins.to(points.dtype), distorts.to(points.dtype), post_rot3.to(points.dtype), post_tran3.to(points.dtype))
        B, X, Y, _ = grid.shape
        B = int(B / 6)
        index_0 = [6 * i + 0 for i in range(B)]
        index_1 = [6 * i + 1 for i in range(B)]
        index_2 = [6 * i + 2 for i in range(B)]
        index_3 = [6 * i + 3 for i in range(B)]
        index_4 = [6 * i + 4 for i in range(B)]
        index_5 = [6 * i + 5 for i in range(B)]
        grid = grid.to(x.dtype)
        output_0 = F.grid_sample(input=x, grid=grid[index_0, ...], mode="nearest", padding_mode="zeros")
        output_1 = F.grid_sample(input=x, grid=grid[index_1, ...], mode="nearest", padding_mode="zeros")
        output_2 = F.grid_sample(input=x, grid=grid[index_2, ...], mode="nearest", padding_mode="zeros")
        output_3 = F.grid_sample(input=x, grid=grid[index_3, ...], mode="nearest", padding_mode="zeros")
        output_4 = F.grid_sample(input=x, grid=grid[index_4, ...], mode="nearest", padding_mode="zeros")
        output_5 = F.grid_sample(input=x, grid=grid[index_5, ...], mode="nearest", padding_mode="zeros")
        out_put = torch.cat([output_0, output_1, output_2, output_3, output_4, output_5], dim=1)  # torch.Size([25, 384, 150, 100])
        out_put = self.layer(out_put)
        out_put = out_put.view(-1, S, *out_put.shape[-3:])
        algin_features = []
        for i in range(S):
            if i == 0:
                algin_features.append(out_put[:,i])
            else:
                algin_feature = self.warp_feature(out_put[:,i], theta_mats[:,i].to(out_put.dtype))
                algin_features.append(self.algin_fusion(torch.cat([algin_features[-1],algin_feature],1)))
        algin_features = torch.stack(algin_features).permute(1,0,2,3,4).flatten(0,1)
        return algin_features # 输出时序帧信息，每一帧都和前一阵融合这样子达到了渐进的融合方式

    def warp_feature(self,features, theta):
        B, C, H, W = features.size()
        grids = F.affine_grid(theta, torch.Size((B,C,H,W)), align_corners=True)
        cropped_features = F.grid_sample(features, grids, align_corners=True)
        return cropped_features
def quaternion_to_rotation_matrix( x: float, y: float, z: float,w: float) -> np.ndarray:
    R = np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
    ])
    
    return R
def TransformationmatrixEgo(orientation,position):
    w,x,y,z = orientation
    rotation_matrix = quaternion_to_rotation_matrix(x,y,z,w)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = position
    return transform

def egopose_alginhistory2current(ego_poses):
    T_ego_his2curs = []
    T_ego2wld_cur = TransformationmatrixEgo(ego_poses[-1]["orientation"],ego_poses[-1]["position"])
    for i,ego_pose in enumerate(ego_poses):
        T_ego2wld_his = TransformationmatrixEgo(ego_pose["orientation"],ego_pose["position"])
        pose_diff = np.linalg.inv(T_ego2wld_cur)@T_ego2wld_his
        T_ego_his2curs.append(pose_diff)
    return np.array(T_ego_his2curs)

def gen_theta_mat(ego_poses):
    sx = 200/2
    sy = 80/2
    theta_mats = []
    for i,ego_pose in enumerate(ego_poses):
        T_ego2wld_cur = TransformationmatrixEgo(ego_pose["orientation"],ego_pose["position"])
        if i == 0:
            pose_diff = np.eye(4)
        else:
            pose_diff = np.linalg.inv(T_ego2wld_pre)@T_ego2wld_cur
        T_ego2wld_pre = T_ego2wld_cur
        yaw = np.arctan2(pose_diff[1,0], pose_diff[0,0])
        dx = pose_diff[0, 3]
        dy = pose_diff[1, 3]
        cos = np.cos(yaw)
        sin = np.sin(yaw)
        eye = np.zeros((3,3),dtype=np.float32)
        rel_pose = eye.copy()
        rel_pose[2,2]=1
        rel_pose[0,0],rel_pose[0,1],rel_pose[0,2] = cos,-sin,dx
        rel_pose[1,0],rel_pose[1,1],rel_pose[1,2] = sin,cos,dy
        pre_mat = np.array([[0., 1./sy, 0.],
                            [1./sx, 0., 1/5],
                            [0., 0., 1.]])

        post_mat = np.array([[0., sx, -sx/5],
                            [sy, 0., 0.],
                            [0., 0., 1.]])
        theta_mat = (pre_mat@(rel_pose@post_mat))[:2,:][None]
        theta_mats.append(theta_mat)
    return np.concatenate(theta_mats,0)


def rebuild_backbone(grid_conf):
    model = BEVBackbone(grid_conf, input_size=(128,384))
    return model

if __name__ == '__main__':
    grid_conf = {
        'xbound': [-80.0, 120.0, 1],
        'ybound': [-40.0, 40.0, 1],
        'zbound': [-2.0, 4.0, 1.0]
    }
    model = rebuild_backbone(grid_conf)
    T_lidar2camera = np.array([[
                    0.010424562939021826,
                    -0.99962866534536,
                    -0.024994439529313277,
                    -0.059431263039249574
                ],
                [
                    0.006448137058870251,
                    0.0250541141879979,
                    -0.9996632045797847,
                    1.744147092885818
                ],
                [
                    0.9999284955562123,
                    0.010257958549557083,
                    0.006708909083106572,
                    -1.9743571849555366
                ],
                [
                    0,
                    0,
                    0,
                    1
                ]]).reshape(4,4)
    dist_coeffs= np.array([[
                -0.034049232722088534,
                0.008472202135916486,
                0,
                0,
                -0.019044060806897155,
                0.009176172281889064,
                0,
                0
            ]]).reshape(1,8)
    camera_matrix=np.array([
                [
                    1901.8956477873126,
                    0,
                    1922.52081253954
                ],
                [
                    0,
                    1902.2429685532018,
                    1078.2301107170085
                ],
                [
                    0,
                    0,
                    1
                ]
            ]).reshape(3,3)
    ego_poses = [
        {
            "orientation": [
                0.7252925277030919,
                0.006445121526768708,
                0.004589807880389847,
                -0.688395339416375
            ],
            "position": [
                -16.00869568908437,
                45.379399067139,
                -0.543346973087456
            ]
        },
        {
            "orientation": [
                0.7254516385501073,
                0.00675402415715356,
                0.004536041108295809,
                -0.6882250559328055
            ],
            "position": [
                -16.00840644428952,
                45.38183205610659,
                -0.5425536577750744
            ]
        },
        {
            "orientation": [
                0.7256345746113962,
                0.00675949530897834,
                0.004310011176511773,
                -0.6880335724048092
            ],
            "position": [
                -16.007917008977827,
                45.383739493563155,
                -0.5423570234377117
            ]
        },
        {
            "orientation": [
                0.7257478117103694,
                0.006642957922078682,
                0.004081862107264589,
                -0.6879166543277585
            ],
            "position": [
                -16.0081965629427,
                45.38496794351029,
                -0.5429472288690743
            ]
        },
        {
            "orientation": [
                0.7257670195042981,
                0.0066346570824429504,
                0.004051102145859878,
                -0.6878966515270655
            ],
            "position": [
                -16.0086749159006,
                45.3839920241313,
                -0.5433879408122706
            ]
        },
        {
            "orientation": [
                0.7258077113155456,
                0.006781442667119014,
                0.00407377133102134,
                -0.6878521514810235
            ],
            "position": [
                -16.007274032017396,
                45.3843939298178,
                -0.5424020189052948
            ]
        },
        {
            "orientation": [
                0.7259736629900775,
                0.006877674717116843,
                0.004023298068784183,
                -0.6876763417429843
            ],
            "position": [
                -16.007786972618277,
                45.38313008782289,
                -0.5430163365819
            ]
        }
        ]

    bs = 2
    seq_len = 7 # 时序
    img_num = 1 # 环视

    T_ego_his2curs = egopose_alginhistory2current(ego_poses)
    print("T_ego_his2curs shape:", T_ego_his2curs.shape)
    theta_mats = gen_theta_mat(ego_poses)
    

    x_input = torch.zeros([bs,seq_len,img_num,3,128,384])

    rots = torch.from_numpy(T_lidar2camera[:3,:3])[None,None,None].expand(bs,seq_len,img_num,3,3) # 1 1 3 3
    trans = torch.from_numpy(T_lidar2camera[:3,3])[None,None,None,None].expand(bs,seq_len,img_num,1,3) # 1 1 1 3
    intrins = torch.from_numpy(camera_matrix)[None,None,None].expand(bs,seq_len,img_num,3,3) # 1 1 1 3
    distorts = torch.from_numpy(dist_coeffs)[None,None,None].expand(bs,seq_len,img_num,1,8) # 1 1 1 3
    post_tran3 = torch.zeros(3)
    post_rot3 = torch.eye(3)
    post_tran3 = post_tran3[None,None,None,None].expand(bs,seq_len,img_num,1,3)
    post_rot3 = post_rot3[None,None,None].expand(bs,seq_len,img_num,3,3)
    theta_mats = torch.from_numpy(theta_mats[None]).expand(bs,seq_len,2,3)
    T_ego_his2curs = torch.from_numpy(T_ego_his2curs[None]).expand(bs,seq_len,4,4)


    
    output = model(x_input,rots, trans, intrins, distorts, post_rot3, post_tran3,theta_mats,T_ego_his2curs)
    print(output.shape)
    