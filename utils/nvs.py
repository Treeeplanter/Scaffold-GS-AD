import torch
import numpy as np
from scene.cameras import Camera
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
import copy
def random_change_view(ori_views: torch.Tensor,phase:int)->torch.Tensor:
    # 定义每个阶段的角度和 egopose 变化范围
    angle_ranges = [(0, 30), (30, 60), (60, 90)]
    egopose_ranges = [(0, 1), (1, 2), (2, 3)]
    
    # 获取当前阶段的角度和 egopose 变化范围
    angle_range = angle_ranges[phase]
    egopose_range = egopose_ranges[phase]
    zfar = 100.0
    znear = 0.01
    trans=np.array([0.0, 0.0, 0.0])
    new_views=[]
    for viewpoint_camera in ori_views:
        new_view=copy.deepcopy(viewpoint_camera)

        #这里的c2ws: x->right, y->down, z->forward
        c2w=new_view.c2w.cpu().numpy()
        FoVx=new_view.FoVx
        FoVy=new_view.FoVy

        
        # 随机生成新的角度和 egopose
        new_angle = np.random.uniform(*angle_range)
        new_angle = -45
        new_egopose = np.random.uniform(*egopose_range)
        new_egopose = 0
        # rl=np.random.uniform(0,1)
        # if rl < 0.5:
        #     new_angle = -new_angle
        new_angle_rad = np.deg2rad(new_angle)
        rotation_matrix = np.array([
                [np.cos(new_angle_rad), 0, np.sin(new_angle_rad), 0],
                [0, 1, 0, 0],
                [-np.sin(new_angle_rad), 0, np.cos(new_angle_rad), 0],
                [0, 0, 0, 1]
        ])
        # 生成一个0到360度的随机角度
        random_angle = np.random.uniform(0, 360)
        random_angle_rad = np.deg2rad(random_angle)

        # 根据随机角度和 new_egopose 计算出 x 和 y 轴的平移量
        translation_x = new_egopose * np.cos(random_angle_rad)
        translation_y = new_egopose * np.sin(random_angle_rad)

        # 创建一个平移矩阵
        translation_matrix = np.array([
            [1, 0, 0, translation_x],
            [0, 1, 0, 0],
            [0, 0, 1, translation_y],
            [0, 0, 0, 1]
        ])
        #生成新的视角
        c2w = np.dot(c2w, rotation_matrix)
        c2w = np.dot(c2w, translation_matrix)
        
        #这个R,T的来源：       
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        # w2c： waymo_world --> opencv_cam 
        world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale=1.0)).transpose(0, 1).cuda()
        # proj : opencv_cam to 0-1-NDC
        projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=FoVx, fovY=FoVy).transpose(0,1).cuda()
        # w2c + c2pixel : X_world * full_proj_transform = pixel
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        c2w = torch.tensor(c2w)
        new_view.c2w=c2w
        new_view.world_view_transform=world_view_transform
        new_view.full_proj_transform=full_proj_transform
        new_view.camera_center=camera_center
        new_views.append(new_view)
    
    return new_views
    