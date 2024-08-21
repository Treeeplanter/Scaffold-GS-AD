import torch
import os
import cv2
import scipy.interpolate
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
from utils.general_utils import PILtoTorch, DepthMaptoTorch, ObjectPILtoTorch
from utils.nvs import random_change_view
plt.rcParams['font.sans-serif'] = ['Times New Roman']
from gaussian_renderer import prefilter_voxel
import numpy as np
from utils.image_utils import psnr
import copy
@torch.no_grad()


def save_log(image,gt_image,depth,gt_depth,iteration,model_path,time_now):
    image_np = image.permute(1, 2, 0).cpu().numpy() 
    gt_np = gt_image.permute(1,2,0).cpu().numpy()
    psnr_ = psnr(image.unsqueeze(0), gt_image.unsqueeze(0)).mean().double()#batch_size=1
    label1 = f"iter:{iteration}, psnr:{psnr_:.2f}"
    times =  time_now/60
    if times < 1:
        end = "min"
    else:
        end = "mins"
    label2 = "training time:%.2f" % times + end

    depth_np = depth.permute(1, 2, 0).cpu().numpy()
    gt_depth = gt_depth.permute(1,2,0).cpu().numpy()
    mask = gt_depth > 0
    mask = np.squeeze(mask)
    depth_np[depth_np > 80] = 80
    depth_np = (depth_np - np.min(depth_np)) / (np.max(depth_np) - np.min(depth_np))*255
    gt_depth = (gt_depth - np.min(gt_depth)) / (np.max(gt_depth) - np.min(gt_depth))*255

    np_gt_depth_map = cv2.applyColorMap(cv2.convertScaleAbs(gt_depth, alpha=1.0), cv2.COLORMAP_JET)
    np_depth_map = cv2.applyColorMap(cv2.convertScaleAbs(depth_np, alpha=1.0), cv2.COLORMAP_JET)
    np_gt_depth_map[~mask] = [255, 255, 255]
    image_with_depth = gt_image.permute(1,2,0).cpu().numpy()*255
    image_with_depth[mask] = np_gt_depth_map[mask]

    # # color_map = plt.get_cmap('jet')  # 'jet' 是一种常见的彩色映射，你也可以选择其他的
    # # # np_depth_map = color_map(depth_np).squeeze(2)[:, :, :3]

    # np_depth_map = depth_np.repeat(3, axis=2)
    # np_gt_depth_map = gt_depth.repeat(3, axis=2)

    image_np_rgb = np.concatenate((gt_np, image_np), axis=1)*255
    depth_np_rgb = np.concatenate((image_with_depth, np_depth_map), axis=1)
    all_image = np.concatenate((image_np_rgb, depth_np_rgb), axis=0)

    image_with_labels = Image.fromarray(all_image.astype('uint8'))  # 转换为8位图像
    
    draw1 = ImageDraw.Draw(image_with_labels)
    font = ImageFont.truetype('./utils/TIMES.TTF', size=40)  # 请将路径替换为您选择的字体文件路径
    text_color = (255, 0, 0)  # 白色
    label1_position = (10, 10)
    label2_position = (image_with_labels.width - 150 - len(label2) * 10, 10)  # 右上角坐标
    draw1.text(label1_position, label1, fill=text_color, font=font)
    draw1.text(label2_position, label2, fill=text_color, font=font)
        
    render_base_path = os.path.join(model_path, f"render")
    image_path = os.path.join(render_base_path,"images")
    if not os.path.exists(os.path.join(model_path, f"render")):
        os.makedirs(render_base_path)
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    image_save_path = os.path.join(image_path,f"{iteration}.jpg")
    image_with_labels.save(image_save_path)


  
    


def render_nvs(scene, gaussians, viewpoints, render_func, pipe, background,iteration, time_now):

    path = os.path.join(scene.model_path, f"nvs")
    if not os.path.exists(path):
        os.makedirs(path)
    nvs_view_points = random_change_view(viewpoints,0)

    
    for viewpoint in nvs_view_points:
        #--------------------------
        #gt rgb and lidar
        #--------------------------
        nvs_path = viewpoint.file_path #like /home/thousands/Baselines/S3Gaussian/data/waymo/processed/training/036/images/090_0.jpg
        nvs_path = nvs_path.replace('_0.jpg', '_1.jpg')
        cam_name = image_path = nvs_path
        image_name = Path(cam_name).stem
        image = Image.open(image_path)
        im_data = np.array(image.convert("RGBA"))
        bg = np.array([0, 0, 0]) # d-nerf 透明背景
        norm_data = im_data / 255.0
        arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
        load_size = [640, 960]
        gt_image =PILtoTorch(image, [load_size[1], load_size[0]])
        gt_image = gt_image[:3, ...]
        gt_image = gt_image.to("cuda")
        
        ##nvs_lidar
        file_name = os.path.basename(nvs_path)
        name_parts = file_name.split('_')
        lidar_idx = name_parts[0]+".bin"
        parent_dir = os.path.dirname(nvs_path)  # 获取父目录，结果为"/home/thousands/Baselines/S3Gaussian/data/waymo/processed/training/036/images"
        parent_dir = os.path.dirname(parent_dir)
        lidar_path = os.path.join(parent_dir, "lidar", lidar_idx)

        lidar_info = np.memmap(#np.memmap是numpy库中的一个函数，它可以将大文件映射到内存中，而不是一次性加载到内存，这样可以节省内存资源。
            lidar_path,
            dtype=np.float32,
            mode="r",
        ).reshape(-1, 10) 
        #).reshape(-1, 14)
        lidar_points = lidar_info[:, 3:6]#3,4,5

        # select lidar points based on a truncated ego-forward-directional range
        # make sure most of lidar points are within the range of the camera
        truncated_min_range, truncated_max_range = -2, 80
        valid_mask = lidar_points[:, 0] < truncated_max_range#truncated_min_range, truncated_max_range = -2, 80，截断范围
        valid_mask = valid_mask & (lidar_points[:, 0] > truncated_min_range)
        lidar_points = lidar_points[valid_mask]

        lidar_to_world = viewpoint.lidar_to_world.numpy()
        
        # transform lidar points to world coordinate system

        lidar_points = (
            lidar_to_world[:3, :3] @ lidar_points.T
            + lidar_to_world[:3, 3:4]
        ).T
        
        # world-lidar-pts --> camera-pts : w2c
        #lidar_points = scene.points.points
        c2w = viewpoint.c2w
        w2c = np.linalg.inv(c2w)
        cam_points = (
            w2c[:3, :3] @ lidar_points.T
            + w2c[:3, 3:4]
        ).T
        # camera-pts --> pixel-pts : intrinsic @ (x,y,z) = (u,v,1)*z
        pixel_points = (
            viewpoint.intrinsic @ cam_points.T
        ).T
        # select points in front of the camera
        pixel_points = pixel_points[pixel_points[:, 2]>0]
        #pixel_points = pixel_points[pixel_points[:, 2]<80]
        # normalize pixel points : (u,v,1)
        image_points = pixel_points[:, :2] / pixel_points[:, 2:]
        # filter out points outside the image
        valid_mask = (
            (image_points[:, 0] >= 0)
            & (image_points[:, 0] < load_size[1])
            & (image_points[:, 1] >= 0)
            & (image_points[:, 1] < load_size[0])
        )
        pixel_points = pixel_points[valid_mask]     # pts_cam : (x,y,z)
        image_points = image_points[valid_mask]     # pts_img : (u,v)
        # compute depth map
        gt_depth_map = np.zeros(load_size)
        image_points = image_points.numpy()
        gt_depth_map[image_points[:, 1].astype(np.int32), image_points[:, 0].astype(np.int32)] = pixel_points[:, 2]#像素值是深度值
        gt_depth_map = DepthMaptoTorch( gt_depth_map)
        gt_depth_map = gt_depth_map.to("cuda")


        #--------------------------
        #render
        #--------------------------
        voxel_visible_mask = prefilter_voxel(viewpoint, gaussians, pipe,background)
        render_pkg = render_func(viewpoint, gaussians, pipe, background,visible_mask=voxel_visible_mask)
        image = render_pkg["render"]
        depth = render_pkg["depth"]
        
        save_log(image,gt_image,depth,gt_depth_map,iteration,path,time_now)

def render_training_image(scene, gaussians, viewpoints, render_func, pipe, background,iteration, time_now, nvs=False):
    def render(scene, gaussians, viewpoint, path, scaling):
        # scaling_copy = gaussians._scaling
        render_pkg = render_func( viewpoint, gaussians, pipe, background)
        label1 = f"iter:{iteration}"
        times =  time_now/60
        if times < 1:
            end = "min"
        else:
            end = "mins"
        label2 = "time:%.2f" % times + end

        image = render_pkg["render"]
        depth = render_pkg["depth"]
        #print(depth.shape)
        image_np = image.permute(1, 2, 0).cpu().numpy()  # 转换通道顺序为 (H, W, 3)
        depth_np = depth.permute(1, 2, 0).cpu().numpy()
        

        # float_img_path = path.replace(".jpg", "_float.jpg")
        # float_img = Image.fromarray((image_np * 255).astype('uint8'))  # 转换为8位图像
        # float_img.save(float_img_path)
        #depth_np = np.repeat(depth_np, 3, axis=2)
       
        if not nvs :
            gt_np = viewpoint.original_image.permute(1,2,0).cpu().numpy()
            gt_depth = viewpoint.depth_map.permute(1,2,0).cpu().numpy()
            # np.save(path.replace(".jpg", "_depth_np"), gt_depth)
            psnr_ = psnr(image.unsqueeze(0), viewpoint.original_image.unsqueeze(0)).mean().double()#batch_size=1
            label1 = f"iter:{iteration}, psnr:{psnr_:.2f}"

            # gt_depth = np.repeat(gt_depth, 3, axis=2)
            """
            # color_map = plt.get_cmap('jet')  # 'jet' 是一种常见的彩色映射，你也可以选择其他的
            # color_depth = color_map(depth_np)
            # gt_color_depth = color_map(gt_depth)
            # #print(color_depth.shape)  #(640, 960, 1, 4) 
            # # 注意:彩色映射返回的图像在最后一个维度有4个通道(RGBA).我们只需要前3个通道(RGB)
            # color_depth = color_depth.squeeze(2)
            # gt_color_depth = gt_color_depth.squeeze(2)
            # color_depth = color_depth[:, :, :3]
            # gt_color_depth = gt_color_depth[:, :, :3]
            # #print(color_depth.shape)#(640, 960, 3)
            # # depth_np = np.repeat(depth_np, 3, axis=2)
            """

        else:
            ##nvs_rgb
            nvs_path = viewpoint.file_path #like /home/thousands/Baselines/S3Gaussian/data/waymo/processed/training/036/images/090_0.jpg
            nvs_path = nvs_path.replace('_0.jpg', '_1.jpg')
            cam_name = image_path = nvs_path
            image_name = Path(cam_name).stem
            image = Image.open(image_path)
            im_data = np.array(image.convert("RGBA"))
            bg = np.array([0, 0, 0]) # d-nerf 透明背景
            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            load_size = [640, 960]
            gt_image =PILtoTorch(image, [load_size[1], load_size[0]])
            gt_image = gt_image[:3, ...]
            # print(gt_image.shape)
            gt_np = gt_image.permute(1,2,0).cpu().numpy()
            # print(gt_np.shape)
            
            ##nvs_lidar

            
            
            file_name = os.path.basename(nvs_path)
            name_parts = file_name.split('_')
            lidar_idx = name_parts[0]+".bin"
            parent_dir = os.path.dirname(nvs_path)  # 获取父目录，结果为"/home/thousands/Baselines/S3Gaussian/data/waymo/processed/training/036/images"
            parent_dir = os.path.dirname(parent_dir)
            lidar_path = os.path.join(parent_dir, "lidar", lidar_idx)

            lidar_info = np.memmap(#np.memmap是numpy库中的一个函数，它可以将大文件映射到内存中，而不是一次性加载到内存，这样可以节省内存资源。
                lidar_path,
                dtype=np.float32,
                mode="r",
            ).reshape(-1, 10) 
            #).reshape(-1, 14)
            lidar_points = lidar_info[:, 3:6]#3,4,5

            # select lidar points based on a truncated ego-forward-directional range
            # make sure most of lidar points are within the range of the camera
            truncated_min_range, truncated_max_range = -2, 80
            valid_mask = lidar_points[:, 0] < truncated_max_range#truncated_min_range, truncated_max_range = -2, 80，截断范围
            valid_mask = valid_mask & (lidar_points[:, 0] > truncated_min_range)
            lidar_points = lidar_points[valid_mask]

            lidar_to_world = viewpoint.lidar_to_world.numpy()
            
            # transform lidar points to world coordinate system

            lidar_points = (
                lidar_to_world[:3, :3] @ lidar_points.T
                + lidar_to_world[:3, 3:4]
            ).T
            
            # transform world-lidar to pixel-depth-map
            

            
            
            # world-lidar-pts --> camera-pts : w2c
            #lidar_points = scene.points.points
            c2w = viewpoint.c2w
            w2c = np.linalg.inv(c2w)
            cam_points = (
                w2c[:3, :3] @ lidar_points.T
                + w2c[:3, 3:4]
            ).T
            # camera-pts --> pixel-pts : intrinsic @ (x,y,z) = (u,v,1)*z
            pixel_points = (
                viewpoint.intrinsic @ cam_points.T
            ).T
            # select points in front of the camera
            pixel_points = pixel_points[pixel_points[:, 2]>0]
            #pixel_points = pixel_points[pixel_points[:, 2]<80]
            # normalize pixel points : (u,v,1)
            image_points = pixel_points[:, :2] / pixel_points[:, 2:]
            # filter out points outside the image
            valid_mask = (
                (image_points[:, 0] >= 0)
                & (image_points[:, 0] < load_size[1])
                & (image_points[:, 1] >= 0)
                & (image_points[:, 1] < load_size[0])
            )
            pixel_points = pixel_points[valid_mask]     # pts_cam : (x,y,z)
            image_points = image_points[valid_mask]     # pts_img : (u,v)
            # compute depth map
            depth_map = np.zeros(load_size)
            image_points = image_points.numpy()
            depth_map[image_points[:, 1].astype(np.int32), image_points[:, 0].astype(np.int32)] = pixel_points[:, 2]#像素值是深度值
            
            # depth_path = path.replace(".jpg", "_depth.jpg")
            # depth_img = Image.fromarray((depth_map * 255).astype('uint8'))  # 转换为8位图像
            # depth_img.save(depth_path)

            # depth_map = interpolate_depth_map(depth_map)
            depth_map = DepthMaptoTorch(depth_map)
            gt_depth = depth_map.permute(1,2,0).cpu().numpy()

            # gt_depth = np.repeat(gt_depth, 3, axis=2)

        # 将原始图像和渲染图像拼接在一起

        color_map = plt.get_cmap('jet')  # 'jet' 是一种常见的彩色映射，你也可以选择其他的
        depth_np/=depth_np.max()
        gt_depth/=gt_depth.max()
        color_depth = color_map(depth_np)
        gt_color_depth = color_map(gt_depth)
        #print(color_depth.shape)  #(640, 960, 1, 4) 
        # 注意：彩色映射返回的图像在最后一个维度有4个通道（RGBA），我们只需要前3个通道（RGB）
        color_depth = color_depth.squeeze(2)
        gt_color_depth = gt_color_depth.squeeze(2)
        color_depth = color_depth[:, :, :3]
        gt_color_depth = gt_color_depth[:, :, :3]

        image_np_rgb = np.concatenate((gt_np, image_np), axis=1)
        depth_np_depth = np.concatenate((gt_color_depth, color_depth), axis=1)
        image_np = np.concatenate((image_np_rgb, depth_np_depth), axis=0)
        # if "fine" in stage:
        #     feat = render_pkg['feat'] # [3,640,960]
        #     feat_np = feat.permute(1,2,0).cpu().numpy()
        #     gt_feat = viewpoint.feat_map.cpu().numpy()
        #     feat_np = np.concatenate((gt_feat, feat_np), axis=1)
        #     image_np = np.concatenate((image_np_rgb, depth_np_depth, feat_np), axis=0)
        image_with_labels = Image.fromarray((np.clip(image_np,0,1) * 255).astype('uint8'))  # 转换为8位图像
            # 创建PIL图像对象的副本以绘制标签
        draw1 = ImageDraw.Draw(image_with_labels)

        # 选择字体和字体大小
        font = ImageFont.truetype('./utils/TIMES.TTF', size=40)  # 请将路径替换为您选择的字体文件路径

        # 选择文本颜色
        text_color = (255, 0, 0)  # 白色

        # 选择标签的位置（左上角坐标）
        label1_position = (10, 10)
        label2_position = (image_with_labels.width - 150 - len(label2) * 10, 10)  # 右上角坐标

        # 在图像上添加标签
        draw1.text(label1_position, label1, fill=text_color, font=font)
        draw1.text(label2_position, label2, fill=text_color, font=font)
        

        image_with_labels.save(path)

    render_base_path = os.path.join(scene.model_path, f"render")
    point_cloud_path = os.path.join(render_base_path,"pointclouds")
    image_path = os.path.join(render_base_path,"images")
    if not os.path.exists(os.path.join(scene.model_path, f"render")):
        os.makedirs(render_base_path)
    if not os.path.exists(point_cloud_path):
        os.makedirs(point_cloud_path)
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    # image:3,800,800
    
    # point_save_path = os.path.join(point_cloud_path,f"{iteration}.jpg")
    for idx in range(len(viewpoints)):
        image_save_path = os.path.join(image_path,f"{iteration}_{idx}.jpg")
        render(scene, gaussians,viewpoints[idx],image_save_path,scaling = 1)
    # render(gaussians,point_save_path,scaling = 0.1)
    # 保存带有标签的图像

    
    
    pc_mask = gaussians.get_opacity
    pc_mask = pc_mask > 0.1
    xyz = gaussians.get_xyz.detach()[pc_mask.squeeze()].cpu().permute(1,0).numpy()
    # visualize_and_save_point_cloud(xyz, viewpoint.R, viewpoint.T, point_save_path)
    # 如果需要，您可以将PIL图像转换回PyTorch张量
    # return image
    # image_with_labels_tensor = torch.tensor(image_with_labels, dtype=torch.float32).permute(2, 0, 1) / 255.0
