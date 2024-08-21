import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch

from depth_anything_v2.dpt import DepthAnythingV2


def DepthEstimate(img_paths, outdir, input_size = 518, encoder = 'vitl', max_depth=80):
    
    load_from ='/home/thousands/Baselines/ls/Scaffold-GS-AD/submodules/Depth_Anything_V2/metric_depth/weight/depth_anything_v2_metric_vkitti_vitl.pth'

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    depth_anything.load_state_dict(torch.load(load_from , map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    depth_images = []
    for k, filename in enumerate(img_paths):
        print(f'Progress {k+1}/{len(img_paths)}: {filename}')
        
        raw_image = cv2.imread(filename)
        
        depth = depth_anything.infer_image(raw_image, input_size)
        depth_images.append(depth)

        output_path = os.path.join(outdir, f"{k:03d}.npy")
        
        np.save(output_path, depth)

    return depth_images