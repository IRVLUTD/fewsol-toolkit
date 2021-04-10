import os, sys
import os.path
import torch
import cv2
import numpy as np
import glob
import random
import math
import scipy.io
import matplotlib.pyplot as plt
from transforms3d.quaternions import mat2quat, quat2mat
from pathlib import Path

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

lib_path = '/home/yuxiang/GitLab/posecnn-pytorch/ycb_render'
add_path(lib_path)
from ycb_renderer import YCBRenderer

if __name__ == '__main__':

    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    folder = Path('/home/yuxiang/GitLab/few-shot-dataset/data/scenes/train/scene_00002')
    # folder = Path('/home/yuxiang/GitLab/few-shot-dataset/data/templates/003_cracker_box')
    meta_paths = sorted(list(folder.glob('meta_*.mat')))
    num = len(meta_paths)

    filename = str(meta_paths[0])
    meta = scipy.io.loadmat(filename)
    width = meta['image_width'][0][0]
    height = meta['image_height'][0][0]
    object_names = meta['object_names']
    num_obj = len(object_names)
    print(object_names)

    renderer = YCBRenderer(width, height, render_marker=False)
    obj_paths = []
    texture_paths = []
    colors = []
    for i in range(num_obj):
        cls = object_names[i].strip()
        obj_paths.append(os.path.join(root_dir, 'data', 'objects', cls, 'model_normalized.obj'))
        texture_paths.append('')
        colors.append(np.random.uniform(0, 1, size=3))
    renderer.load_objects(obj_paths, texture_paths, colors)
    renderer.set_camera_default()

    intrinsic = meta['intrinsic_matrix']
    print(intrinsic)
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    px = intrinsic[0, 2]
    py = intrinsic[1, 2]
    zfar = 6.0
    znear = 0.01

    for i in range(num):
        filename = str(meta_paths[i])
        print(filename)
        meta = scipy.io.loadmat(filename)

        # read image
        filename = filename.replace('meta', 'rgb')
        filename = filename.replace('mat', 'jpg')
        im_bullet = cv2.imread(filename)
        print(filename)

        # read depth
        filename = filename.replace('jpg', 'png')
        filename = filename.replace('rgb', 'depth')
        depth = cv2.imread(filename, cv2.IMREAD_ANYDEPTH)

        # render poses
        poses = meta['objects_in_camera']

        poses_all = []
        cls_indexes = []
        for j in range(num_obj):
            print(poses[:, :, j])
            qt = np.zeros((7, ), dtype=np.float32)
            qt[:3] = poses[:3, 3, j]
            qt[3:] = mat2quat(poses[:3, :3, j])
            print(qt)
            poses_all.append(qt)
            cls_indexes.append(j)

        renderer.set_poses(poses_all)

        # sample lighting
        renderer.set_light_pos(np.random.uniform(-0.5, 0.5, 3))
        intensity = np.random.uniform(0.8, 2)
        light_color = intensity * np.random.uniform(0.9, 1.1, 3)
        renderer.set_light_color(light_color)
            
        # rendering
        renderer.set_projection_matrix(width, height, fx, fy, px, py, znear, zfar)
        image_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
        seg_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
        pc_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
        renderer.render(cls_indexes, image_tensor, seg_tensor, pc2_tensor=pc_tensor)
        image_tensor = image_tensor.flip(0)
        seg_tensor = seg_tensor.flip(0)
        pc_tensor = pc_tensor.flip(0)

        # convert image
        im = image_tensor.cpu().numpy()
        im = np.clip(im, 0, 1) * 255
        im = im.astype(np.uint8)

        # visualization
        fig = plt.figure()
        ax = fig.add_subplot(1, 3, 1)
        plt.imshow(im_bullet[:, :, (2, 1, 0)])
        ax = fig.add_subplot(1, 3, 2)
        plt.imshow(depth)
        ax = fig.add_subplot(1, 3, 3)
        plt.imshow(im)
        plt.show()
