#!/usr/bin/env python3
import os
import time
import json
import cv2
import scipy.io
import numpy as np
from simulation_util import imread_indexed
from matplotlib import pyplot as plt


def load_object_rgbd(scene_folder, i):
    color_file = os.path.join(scene_folder, '%06d-color.jpg' % i)
    color = cv2.imread(color_file)
    color = np.ascontiguousarray(color[:, :, ::-1])

    depth_file = os.path.join(scene_folder, '%06d-depth.png' % i)
    depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)

    meta_file = os.path.join(scene_folder, '%06d-meta.mat' % i)
    meta = scipy.io.loadmat(meta_file)

    seg_file = os.path.join(scene_folder, '%06d-label-binary.png' % i)
    label = imread_indexed(seg_file)

    return color, depth, label, meta


def load_frame_rgbd(scene_folder, i):
    color_file = os.path.join(scene_folder, 'rgb_%05d.jpg' % i)
    color = cv2.imread(color_file)
    color = np.ascontiguousarray(color[:, :, ::-1])

    depth_file = os.path.join(scene_folder, 'depth_%05d.png' % i)
    depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)

    meta_file = os.path.join(scene_folder, 'meta_%05d.mat' % i)
    meta = scipy.io.loadmat(meta_file)

    seg_file = os.path.join(scene_folder, 'segmentation_%05d.png' % i)
    label = imread_indexed(seg_file)

    print('===================================')
    print(color_file)
    print(depth_file)
    print(seg_file)
    print(meta_file)
    print('===================================')

    return color, depth, label, meta



if __name__ == '__main__':

    root_dir = '/home/yuxiang/Datasets/FewSOL'
    scene_dir = root_dir + '/google_scenes/train'
    obj_dir = root_dir + '/synthetic_objects'
    
    subdirs = sorted(os.listdir(scene_dir))
    num = len(subdirs)

    # load mesh names
    filename = '../data/synthetic_objects_folders.txt'
    meshes = []
    with open(filename) as f:
        for line in f:
            meshes.append(line.strip())

    # for each scene
    for i in range(num):
        # read scene description
        filename = os.path.join(scene_dir, subdirs[i], 'scene_description.txt')
        print(filename)

        scene_description = json.load(open(filename))
        print(scene_description.keys())

        # get object names
        objects = scene_description['object_descriptions']
        obj_names = []
        for obj in objects:
            mesh_filename = obj['mesh_filename']
            names = mesh_filename.split('/')
            # get object name
            obj_name = names[-3]
            obj_names.append(obj_name)
        print(obj_names)
        n = len(obj_names)
        
        # load one image
        scene_folder = os.path.join(scene_dir, subdirs[i])
        index = np.random.randint(0, 7)
        color, depth, label, meta = load_frame_rgbd(scene_folder, index)
        
        # visualization
        fig = plt.figure()
        ax = fig.add_subplot(1+n, 3, 1)
        plt.title('color')
        plt.imshow(color)
        ax = fig.add_subplot(1+n, 3, 2)
        plt.title('depth')
        plt.imshow(depth)
        ax = fig.add_subplot(1+n, 3, 3)
        plt.title('label')        
        plt.imshow(label)
        
        for j in range(n):
            obj_name = obj_names[j]
            scene_folder = os.path.join(obj_dir, obj_name)
            color, depth, label, meta = load_object_rgbd(scene_folder, 0)
            ax = fig.add_subplot(1+n, 3, 3 + j*3 + 1)
            plt.imshow(color) 
            
            color, depth, label, meta = load_object_rgbd(scene_folder, 1)
            ax = fig.add_subplot(1+n, 3, 3 + j*3 + 2)
            plt.imshow(color)  
            
            color, depth, label, meta = load_object_rgbd(scene_folder, 2)
            ax = fig.add_subplot(1+n, 3, 3 + j*3 + 3)
            plt.imshow(color)                                               
        
        plt.show()
