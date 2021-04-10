#!/usr/bin/env python3
import sys
import os
import argparse
import glob
import cv2
import numpy as np
import copy
from image_loader import ImageLoader


def parse_args():
    parser = argparse.ArgumentParser(
        description='View point cloud and ground-truth hand & object poses in 3D.'
    )
    parser.add_argument('--name',
                        help="Name of the sequence",
                        default=None,
                        type=str)
    parser.add_argument('--no-preload', action='store_true', default=False)
    parser.add_argument('--use-cache', action='store_true', default=False)
    parser.add_argument('--device',
                        help='Device for data loader computation',
                        default='cuda:0',
                        type=str)
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--depth', dest='depth_name',
                        help='depth image pattern',
                        default='*depth.png', type=str)
    parser.add_argument('--color', dest='color_name',
                        help='color image pattern',
                        default='*color.png', type=str)
    parser.add_argument('--meta', dest='meta_name',
                        help='meta file pattern',
                        default='*meta.mat', type=str)
    parser.add_argument('--imgdir', dest='imgdir',
                        help='path of the directory with the test images',
                        default='data/Images', type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # list subdirs
    subdirs = os.listdir(args.imgdir)
    print(subdirs)

    for dirname in subdirs:

        # list images
        images_color = []
        filename = os.path.join(args.imgdir, dirname, args.color_name)
        print(filename)
        files = glob.glob(filename)
        for i in range(len(files)):
            filename = files[i]
            images_color.append(filename)
        images_color.sort()

        if len(images_color) == 0:
            print('no images in path %s' % (filename))
            sys.exit(1)

        images_depth = []
        filename = os.path.join(args.imgdir, dirname, args.depth_name)
        files = glob.glob(filename)
        for i in range(len(files)):
            filename = files[i]
            images_depth.append(filename)
        images_depth.sort()

        images_meta = []
        filename = os.path.join(args.imgdir, dirname, args.meta_name)
        files = glob.glob(filename)
        for i in range(len(files)):
            filename = files[i]
            images_meta.append(filename)
        images_meta.sort()

        # image loader
        device = 'cuda:{:d}'.format(args.gpu_id)
        loader = ImageLoader(images_color, images_depth, images_meta, device)
        loader.process_clustering_labels()
