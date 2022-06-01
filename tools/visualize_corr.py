#!/usr/bin/env python3
import sys
import os
import argparse
import glob
import cv2
import numpy as np
import copy
from image_loader import ImageLoader

COLOR_RED = [0, 0, 255]
COLOR_GREEN = [0,255,0]

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
                        default='*color.jpg', type=str)
    parser.add_argument('--meta', dest='meta_name',
                        help='meta file pattern',
                        default='*meta.mat', type=str)
    parser.add_argument('--imgdir', dest='imgdir',
                        help='path of the directory with the test images',
                        default='data/Images', type=str)
    args = parser.parse_args()
    return args


def draw_reticle(img, x, y, label_color):
    white = (255,255,255)
    x = int(x)
    y = int(y)
    cv2.circle(img,(x,y),10,label_color,1)
    cv2.circle(img,(x,y),11,white,1)
    cv2.circle(img,(x,y),12,label_color,1)
    cv2.line(img,(x,y+1),(x,y+3),white,1)
    cv2.line(img,(x+1,y),(x+3,y),white,1)
    cv2.line(img,(x,y-1),(x,y-3),white,1)
    cv2.line(img,(x-1,y),(x-3,y),white,1)


class CorrVisualization(object):
    """
    Launches a live interactive heatmap visualization.

    Edit config/dense_correspondence/heatmap_vis/heatmap.yaml to specify which networks
    to visualize. Specifically add the network you want to visualize to the "networks" list.
    Make sure that this network appears in the file pointed to by EVAL_CONFIG

    Usage: Launch this file with python after sourcing the environment with
    `use_pytorch_dense_correspondence`

    Then `python live_heatmap_visualization.py`.

    Keypresses:
        n: new set of images
        s: swap images
        p: pause/un-pause
    """

    def __init__(self, loader):
        self._loader = loader
        self._reticle_color = COLOR_GREEN
        self._paused = False

    def _get_new_images(self):
        """
        Gets a new pair of images
        :return:
        :rtype:
        """

        self.sample1 = self._loader.get_random_sample()
        self.sample2 = self._loader.get_random_sample()

        self.img1 = self.sample1['image']
        self.img2 = self.sample2['image']

        cv2.imshow('source', self.img1)
        cv2.imshow('target', self.img2)

        self.find_best_match(None, 0, 0, None, None)


    def find_best_match(self, event, u, v, flags, param):

        """
        For each network, find the best match in the target image to point highlighted
        with reticle in the source image. Displays the result
        :return:
        :rtype:
        """

        if self._paused:
            return

        img_1_with_reticle = np.copy(self.img1)
        draw_reticle(img_1_with_reticle, u, v, self._reticle_color)
        cv2.imshow("source", img_1_with_reticle[:, :, (2, 1, 0)])

        # find correspondence
        if 'center' in self.sample1['meta'] and 'center' in self.sample2['meta']:
            RT1 = self.sample1['meta']['center']
            RT2 = self.sample2['meta']['center']
        else:
            RT1 = self.sample1['meta']['camera_pose']
            RT2 = self.sample2['meta']['camera_pose']
        delta = np.matmul(RT2, np.linalg.inv(RT1))

        x1 = np.zeros((4, 1), dtype=np.float32)
        x1[0] = self.sample1['pcloud'][v, u, 0]
        x1[1] = self.sample1['pcloud'][v, u, 1]
        x1[2] = self.sample1['pcloud'][v, u, 2]
        x1[3] = 1.0
        x2 = np.matmul(delta, x1)

        best_match_uv = np.matmul(self.sample2['meta']['intrinsic_matrix'], x2[:3])
        best_match_uv = best_match_uv / best_match_uv[2]

        if np.isfinite(best_match_uv[0]) and np.isfinite(best_match_uv[1]):
            img_2_with_reticle = np.copy(self.img2)
            draw_reticle(img_2_with_reticle, best_match_uv[0], best_match_uv[1], self._reticle_color)
            cv2.imshow("target", img_2_with_reticle[:, :, (2, 1, 0)])


    def run(self):
        self._get_new_images()
        cv2.namedWindow('target')
        cv2.setMouseCallback('source', self.find_best_match)

        self._get_new_images()

        while True:
            k = cv2.waitKey(20) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('n'):
                self._get_new_images()
            elif k == ord('p'):
                if self._paused:
                    print("un pausing")
                    self._paused = False
                else:
                    print("pausing")
                    self._paused = True


if __name__ == "__main__":
    args = parse_args()

    # list images
    images_color = []
    filename = os.path.join(args.imgdir, args.color_name)
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
    filename = os.path.join(args.imgdir, args.depth_name)
    files = glob.glob(filename)
    for i in range(len(files)):
        filename = files[i]
        images_depth.append(filename)
    images_depth.sort()

    images_meta = []
    filename = os.path.join(args.imgdir, args.meta_name)
    files = glob.glob(filename)
    for i in range(len(files)):
        filename = files[i]
        images_meta.append(filename)
    images_meta.sort()

    # image loader
    device = 'cuda:{:d}'.format(args.gpu_id)
    loader = ImageLoader(images_color, images_depth, images_meta, device=device)

    corr_vis = CorrVisualization(loader)
    print("starting correspondence vis")
    corr_vis.run()
    cv2.destroyAllWindows()

cv2.destroyAllWindows()
