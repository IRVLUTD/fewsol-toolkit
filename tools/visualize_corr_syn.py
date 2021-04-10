#!/usr/bin/env python3
import sys
import os
import argparse
import glob
import cv2
import numpy as np
import copy
import scipy.io
from transforms3d.quaternions import mat2quat, quat2mat
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
                        default='*color.png', type=str)
    parser.add_argument('--seg', dest='seg_name',
                        help='segmentation file pattern',
                        default='*segmentation.png', type=str)
    parser.add_argument('--meta', dest='meta_name',
                        help='meta file pattern',
                        default='*meta.mat', type=str)
    parser.add_argument('--imgdir', dest='imgdir',
                        help='path of the directory with the test images',
                        default='data/Images', type=str)
    parser.add_argument('--tmpdir', dest='tmpdir',
                        help='path of the directory with the template images',
                        default='data/templates', type=str)
    args = parser.parse_args()
    return args


def draw_reticle(img, x, y, label_color):
    white = (255,255,255)
    cv2.circle(img,(x,y),10,label_color,1)
    cv2.circle(img,(x,y),11,white,1)
    cv2.circle(img,(x,y),12,label_color,1)
    cv2.line(img,(x,y+1),(x,y+3),white,1)
    cv2.line(img,(x+1,y),(x+3,y),white,1)
    cv2.line(img,(x,y-1),(x,y-3),white,1)
    cv2.line(img,(x-1,y),(x-3,y),white,1)


def load_templates(folder, object_names, height, width):
    
    templates = []
    num = len(object_names)
    for i in range(num):
        cls = object_names[i].strip()
        filename = os.path.join(folder, cls, 'meta_*.mat')
        meta_files = sorted(glob.glob(filename))

        template = {'meta_files': meta_files}
        template['object_name'] = cls
        print(meta_files)

        # pose
        n = len(meta_files)
        poses = np.zeros((4, 4, n), dtype=np.float32)
        depths = np.zeros((height, width, n), dtype=np.float32)
        for j in range(n):
            filename = meta_files[j]
            meta = scipy.io.loadmat(filename)
            pose = meta['objects_in_camera'][:, :, 0]
            poses[:, :, j] = pose

            # depth
            depth_filename = filename.replace('meta', 'depth')
            depth_filename = depth_filename.replace('.mat', '.png')
            depth = cv2.imread(depth_filename, cv2.IMREAD_ANYDEPTH)
            depths[:, :, j] = depth.astype(np.float32) / 1000.0

        template['poses'] = poses
        template['depths'] = depths
        templates.append(template)

    return templates



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

    def __init__(self, loader, templates):
        self._loader = loader
        self._templates = templates
        self._reticle_color = COLOR_GREEN
        self._paused = False


    def _get_template(self, sample):

        meta = sample['meta']
        pcloud = sample['pcloud']
        label = sample['label']
        width = meta['image_width'][0][0]
        height = meta['image_height'][0][0]

        # sample an object
        index = np.random.randint(0, len(self._templates))
        template = self._templates[index]
        poses_tmp = template['poses']
        depths_tmp = template['depths']
        object_name = template['object_name'].strip()

        # find the mask of the object
        cls_index = meta[object_name][0][0]
        print(object_name, cls_index)

        # 3D points of the object
        x3d = pcloud[label == cls_index]
        num = x3d.shape[0]
        x3d = np.concatenate((x3d, np.ones((num, 1), dtype=np.float32)), axis=1)

        # find the object pose
        for i in range(len(meta['object_names'])):
            cls = meta['object_names'][i]
            if object_name == cls.strip():
                pose = meta['objects_in_camera'][:, :, i]
                break

        # find the most visible template
        K = meta['intrinsic_matrix']
        num = poses_tmp.shape[2]
        count = np.zeros((num, ), dtype=np.float32)
        for i in range(num):
            # transform points
            RT = poses_tmp[:, :, i]
            x3d_tmp = np.matmul(RT, np.linalg.inv(pose).dot(x3d.transpose()))
            z_values = x3d_tmp[2, :]

            # projection
            x2d = np.matmul(K, x3d_tmp[:3, :])
            x1 = np.round(x2d[0, :] / x2d[2, :]).astype(np.int32)
            y1 = np.round(x2d[1, :] / x2d[2, :]).astype(np.int32)
            x1[x1 < 0] = 0
            x1[x1 >= width] = width - 1
            y1[y1 < 0] = 0
            y1[y1 >= height] = height - 1

            # query depth
            z_tmp = depths_tmp[y1, x1, i]

            # compare depth
            diff = abs(z_tmp - z_values)
            count[i] = np.sum(diff < 0.01)

        # read file
        index = np.argmax(count)
        filename = template['meta_files'][index]
        print(index, count)
        print(filename)
        meta_template = scipy.io.loadmat(filename)

        rgb_filename = filename.replace('meta', 'rgb')
        rgb_filename = rgb_filename.replace('.mat', '.jpg')
        print(rgb_filename)
        color = cv2.imread(rgb_filename)
        color = np.ascontiguousarray(color[:, :, ::-1])

        depth_filename = filename.replace('meta', 'depth')
        depth_filename = depth_filename.replace('.mat', '.png')
        print(depth_filename)
        depth = cv2.imread(depth_filename, cv2.IMREAD_ANYDEPTH)

        p, m = self._loader.deproject_depth_and_filter_points(depth, meta_template)
        RT_template = poses_tmp[:, :, index]
        RT_object = pose
        sample = {'image': color, 'pcloud': p, 'meta': meta_template, 'RT_template': RT_template, 'RT_object': RT_object}
        return sample


    def _get_new_images(self):
        """
        Gets a new pair of images
        :return:
        :rtype:
        """

        self.sample1 = self._loader.get_random_sample()
        self.sample2 = self._get_template(self.sample1)

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
        RT1 = self.sample2['RT_object']
        RT2 = self.sample2['RT_template']
        delta = np.matmul(RT2, np.linalg.inv(RT1))

        x1 = np.zeros((4, 1), dtype=np.float32)
        x1[0] = self.sample1['pcloud'][v, u, 0]
        x1[1] = self.sample1['pcloud'][v, u, 1]
        x1[2] = self.sample1['pcloud'][v, u, 2]
        x1[3] = 1.0
        x2 = np.matmul(delta, x1)

        best_match_uv = np.matmul(self.sample2['meta']['intrinsic_matrix'], x2[:3])
        best_match_uv = best_match_uv / best_match_uv[2]

        img_2_with_reticle = np.copy(self.img2)
        draw_reticle(img_2_with_reticle, best_match_uv[0], best_match_uv[1], self._reticle_color)
        cv2.imshow("target", img_2_with_reticle[:, :, (2, 1, 0)])


    def run(self):
        self._get_new_images()
        cv2.namedWindow('target')
        cv2.setMouseCallback('source', self.find_best_match)

        # self._get_new_images()

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

    # detph
    images_depth = []
    filename = os.path.join(args.imgdir, args.depth_name)
    files = glob.glob(filename)
    for i in range(len(files)):
        filename = files[i]
        images_depth.append(filename)
    images_depth.sort()

    # segmentation
    images_seg = []
    filename = os.path.join(args.imgdir, args.seg_name)
    files = glob.glob(filename)
    for i in range(len(files)):
        filename = files[i]
        images_seg.append(filename)
    images_seg.sort()

    # meta data
    images_meta = []
    filename = os.path.join(args.imgdir, args.meta_name)
    files = glob.glob(filename)
    for i in range(len(files)):
        filename = files[i]
        images_meta.append(filename)
    images_meta.sort()

    # image loader
    device = 'cuda:{:d}'.format(args.gpu_id)
    loader = ImageLoader(images_color, images_depth, images_meta, images_seg, device)

    # templates
    meta = loader.load_meta(0)
    width = meta['image_width'][0][0]
    height = meta['image_height'][0][0]
    object_names = meta['object_names']
    templates = load_templates(args.tmpdir, object_names, height, width)

    corr_vis = CorrVisualization(loader, templates)
    print("starting correspondence vis")
    corr_vis.run()
    cv2.destroyAllWindows()

cv2.destroyAllWindows()
