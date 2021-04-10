import torch
import yaml
import glob
import os, sys
import numpy as np
import cv2
import scipy.io
import matplotlib.pyplot as plt
import fewshot_cuda
from simulation_util import imread_indexed
from marker_board import MarkerBoard
from scipy.spatial.transform import Rotation as Rot
from transforms3d.quaternions import mat2quat, quat2mat

def mask_to_tight_box(mask):
    """ Return bbox given mask

        @param mask: a [H x W] numpy array
    """
    a = np.transpose(np.nonzero(mask))
    bbox = np.min(a[:, 1]), np.min(a[:, 0]), np.max(a[:, 1]), np.max(a[:, 0])
    return bbox  # x_min, y_min, x_max, y_max

class ImageLoader():

    def __init__(self, images_color, images_depth, images_meta, images_seg=None,
                 device='cuda:0',
                 cap_depth=False):
        """ TODO(ywchao): complete docstring.
        Args:
        device: A torch.device string argument. The specified device is used only
          for certain data loading computations, but not storing the loaded data.
          Currently the loaded data is always stored as numpy arrays on cpu.
         """
        assert device in ('cuda', 'cpu') or device.split(':')[0] == 'cuda'
        self._images_color = images_color
        self._images_depth = images_depth
        self._images_seg = images_seg
        self._images_meta = images_meta
        self._device = torch.device(device)
        self._cap_depth = cap_depth

        self._num_frames = len(images_color)
        self._h = 480
        self._w = 640
        self._depth_bound = 20.0
        self._marker_board = MarkerBoard()

        # tex coord
        y, x = torch.meshgrid(torch.arange(self._h), torch.arange(self._w))
        x = x.float()
        y = y.float()
        s = torch.stack((x / (self._w - 1), y / (self._h - 1)), dim=2)
        self._pcd_tex_coord = [s.numpy()]

        # colored point cloud 
        self._pcd_rgb = [np.zeros((self._h, self._w, 3), dtype=np.uint8)]
        self._pcd_vert = [np.zeros((self._h, self._w, 3), dtype=np.float32)]
        self._pcd_mask = [np.zeros((self._h, self._w), dtype=np.bool)]
        self._pcd_meta = [None]
        self._frame = 0
        self._num_cameras = 1

        self._intrinsic_matrix = np.array([[611.10888672, 0., 315.51083374],
                                          [0., 610.02844238, 237.73669434],
                                          [0, 0, 1]])
        self._master_intrinsics = self._intrinsic_matrix

    def load_frame_rgbd(self, i):
        color_file = self._images_color[i]
        color = cv2.imread(color_file)
        color = np.ascontiguousarray(color[:, :, ::-1])

        depth_file = self._images_depth[i]
        depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)

        meta_file = self._images_meta[i]
        meta = scipy.io.loadmat(meta_file)

        if self._images_seg is not None:
            seg_file = self._images_seg[i]
            label = imread_indexed(seg_file)
        else:
            seg_file = None
            label = None

        print('===================================')
        print(color_file)
        print(depth_file)
        print(seg_file)
        print(meta_file)
        print('===================================')

        return color, depth, label, meta

    def load_meta(self, i):
        meta_file = self._images_meta[i]
        print(meta_file)
        meta = scipy.io.loadmat(meta_file)
        return meta

    def deproject_depth_and_filter_points(self, depth, meta):

        if 'intrinsic_matrix' in meta:
            intrinsic_matrix = meta['intrinsic_matrix']
        else:
            intrinsic_matrix = self._intrinsic_matrix

        # backproject depth
        depth = depth.astype(np.float32) / 1000.0
        depth = torch.from_numpy(depth).to(self._device)
        fx = intrinsic_matrix[0, 0]
        fy = intrinsic_matrix[1, 1]
        px = intrinsic_matrix[0, 2]
        py = intrinsic_matrix[1, 2]
        im_pcloud = fewshot_cuda.backproject_forward(fx, fy, px, py, depth)[0]

        m = depth < self._depth_bound
        p = im_pcloud.cpu().numpy()
        m = m.cpu().numpy()
        return p, m

    def step(self):
        self._frame = (self._frame + 1) % self._num_frames
        self.update_pcd()

    def update_pcd(self):
        rgb, d, label, meta = self.load_frame_rgbd(self._frame)
        p, m = self.deproject_depth_and_filter_points(d, meta)
        meta = self.compute_marker_board_center(meta)

        self._pcd_rgb[0] = rgb
        self._pcd_vert[0] = p
        self._pcd_mask[0] = m
        self._pcd_meta[0] = meta

    def get_random_sample(self):
        idx = np.random.randint(0, self._num_frames)
        rgb, d, label, meta = self.load_frame_rgbd(idx)
        p, m = self.deproject_depth_and_filter_points(d, meta)
        meta = self.compute_marker_board_center(meta)
        sample = {'image': rgb, 'pcloud': p, 'label': label, 'meta': meta}
        return sample

    # compute marker board center with RANSAC
    def compute_marker_board_center(self, meta):

        # collect hypotheses
        keys = []
        RT_centers = []
        for key in meta:
            if 'ar_marker' in key:
                idx = int(key[-2:])
                if idx >= self._marker_board.num_markers:
                    continue
                RT_relative = self._marker_board.get_relative_pose(idx)

                pose = meta[key].flatten()
                RT = np.eye(4, dtype=np.float32)
                RT[:3, :3] = quat2mat(pose[3:])
                RT[:3, 3] = pose[:3]

                RT_final = np.matmul(RT, RT_relative)
                keys.append(key)
                RT_centers.append(RT_final)

        # compute errors for hypotheses
        num = len(keys)
        errors = np.zeros((num, ), dtype=np.float32)
        angles = np.zeros((num, ), dtype=np.float32)
        for i in range(num):
            RT_center = RT_centers[i]
            error = 0
            angle = 0
            for j in range(num):
                if j == i:
                    continue

                # pose from hypothesis
                RT_relative = self._marker_board.get_relative_pose(j)
                RT = np.matmul(RT_center, np.linalg.inv(RT_relative))

                # pose from observation
                pose = meta[keys[j]].flatten()
                RT_marker = np.eye(4, dtype=np.float32)
                RT_marker[:3, :3] = quat2mat(pose[3:])
                RT_marker[:3, 3] = pose[:3]

                # angular error
                q1 = mat2quat(RT_marker[:3, :3])
                q2 = mat2quat(RT[:3, :3])
                angle += 2 * np.arccos(np.dot(q1, q2))

                # translation error
                t1 = RT_marker[:3, 3]
                t2 = RT[:3, 3]
                distance = np.linalg.norm(t1 - t2)
                error += distance
            errors[i] = error
            angles[i] = angle

        # find the minimum error hypothesis
        if num > 0:
            index = np.argmin(errors)
            meta['center'] = RT_centers[index]
        return meta


    def process_clustering_labels(self):
        for i in range(self._num_frames):
            filename = self._images_depth[i]
            filename = filename.replace('depth', 'label-clustering')

            # read label
            label = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

            # read meta data
            meta_file = self._images_meta[i]
            meta = scipy.io.loadmat(meta_file)
            meta = self.compute_marker_board_center(meta)

            if 'center' in meta.keys():

                # get center pixel location of the marker board
                x2d = self._marker_board.project_center(meta['center'], meta['intrinsic_matrix'])

                # find the cluster closest to the center
                mask_ids = np.unique(label)
                if mask_ids[0] == 0:
                    mask_ids = mask_ids[1:]
                num = mask_ids.shape[0]
                distances = np.zeros((num, ), dtype=np.float32)

                for index, mask_id in enumerate(mask_ids):
                    mask = (label == mask_id).astype(np.float32)
                    x_min, y_min, x_max, y_max = mask_to_tight_box(mask)
                    cx = (x_min + x_max) / 2
                    cy = (y_min + y_max) / 2
                    distances[index] = (cx - x2d[0]) * (cx - x2d[0]) + (cy - x2d[1]) * (cy - x2d[1])

                if num == 0:
                    mask_final = 255 * (label > 0).astype(np.uint8)
                else:
                    index = np.argmin(distances)
                    mask_final = 255 * (label == mask_ids[index]).astype(np.uint8)
            else:
                mask_final = 255 * (label > 0).astype(np.uint8)

            # save mask
            filename = self._images_depth[i].replace('depth', 'label-binary')
            cv2.imwrite(filename, mask_final)
            print(filename)

            '''
            fig = plt.figure()
            ax = fig.add_subplot(1, 2, 1)
            plt.imshow(label)
            if 'center' in meta.keys():
                plt.plot(x2d[0], x2d[1], 'ro', markersize=3.0)
            ax = fig.add_subplot(1, 2, 2)
            plt.imshow(mask_final)
            plt.show()
            '''


    @property
    def num_frames(self):
        return self._num_frames

    @property
    def num_cameras(self):
        return self._num_cameras

    @property
    def dimensions(self):
        return self._w, self._h

    @property
    def master_intrinsics(self):
        return self._master_intrinsics

    @property
    def pcd_rgb(self):
        return self._pcd_rgb

    @property
    def pcd_vert(self):
        return self._pcd_vert

    @property
    def pcd_tex_coord(self):
        return self._pcd_tex_coord

    @property
    def pcd_mask(self):
        return self._pcd_mask

    @property
    def pcd_meta(self):
        return self._pcd_meta
