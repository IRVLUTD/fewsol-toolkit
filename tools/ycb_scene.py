import random
import os
import time
import sys
import cv2
import pybullet as p
import numpy as np
import pybullet_data
import glob
import gym
import IPython
import scipy.io
import pkgutil

from gym import spaces
from PIL import Image
from transforms3d import quaternions 
from transforms3d.quaternions import quat2mat
from simulation_util import projection_to_intrinsics, view_to_extrinsics, tf_quat
import simulation_util as sim_util


class YCBEnv():
    """Class for environment with ycb objects.
         adapted from kukadiverse env in pybullet
    """

    def __init__(self,
                 urdfRoot=pybullet_data.getDataPath(),
                 actionRepeat=130,  
                 isEnableSelfCollision=True,
                 renders=False,
                 isDiscrete=False,
                 maxSteps=800,
                 dtheta=0.1,
                 blockRandom=0.5,
                 target_obj=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21], # no large clamp
                 all_objs=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21],
                 cameraRandom=0,
                 width=640,
                 height=480,
                 numObjects=6,
                 random_urdf=False,
                 egl_render=False,
                 cache_objects=False,
                 isTest=False):
        """Initializes the pandaYCBObjectEnv.

        Args:
            urdfRoot: The diretory from which to load environment URDF's.
            actionRepeat: The number of simulation steps to apply for each action.
            isEnableSelfCollision: If true, enable self-collision.
            renders: If true, render the bullet GUI.
            isDiscrete: If true, the action space is discrete. If False, the
                action space is continuous.
            maxSteps: The maximum number of actions per episode.
            blockRandom: A float between 0 and 1 indicated block randomness. 0 is
                deterministic.
            cameraRandom: A float between 0 and 1 indicating camera placement
                randomness. 0 is deterministic.
            width: The observation image width.
            height: The observation image height.
            numObjects: The number of objects in the bin.
            isTest: If true, use the test set of objects. If false, use the train
                set of objects.
        """

        self._timeStep = 1. / 1000.  
        self._urdfRoot = urdfRoot
        self._observation = []
        self._renders = renders
        self._maxSteps = maxSteps
        self._actionRepeat = actionRepeat
        self._env_step = 0

        self._cam_dist = 1.3
        self._cam_yaw = 180
        self._cam_pitch = -40
        self._root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
        self._p = p
        self._target_objs = target_obj
        self._all_objs = all_objs
        self._window_width = width
        self._window_height = height
        self._blockRandom = blockRandom
        self._cameraRandom = cameraRandom
        self._numObjects = numObjects
        self._shift = [0.0, 0.0, 0.0] # to work without axis in DIRECT mode
        self._egl_render = egl_render
        
        self._cache_objects = cache_objects
        self._object_cached = False
        self.target_idx = 0
        self.connect()

    def connect(self):
        if self._renders:
            self.cid = p.connect(p.SHARED_MEMORY)
            if (self.cid < 0):
                self.cid = p.connect(p.GUI)
                p.resetDebugVisualizerCamera(1.3, 0, -41., [0, 0, 0])
        else:
            self.cid = p.connect(p.DIRECT)
        
        egl = pkgutil.get_loader('eglRenderer')
        if self._egl_render and egl:
            p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
        else:
            p.loadPlugin("eglRendererPlugin")
        self.connected = True

    def disconnect(self): 
        p.disconnect()
        self.connected = False

    def cache_objects(self):
        """
        Load all YCB objects and set up
        """        
        obj_path = self._root_dir + '/data/objects/'
        ycb_objects = sorted([m for m in os.listdir(obj_path) if m.startswith('0')])
        print(ycb_objects)
        ycb_path = ['data/objects/' + ycb_objects[i] for i in self._all_objs]
        print(ycb_path)
        self.object_names = [ycb_objects[i] for i in self._all_objs]
        self.object_sizes = []
        
        pose = np.zeros([len(ycb_path), 3])
        pose[:, 0] = -5. - np.linspace(0, 4, len(ycb_path)) # place in the back
        orn = np.array([0, 0, 0, 1], dtype=np.float32)
        objects_paths = [p_.strip() for p_ in ycb_path] 
        objectUids = []
        self.obj_path = objects_paths + self.obj_path
        self.cache_object_poses = []

        for i, name in enumerate(objects_paths):
            trans = pose[i]
            self.cache_object_poses.append((trans.copy(), np.array(orn).copy()))
            filename = os.path.join(self._root_dir, name, 'model_normalized.urdf')
            print('loading %s' % (filename))
            uid = self._add_mesh(filename, trans, orn)  # xyzw
            objectUids.append(uid)

            # query object size
            aabb = p.getAABB(uid)
            self.object_sizes.append(aabb)

        self._object_cached = True
        self.cached_objects = [False] * len(self.obj_path)
        
        return objectUids

    def _add_mesh(self, obj_file, trans, quat, scale=1):
        bid = p.loadURDF(obj_file, trans, quat, globalScaling=scale)
        return bid


    def change_camera_pose(self, yaw=0):
        # set the camera settings.
        look = [0, 0, self.table_top_z]
        distance = 0.7
        pitch = -45
        roll = 0
        fov = 45.0
        self.near = 0.01
        self.far = 10
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(look, distance, yaw, pitch, roll, 2)
        print(self._view_matrix)
        aspect_ratio = float(self._window_width) / self._window_height
        self._proj_matrix = p.computeProjectionMatrixFOV(fov, aspect_ratio, self.near, self.far)
        self._intrinsic_matrix = projection_to_intrinsics(self._proj_matrix, self._window_width, self._window_height)


    def reset(self):
        """Environment reset  
        """

        # set table and plane
        p.resetSimulation()
        p.setTimeStep(self._timeStep)
        plane_file = os.path.join(self._root_dir, 'data/objects/floor/model_normalized.urdf')
        table_file = os.path.join(self._root_dir, 'data/objects/table/models/model_normalized.urdf')
        self.obj_path = [plane_file, table_file]
        plane_id = p.loadURDF(plane_file, [0, 0, 0])
        table_id = p.loadURDF(table_file, 0, 0, 0, 0.707, 0., 0., 0.707)

        # intialize objects
        p.setGravity(0,0,-9.81)     
        p.stepSimulation()

        # check table size
        aabb = p.getAABB(table_id)
        self.table_top_z = aabb[1][2]
        print('table top z coordinate is %f' % (self.table_top_z))

        # set the camera settings
        self.change_camera_pose()

        if not self._object_cached:
            self._objectUids = self.cache_objects()

        self._objectUids += [plane_id, table_id]
        self._env_step = 0
        return self._get_observation()

    def reset_objects(self):
        for idx, obj in enumerate(self._objectUids):
            if self.cached_objects[idx]:
                p.resetBasePositionAndOrientation(obj, self.cache_object_poses[idx][0], self.cache_object_poses[idx][1])
            self.cached_objects[idx] = False

    def cache_reset(self, scene_file=None):
        
        # self.disconnect()
        # self.connect()
        self.reset_objects()

        self._randomly_place_objects(self._get_random_object(self._numObjects)) 
        self._env_step = 0 
        self.obj_names, self.obj_poses = self.get_env_info()
        return self._get_observation()

    def _randomly_place_objects(self, urdfList, scale=1, poses=None):
        """ 
        Randomize positions of each object urdf.
        """
         
        xpos = 0.5 + 0.2 * (self._blockRandom * random.random() - 0.5)  - self._shift[0]
        ypos = 0.5 * self._blockRandom * (random.random() - 0.5)  - self._shift[0]
        orn = p.getQuaternionFromEuler([0, 0, 0])  # 

        p.resetBasePositionAndOrientation(self._objectUids[self.target_idx], \
                 [xpos, ypos, -.45 - self._shift[2]], [orn[0], orn[1], orn[2], orn[3]] )
        self.cached_objects[self.obj_path.index(urdfList[0])] = True

        for _ in range(3000):
            p.stepSimulation()

        pos = np.array([[xpos, ypos]])
        k = 0
        max_cnt = 50

        for i, name in enumerate(urdfList[1:]):
            obj_idx = self.obj_path.index(name)
            _safeDistance = np.random.uniform(self._safeDistance - 0.02, self._safeDistance)
            cnt = 0
            if self.cached_objects[obj_idx]:
                continue
          
            while cnt < max_cnt:
                cnt += 1
                xpos_ = xpos - self._blockRandom * 0.3 * random.random()
                ypos_ = ypos - self._blockRandom * 1 * (random.random() - 0.5) # 0.5
                xy = np.array([[xpos_, ypos_]])
            
                if np.amin(np.linalg.norm(xy - pos, axis=-1)) > _safeDistance and \
                    (xpos_ > 0.35- self._shift[0] and xpos_ < 0.65- self._shift[0]) and \
                    (ypos_ < 0.20- self._shift[1] and ypos_ > -0.20- self._shift[1]): # 0.15
                    break
       
            pos = np.concatenate([pos, xy], axis=0)
            xpos = xpos_
            angle = np.random.uniform(-np.pi, np.pi)
            orn = p.getQuaternionFromEuler([0, 0, angle])
            p.resetBasePositionAndOrientation(self._objectUids[obj_idx], 
                            [xpos, ypos_, -.35- self._shift[2]], [orn[0], orn[1], orn[2], orn[3]] )  # xyzw
            self.cached_objects[obj_idx] = True

            for _ in range(5000):
                p.stepSimulation()
        
        return []


    def place_an_object_table_center(self, target_idx, x=0, yaw=0, pitch=0, roll=0):
        """ 
        Randomize positions of each object urdf.
        """

        # get object size in z direction
        aabb = self.object_sizes[target_idx]

        # object pose         
        xpos = x
        ypos = 0
        if yaw == 0:
            zpos = self.table_top_z + abs(aabb[0][2])
        else:
            zpos = self.table_top_z + abs(aabb[1][2])
        orn = p.getQuaternionFromEuler([yaw, pitch, roll])

        p.resetBasePositionAndOrientation(self._objectUids[target_idx], \
                 [xpos, ypos, zpos], [orn[0], orn[1], orn[2], orn[3]])

        for _ in range(0):
            p.stepSimulation()

        return self._get_observation()


    def _get_random_object(self, num_objects, ycb=True):
        """
        Randomly choose an object urdf from the selected objects
        """         
         
        self.target_idx = self._all_objs.index(self._target_objs[np.random.randint(0, len(self._target_objs))])  #
        obstacle = np.random.choice(range(len(self._all_objs)), self._numObjects - 1).tolist()
        selected_objects = [self.target_idx] + obstacle
        selected_objects_filenames = [self.obj_path[selected_object] for selected_object in selected_objects]
        return selected_objects_filenames

    def _get_observation(self):

        _, _, rgba, depth, mask = p.getCameraImage(width=self._window_width,
                                 height=self._window_height,
                                 viewMatrix=self._view_matrix,
                                 projectionMatrix=self._proj_matrix,
                                 physicsClientId=self.cid)

        depth = self.far * self.near / (self.far - (self.far - self.near) * depth)
        obs = np.concatenate([rgba[..., :3], depth[...,None], mask[...,None]], axis=-1)
        return obs

    def _get_target_obj_pose(self):
        return p.getBasePositionAndOrientation(self._objectUids[self.target_idx])[0]

    def get_env_info(self, add_table_floor=False):
        poses = []
        obj_dir = []

        for idx, uid in enumerate(self._objectUids[:-2]):
            if self.cached_objects[idx]:           
                pos, orn = p.getBasePositionAndOrientation(uid) # center offset of base
                obj_pose = list(pos) + [orn[3], orn[0], orn[1], orn[2]]
                poses.append(obj_pose)
                obj_dir.append(self.obj_path[idx]) # .encode("utf-8")
        
        if add_table_floor: # compatibility
            obj_dir.extend(['data/' + '/'.join(self.obj_path[-2].split('/')[-3:-1]),
                            'data/' + '/'.join(self.obj_path[-1].split('/')[-4:-1])])
            for i in range(2):
                pos, orn = p.getBasePositionAndOrientation(self._objectUids[i]) # center offset of base
                obj_pose = list(pos) + [orn[3], orn[0], orn[1], orn[2]]
                poses.append(obj_pose)
 
        return obj_dir, poses


    def save_data(self, target_idx, obs, folder, count):

        # rgb
        filename = os.path.join(folder, 'rgb_%05d.jpg' % count)
        im = obs[:, :, :3]
        cv2.imwrite(filename, im[:, :, (2, 1, 0)])

        # depth
        depth = obs[:, :, 3]
        depth_32 = depth * 1000
        depth_cv = np.array(depth_32, dtype=np.uint16)
        filename = os.path.join(folder, 'depth_%05d.png' % count)
        cv2.imwrite(filename, depth_cv)

        # segmentation mask
        mask = obs[:, :, 4]
        mask[mask > 1] = 2
        filename = os.path.join(folder, 'segmentation_%05d.png' % count)
        sim_util.imwrite_indexed(filename, mask.astype(np.uint8))

        # meta data
        meta = {'intrinsic_matrix': self._intrinsic_matrix}
        meta['image_width'] = self._window_width
        meta['image_height'] = self._window_height

        object_names = []
        object_names.append(self.object_names[target_idx])
        meta['object_names'] = object_names
        meta[self.object_names[target_idx]] = 2

        camera_pose = view_to_extrinsics(self._view_matrix)
        meta['camera_pose'] = camera_pose
        print(meta['object_names'])
        print('camera pose')
        print(camera_pose)

        # object pose in world and in camera
        object_poses = np.zeros((4, 4, 1), dtype=np.float32)
        objects_in_camera = np.zeros((4, 4, 1), dtype=np.float32)

        uid = self._objectUids[target_idx]
        pos, orn = p.getBasePositionAndOrientation(uid)
        object_pose = np.eye(4, dtype=np.float32)
        object_pose[:3, :3] = quat2mat(tf_quat(orn))
        object_pose[:3, 3] = pos

        object_poses[:, :, 0] = object_pose
        meta['object_poses'] = object_poses
        print('object poses', object_poses)

        # relative pose
        objects_in_camera[:, :, 0] = np.matmul(camera_pose, object_pose)
        meta['objects_in_camera'] = objects_in_camera
        print('objects_in_camera', meta['objects_in_camera'])
        print('=============================')
        
        filename = os.path.join(folder, 'meta_%05d.mat' % count)
        scipy.io.savemat(filename, meta, do_compression=True)

 
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', help='filename', type=str,
        default='scene_5')
    parser.add_argument('-d', '--dir', help='filename', type=str,
        default='../data/templates/')  
    args = parser.parse_args()

    # setup bullet env
    target_obj = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20]  # no large clamp
    # target_obj = [1]
    env = YCBEnv(renders=True, egl_render=False, target_obj=target_obj, all_objs=target_obj)
    env.reset()

    # put one object onto the table
    num = len(target_obj)
    for i in range(num):
        target_idx = i

        folder = args.dir + env.object_names[target_idx]
        if not os.path.exists(folder):
            os.mkdir(folder)

        count = 0
        for i in range(2):
            if i == 0:
                env.place_an_object_table_center(target_idx, yaw=0)
            else:
                env.place_an_object_table_center(target_idx, yaw=np.pi)

            for yaw in range(0, 360, 30):
                env.change_camera_pose(yaw)
                obs = env._get_observation()
                env.save_data(target_idx, obs, folder, count)
                count += 1

        # move object away
        env.place_an_object_table_center(target_idx, x=-5, yaw=0)
