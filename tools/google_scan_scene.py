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
from pyquaternion import Quaternion
from geo.transform import Transform
from simulation_util import projection_to_intrinsics, view_to_extrinsics, tf_quat
import simulation_util as sim_util


class GoogleEnv():
    """Class for environment with google scanned objects.
         adapted from kukadiverse env in pybullet
    """

    def __init__(self,
                 model_dir,
                 model_names,
                 urdfRoot=pybullet_data.getDataPath(),
                 actionRepeat=130,  
                 isEnableSelfCollision=True,
                 renders=False,
                 isDiscrete=False,
                 maxSteps=800,
                 dtheta=0.1,
                 blockRandom=0.5,
                 cameraRandom=0,
                 width=640,
                 height=480,
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

        self.model_names = model_names
        self._model_dir = model_dir
        self._model_names = model_names
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
        self._window_width = width
        self._window_height = height
        self._blockRandom = blockRandom
        self._cameraRandom = cameraRandom
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
        Load all objects and set up
        """
        object_paths = [os.path.join(self._model_dir, name, 'meshes', 'model.obj') for name in self._model_names]
        self.obj_path = object_paths
        self.cache_object_poses = []
        self.object_sizes = []
        objectUids = []
        
        pose = np.zeros([len(object_paths), 3])
        pose[:, 0] = -5. - np.linspace(0, 4, len(object_paths)) # place in the back
        orn = np.array([0, 0, 0, 1], dtype=np.float32)

        for i, filename in enumerate(object_paths):
            trans = pose[i]
            self.cache_object_poses.append((trans.copy(), np.array(orn).copy()))

            transform = Transform(translation=trans, 
                                  rotation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0),
                                  scale=np.ones(3))

            print('loading %d: %s' % (i, filename))
            uid = self._add_mesh(i, filename, transform)  # xyzw
            objectUids.append(uid)

            # query object size
            aabb = p.getAABB(uid)
            self.object_sizes.append(aabb)

        self._object_cached = True
        self.cached_objects = [False] * len(self.obj_path)
        
        return objectUids


    def _add_mesh(self, obj_id, obj_file, transform, vis_mesh_file=None, texture_file='', static=False):

        if static:
            cid = p.createCollisionShape(p.GEOM_MESH, fileName=obj_file, meshScale=transform.scale,
                                         flags=p.GEOM_FORCE_CONCAVE_TRIMESH, physicsClientId=self._pid)
        else:
            cid = p.createCollisionShape(p.GEOM_MESH, fileName=obj_file, meshScale=transform.scale,
                                         physicsClientId=self.cid)

        vid = -1
        if vis_mesh_file:
            vid = p.createVisualShape(p.GEOM_MESH, fileName=vis_mesh_file, meshScale=transform.scale,
                                      physicsClientId=self.cid)

        rot_q = np.roll(transform.rotation.elements, -1)  # w,x,y,z -> x,y,z,w (which pybullet expects)
        mass = 0 if static else 1
        bid = p.createMultiBody(baseMass=mass,
                                baseCollisionShapeIndex=cid,
                                baseVisualShapeIndex=vid,
                                basePosition=transform.translation,
                                baseOrientation=rot_q,
                                physicsClientId=self.cid)
        return bid


    def change_camera_pose(self, yaw=0, pitch=-45, roll=0, distance=0.7):
        # set the camera settings.
        look = [0, 0, self.table_top_z]
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


    def save_data(self, target_idx, obs, folder, count):

        # rgb
        filename = os.path.join(folder, '%06d-color.jpg' % count)
        im = obs[:, :, :3]
        cv2.imwrite(filename, im[:, :, (2, 1, 0)])

        # depth
        depth = obs[:, :, 3]
        depth_32 = depth * 1000
        depth_cv = np.array(depth_32, dtype=np.uint16)
        filename = os.path.join(folder, '%06d-depth.png' % count)
        cv2.imwrite(filename, depth_cv)

        # segmentation mask
        mask = obs[:, :, 4]
        label = mask.copy()
        label[mask > 1] = 255
        label[mask <= 1] = 0
        filename = os.path.join(folder, '%06d-label-binary.png' % count)
        cv2.imwrite(filename, label.astype(np.uint8))

        # meta data
        meta = {'intrinsic_matrix': self._intrinsic_matrix}
        meta['image_width'] = self._window_width
        meta['image_height'] = self._window_height

        object_names = []
        object_names.append(self.model_names[target_idx])
        meta['object_names'] = object_names

        camera_pose = view_to_extrinsics(self._view_matrix)
        meta['camera_pose'] = camera_pose
        print(meta['object_names'])

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

        # relative pose
        objects_in_camera[:, :, 0] = np.matmul(camera_pose, object_pose)
        meta['objects_in_camera'] = objects_in_camera
        print('=============================')
        
        filename = os.path.join(folder, '%06d-meta.mat' % count)
        scipy.io.savemat(filename, meta, do_compression=True)

 
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', help='filename', type=str,
        default='scene_5')
    parser.add_argument('-d', '--dir', help='filename', type=str,
        default='../data/synthetic_objects/')
    parser.add_argument('-m', '--model_dir', help='filename', type=str,
        default='../data/google_scan_selected/')
    args = parser.parse_args()

    # list models
    model_names = sorted(os.listdir(args.model_dir))
    model_names = model_names[170:]

    # setup bullet env
    env = GoogleEnv(args.model_dir, model_names, renders=True, egl_render=False)
    env.reset()

    # put one object onto the table
    num = len(model_names)
    for i in range(num):
        target_idx = i

        folder = args.dir + env.model_names[target_idx]
        if not os.path.exists(folder):
            os.mkdir(folder)

        count = 0
        roll = 0
        distance = 0.8
        env.place_an_object_table_center(target_idx, yaw=0)
        for j in range(9):
            if j == 0:
                yaw = 0
                pitch = -30
            elif j == 1:
                yaw = 0
                pitch = -45
            elif j == 2:
                yaw = 0
                pitch = -60
            elif j == 3:
                yaw = 45
                pitch = -30
            elif j == 4:
                yaw = 45
                pitch = -45
            elif j == 5:
                yaw = 45
                pitch = -60
            elif j == 6:
                yaw = -45
                pitch = -30
            elif j == 7:
                yaw = -45
                pitch = -45
            elif j == 8:
                yaw = -45
                pitch = -60

            env.change_camera_pose(yaw, pitch, roll, distance)
            obs = env._get_observation()
            env.save_data(target_idx, obs, folder, count)
            count += 1

        # move object away
        env.place_an_object_table_center(target_idx, x=-5, yaw=0)
