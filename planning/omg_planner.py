import _init_paths
import time
import numpy as np
import datetime
import scipy
import omg.config as config
from omg.core import PlanningScene
from omg.util import wrap_value, relative_pose, pack_pose, compose_pose

# OMG Planner
class OMGPlanner(object):

    def __init__(self):

        # config planner
        config.cfg.traj_init = 'grasp'
        config.cfg.scene_file = ''
        config.cfg.ol_alg = 'MD'
        config.cfg.cam_V = np.array([[-0.39, -0.92,  0.,    0.44],
                                     [-0.57,  0.24, -0.78,  0.64],
                                     [ 0.72, -0.31, -0.62,  2.55 + 0.5],
                                     [ 0.,    0.,    0.,    1.  ]])
        '''
        config.cfg.cam_V = np.array([[ 0.5471, -0.8371,  0.0049,  0.    ],
                                     [-0.4446, -0.2956, -0.8455,  0.    ],
                                     [ 0.7092,  0.4604, -0.5339,  2.6657 + 0.5],
                                     [ 0.    ,  0.    ,  0.    ,  1.    ]], dtype=np.float32)
        '''

        self.scene = PlanningScene(config.cfg)


    def populate_scene(self, object_names, model_names, object_poses, flag_compute_grasp):
        for i, pose in enumerate(object_poses):  
            print(object_names[i], model_names[i])
            self.scene.env.add_object(object_names[i], pose[:3], pose[3:], compute_grasp=flag_compute_grasp[i], model_name=model_names[i])
        self.scene.env.combine_sdfs()


    def update_objects(self, object_names, model_names, object_poses):
        for i, pose in enumerate(object_poses):
            name = object_names[i]
            if name in self.scene.env.names:
                index = self.scene.env.names.index(name)
                self.scene.env.update_pose(name, pose)
            else:
                print('adding %s to planner' % (name))
                self.scene.insert_object(name, pose[:3], pose[3:], model_name=model_names[i])
    

    def plan_to_target(self, start_conf, target_name):
        self.scene.traj.start = start_conf
        self.scene.env.set_target(target_name)
        self.scene.update_planner()
        info = self.scene.step()
        joint_trajectory_points = np.concatenate([self.scene.planner.history_trajectories[-1]], axis=0)
        return joint_trajectory_points, info[-1]['execute'], info[-1]['standoff_idx']


    def plan_to_conf(self, start_conf, end_conf, base_obstacle_weight=1.0, disable_list=[]):
        self.scene.traj.start = start_conf
        dummy_obstacle_weight = config.cfg.base_obstacle_weight
        config.cfg.base_obstacle_weight = base_obstacle_weight
        dummy_goal_set = config.cfg.goal_set_proj
        config.cfg.disable_collision_set = disable_list # add handle and top drawer here
        setattr(config.cfg, 'goal_set_proj', False)
        dummy_standoff = config.cfg.use_standoff
        config.cfg.use_standoff = False
        config.cfg.get_global_param()
        self.scene.traj.end = end_conf
        self.scene.reset(lazy=True)
        info = self.scene.step()
        joint_trajectory_points = np.concatenate((self.scene.traj.start[np.newaxis, :], \
            self.scene.traj.data, self.scene.traj.end[np.newaxis, :]), axis=0) 
        setattr(config.cfg, 'goal_set_proj', dummy_goal_set)
        setattr(config.cfg, "use_standoff", dummy_standoff)
        config.cfg.disable_collision_set = []
        config.cfg.base_obstacle_weight = dummy_obstacle_weight
        config.cfg.get_global_param()
        return joint_trajectory_points, info[-1]['execute']


    def plan_to_place_target(self, start_conf, target_name, place_translation=[0.0, -0.3, 0], is_delta=True, apply_standoff=False, debug=False):
        """
        place a placement trajectory for a target object to a delta translation
        """
        self.scene.traj.start = start_conf
        print('start conf in place', start_conf)
        config.cfg.disable_collision_set = [target_name]
        dummy_standoff = config.cfg.use_standoff
        config.cfg.use_standoff = apply_standoff
        if apply_standoff:
            config.cfg.increment_iks = True;
        obj_idx = self.scene.env.names.index(target_name)

        # compute relative pose and attach object
        target = self.scene.env.objects[obj_idx]
        grasp_pose = target.pose.copy()
        start_joints = wrap_value(start_conf)
        robot = self.scene.env.robot
        start_hand_pose = robot.robot_kinematics.forward_kinematics_parallel(
            start_joints[None,...], base_link=config.cfg.base_link)[0][7]

        target.rel_hand_pose = relative_pose(pack_pose(start_hand_pose), grasp_pose) # object -> hand
        print('rel hand pose %.6f, %.6f, %.6f' % (target.rel_hand_pose[0], target.rel_hand_pose[1], target.rel_hand_pose[2]))
        place_pose = grasp_pose.copy()
        if is_delta:
            place_pose[:3] += np.array(place_translation)
        else:
            place_pose[:3] = np.array(place_translation)
        self.scene.env.update_pose(target_name, place_pose) # update delta pose

        target.attached = True   
        self.scene.env.set_target(target_name)
        self.scene.update_planner()
        robot.resample_attached_object_collision_points(target)

        # placement fail
        if len(self.scene.traj.goal_set) == 0:
            print('please update place pose, there is no IK')
            config.cfg.disable_collision_set = [] 
            target.attached = False
            setattr(config.cfg, 'use_standoff', dummy_standoff)
            self.scene.env.update_pose(target_name, grasp_pose)
            robot.reset_hand_points()
            return None, False, grasp_pose, -1
        else: 
            info = self.scene.step()
            joint_trajectory_points = np.concatenate([self.scene.planner.history_trajectories[-1]], axis=0)
            config.cfg.disable_collision_set = [] 
            standoff_idx = info[-1]['standoff_idx']
            setattr(config.cfg, 'use_standoff', dummy_standoff) 
            end_hand_pose = robot.robot_kinematics.forward_kinematics_parallel(\
                wrap_value(joint_trajectory_points[[standoff_idx], ...]), base_link=config.cfg.base_link)[0][7]

            place_pose = compose_pose(pack_pose(end_hand_pose), target.rel_hand_pose)
            self.scene.env.update_pose(target_name, place_pose)

            if debug:
                if apply_standoff:
                    traj_data = self.scene.planner.history_trajectories[-1]
                    self.scene.fast_debug_vis(
                        traj=traj_data[:standoff_idx],
                        interact=1,
                        collision_pt=False,
                        nonstop=False,
                        write_video=True,
                    )
                    target.attached = False  # open the finger here
                    self.scene.fast_debug_vis(
                        traj=traj_data[standoff_idx:],
                        interact=1,
                        collision_pt=False,
                        nonstop=False,
                        traj_type=1,
                        write_video=True,
                    )
                else:
                    self.scene.fast_debug_vis(
                        interact=1,
                        collision_pt=False,
                        nonstop=False,
                        write_video=True,
                    )


            target.attached = False
            robot.reset_hand_points()
            print('collide: ', info[-1]['collide'])
            print('smooth: ', info[-1]['smooth'])
            return joint_trajectory_points, info[-1]['execute'], grasp_pose, standoff_idx


    def interface(self, start_conf, object_names, model_names, object_poses, flag_compute_grasp):
        self.scene.traj.start = start_conf
        self.scene.env.clear()
        start_time = time.time()
        self.populate_scene(object_names, model_names, object_poses, flag_compute_grasp)
        self.scene.reset()
        print('populate scene time: {:.3f}'.format(time.time() - start_time))


    def save_data(self, joint_listener):
        now = datetime.datetime.now()
        filename = "data/{:%m%dT%H%M%S}.mat".format(now)
        joint_position = joint_listener.joint_position

        model_names = []
        object_poses = []
        object_names = []
        for i in range(len(self.scene.env.objects)):
            obj = self.scene.env.objects[i]
            model_names.append(obj.model_name)
            object_poses.append(obj.pose)
            object_names.append(obj.name)

        meta = {'object_lists': model_names, 'object_poses': object_poses, 'object_names': object_names, 'joint_position': joint_position}
        scipy.io.savemat(filename, meta, do_compression=True)
        print('save data to {}'.format(filename))
