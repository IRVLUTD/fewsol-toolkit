#!/usr/bin/env python

import rospy
import tf
import tf2_ros
import _init_paths
import argparse
import signal
import pprint
import numpy as np
import os
import sys
import cv2
import datetime
import scipy.io
import tf.transformations as tra
from omg_planner import OMGPlanner
from pose_listener import PoseListener
from image_listener import ImageListener
from joint_listener import JointListener
from moveit import MoveitBridge
from transforms3d.quaternions import mat2quat, quat2mat
from lula_franka.franka import Franka


# joints for capture images
initial_joints = np.array(
[[0.23165384499058062, -1.43692836873957, 0.2087676826550801, -2.9796533541299595, -0.14206462124978236, 2.1601113103207332, 1.3046104287096685],
 [-0.0125524318020817, -1.6651628335684059, 0.08004616956553436, -2.586327477572257, 0.056178462145883255, 1.562259819257431, 0.8958728753361437],
 [-0.09706973141537426, -0.6791784113164534, 0.031992022070445575, -2.1595976422619834, 0.09324732653962271, 1.8413190482722357, 0.7808587242711736],
 [0.8886769078747735, -1.6388078182143915, -1.4399664068482867, -2.0016600849164385, -0.7038782951515943, 1.8698749156461065, -0.43667620701812443],
 [0.7670408177124824, -1.5357036861826696, -1.0912827538440102, -2.0131726698725694, -0.6921630678033399, 1.5991173117304207, -0.19940106165077856],
 [1.3474111886221858, -0.9861252284831221, -1.0257600915286966, -1.5518080450191833, -0.6621744801395681, 1.5131540506151107, 0.10139802777440748],
 [-1.0320341437573852, -1.8246343361823816, 1.3446373519728598, -1.7042517912065238, 0.8622671784044783, 1.5280847254124903, 2.051317493356832],
 [-0.9423370050443863, -1.7039398984254293, 1.040163923622985, -2.199651207736572, 0.9373933622230746, 1.6402310081105234, 1.5053702128547854],
 [-1.6356695315241119, -1.4953946400946194, 1.0445229034437786, -1.3499967386406346, 1.0005940594793823, 1.2524944467434096, 1.8225632318026572]])


def setup_planner(planner, moveit):

  # prepare object pose
  ycb_name = '003_cracker_box'
  object_names = []
  model_names = []
  object_poses = []
  object_names.append(ycb_name)
  model_names.append(ycb_name)
  flag_compute_grasp = [False] * len(model_names)
  pose = np.zeros((7, ), dtype=np.float32)
  object_poses.append(pose)

  # create planner interface
  start_conf = np.append(moveit.home_q, [0.0398, 0.0398])
  planner.interface(start_conf, object_names, model_names, object_poses, flag_compute_grasp)


# save data
def save_data(save_dir, count, im, depth, intrinsic_matrix, marker_names, marker_poses, joint_position):

    # color image
    filename = save_dir + '/%06d-color.jpg' % count
    cv2.imwrite(filename, im)
    print(filename)

    # depth
    filename = save_dir + '/%06d-depth.png' % count
    cv2.imwrite(filename, depth)
    print(filename)

    # meta data
    meta = {'intrinsic_matrix': intrinsic_matrix}
    meta['joint_position'] = joint_position
    num = len(marker_names)
    for i in range(num):
        name = marker_names[i]
        pose = marker_poses[i]
        meta[name] = pose
    filename = save_dir + '/%06d-meta.mat' % count
    print(filename)
    scipy.io.savemat(filename, meta, do_compression=True)  


if __name__ == '__main__':
    rospy.init_node('collect_images')
    debug = False

    # create robot
    franka = Franka(is_physical_robot=True)
    moveit = MoveitBridge(franka)
    moveit.open_gripper()
    moveit.retract()

    # create listeners
    image_listener = ImageListener()
    intrinsic_matrix = image_listener.intrinsic_matrix
    joint_listener = JointListener()
    pose_listener = PoseListener('/ar_pose_marker')
    rospy.sleep(1.0)

    # omg planner
    planner = OMGPlanner()
    setup_planner(planner, moveit)

    # output dir
    this_dir = os.path.dirname(__file__)
    outdir = os.path.join(this_dir, '..', 'data', 'real_objects')
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    # main loop
    num = initial_joints.shape[0]
    while True:

        # save directory
        now = datetime.datetime.now()
        seq_name = "{:%m%dT%H%M%S}/".format(now)
        save_dir = os.path.join(outdir, seq_name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        # for each position
        for i in range(num):
            target_conf = np.append(initial_joints[i, :], [0.0398, 0.0398])

            # planning
            joint_position = joint_listener.joint_position
            print('currect joints: ', joint_position)
            print('plan to config')
            print(target_conf)
            traj, flag_execute = planner.plan_to_conf(joint_position, target_conf, disable_list=[])
            if debug:
                planner.scene.fast_debug_vis(traj, collision_pt=False)

            # execute the plan
            moveit.execute(traj)
            rospy.sleep(0.5)

            # query the image
            im = image_listener.im
            depth = image_listener.depth

            # query marker poses
            marker_names, marker_poses = pose_listener.get_marker_poses()

            # query joints
            joint_position = joint_listener.joint_position

            # save data
            save_data(save_dir, i, im, depth, intrinsic_matrix, marker_names, marker_poses, joint_position)

        # time to change object
        raw_input('continue?')
