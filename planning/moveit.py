#!/usr/bin/env python
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from __future__ import print_function
import rospy
import numpy as np
import tf.transformations as tra
from lula_control.frame_commander import RobotConfigModulator, FrameConvergenceCriteria, TimeoutMonitor
from trac_ik_python.trac_ik import IK

try:
    import PyKDL as kdl
    from kdl_parser_py import urdf
except ImportError as e:
    rospy.logwarn("Could not load kdl parser. Try: `sudo apt install "
                  "ros-kinetic-kdl-*`")
    raise e


def lula_go_local(ee, T, high_precision=True, wait_for_target=False):
    """
    Convert between transform and the lula format. Specify goal.
    """
    orig = T[:3, 3]
    axis_x = T[:3, 0]
    axis_z = T[:3, 2]
    ee.go_local(
        orig=orig,
        axis_x=axis_x,
        axis_z=axis_z,
        use_target_weight_override=high_precision,
        use_default_config=False,
        wait_for_target=wait_for_target)


class MoveitBridge(object):

    def __init__(self, robot_interface, ee_link='right_gripper'):
        """
        Create the MoveIt bridge, with reasonable default values.
        """
        
        self.franka = robot_interface
        self.config_modulator = RobotConfigModulator()
        self.home_q = [-0.059293474684699775, -1.6124685985429639, -0.19709729059328113, -2.5317617220662476, 
                       -0.09526965726127999, 1.678176488975683, 0.5879584750097497]

        # initialize forward kinematics
        self.base_link = 'base_link'
        self.ee_link = ee_link

        success, kdl_tree = urdf.treeFromParam('/robot_description')
        if not success:
            raise RuntimeError(
                "Could not create kinematic tree from /robot_description.")

        self.kdl_chain = kdl_tree.getChain(self.base_link, ee_link)
        print("Number of joints in KDL chain:", self.kdl_chain.getNrOfJoints())
        self.kdl_fk = kdl.ChainFkSolverPos_recursive(self.kdl_chain)
        self.ik_solver = IK(self.base_link, ee_link, timeout=0.05, solve_type="Distance")


    def close_gripper(self, controllable_object=None, force=40., speed=0.1, wait=True):
        """
        Properly close the gripper.
        """

        self.franka.end_effector.gripper.close(
            controllable_object,
            wait=wait,
            speed=speed,
            actuate_gripper=True,
            force=force)

    def open_gripper(self, speed=0.1, wait=True):
        self.franka.end_effector.gripper.open(speed=speed, wait=wait)


    def retract(self, speed=None, wait=True):
        """
        Take the arm back to its home position.
        """
        if speed is not None:
            self.franka.set_speed(speed)
        self.franka.end_effector.go_local(orig=[], axis_x=[], axis_y=[], axis_z=[])
        self.go_local(q=self.home_q, wait=wait)


    def go_local(self, T=None, q=None, wait=False, controlled_frame=None,
                 time=3., err_threshold=0.003):
        """
        Move locally to transform or joint configuration q.
        """
        if T is None and q is not None:
            T = self.forward_kinematics(q)

        if controlled_frame is None:
            controlled_frame = self.franka.end_effector
        if q is not None:
            self.config_modulator.send_config(q)

        lula_go_local(self.franka.end_effector, T, wait_for_target=False)

        if wait:
            conv = FrameConvergenceCriteria(
                target_orig=T[:3, 3],
                target_axis_x=T[:3, 0],
                target_axis_z=T[:3, 2],
                required_orig_err=err_threshold,
                timeout_monitor=TimeoutMonitor(time)
            )
            rate = rospy.Rate(30)
            while not rospy.is_shutdown() and not self.franka.end_effector.is_preempted:
                if conv.update(
                        self.franka.end_effector.frame_status.orig,
                        self.franka.end_effector.frame_status.axis_x,
                        self.franka.end_effector.frame_status.axis_y,
                        self.franka.end_effector.frame_status.axis_z,
                        verbose=True):
                    break
                rate.sleep()


    def joint_list_to_kdl(self, q):
        if q is None:
            return None
        if isinstance(q, np.matrix) and q.shape[1] == 0:
            q = q.T.tolist()[0]
        q_kdl = kdl.JntArray(len(q))
        for i, q_i in enumerate(q):
            q_kdl[i] = q_i
        return q_kdl


    def forward_kinematics(self, q):
        ee_frame = kdl.Frame()
        kinematics_status = self.kdl_fk.JntToCart(self.joint_list_to_kdl(q),
                                                  ee_frame)
        if kinematics_status >= 0:
            p = ee_frame.p
            M = ee_frame.M
            return np.array([[M[0, 0], M[0, 1], M[0, 2], p.x()],
                             [M[1, 0], M[1, 1], M[1, 2], p.y()],
                             [M[2, 0], M[2, 1], M[2, 2], p.z()],
                             [0, 0, 0, 1]])
        else:
            return None


    def execute(self, traj, time=0.1, wait=False):
        """
        Execute a trajectory.
        """

        n = traj.shape[0]

        for i in range(n):
            positions = traj[i, :-2]

            # Command frame
            ee_frame = self.forward_kinematics(positions)
            if i == n - 1:
                self.go_local(T=ee_frame, q=traj[-1, :-2], wait=True)
            else:
                self.go_local(T=ee_frame, q=traj[-1, :-2], wait=wait)

            if i > 0:
                # first time_from_start is usually zero - so we can ignore it
                rospy.sleep(time)


    def ik(self, T, q0):
        rot = tra.quaternion_from_matrix(T)
        pos = T[:3, 3]
        result = self.ik_solver.get_ik(
            qinit=q0,
            x=pos[0],
            y=pos[1],
            z=pos[2],
            rx=rot[0],
            ry=rot[1],
            rz=rot[2],
            rw=rot[3],)
        return result
