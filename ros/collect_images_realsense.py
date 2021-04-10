#!/usr/bin/env python

# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

"""collect images from Intel RealSense Camera"""

import rospy
import message_filters
import cv2
import argparse
import pprint
import time, os, sys
import os.path as osp
import numpy as np
import datetime
import scipy.io
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from ar_track_alvar_msgs.msg import AlvarMarkers
from transforms3d.quaternions import mat2quat, quat2mat

def parse_args():
    parser = argparse.ArgumentParser(
        description='Collect RGB-D images from realsense'
    )
    parser.add_argument('--no-marker', action='store_true', default=False)
    args = parser.parse_args()
    return args


class ImageListener:

    def __init__(self, with_marker=True):

        self.cv_bridge = CvBridge()
        self.count = 0
        self.with_marker = with_marker
        self.markers = None

        # output dir
        this_dir = osp.dirname(__file__)
        self.outdir = osp.join(this_dir, '..', 'data')
        if not osp.exists(self.outdir):
            os.mkdir(self.outdir)

        now = datetime.datetime.now()
        seq_name = "{:%m%dT%H%M%S}/".format(now)
        self.save_dir = osp.join(self.outdir, seq_name)
        if not osp.exists(self.save_dir):
            os.mkdir(self.save_dir)

        # initialize a node
        rospy.init_node("image_listener")

        # rgbd subscriber
        rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image, queue_size=2)
        depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, queue_size=2)
        self.target_frame = '/measured/camera_color_optical_frame'

        queue_size = 1
        slop_seconds = 0.025
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size, slop_seconds)
        ts.registerCallback(self.callback)

        # query camera intrinsics
        msg = rospy.wait_for_message('/camera/color/camera_info', CameraInfo)
        K = np.array(msg.K).reshape(3, 3)
        self.intrinsic_matrix = K
        print('Intrinsics matrix : ')
        print(self.intrinsic_matrix)

        # marker pose subscriber
        if self.with_marker:
            marker_sub = rospy.Subscriber('/ar_pose_marker', AlvarMarkers, self.callback_marker, queue_size=1)
        rospy.sleep(3.0)


    def callback_marker(self, markers):
        self.markers = markers


    def callback(self, rgb, depth):

        # check marker poses
        if self.with_marker:
            if self.markers is None:
                return
            markers = self.markers.markers
            if len(markers) < 3:
                return

        # convert depth
        if depth.encoding == '32FC1':
            depth_32 = self.cv_bridge.imgmsg_to_cv2(depth) * 1000
            depth_cv = np.array(depth_32, dtype=np.uint16)
        elif depth.encoding == '16UC1':
            depth_cv = self.cv_bridge.imgmsg_to_cv2(depth)
        else:
            rospy.logerr_throttle(
                1, 'Unsupported depth type. Expected 16UC1 or 32FC1, got {}'.format(
                    depth.encoding))
            return

        # save data
        im = self.cv_bridge.imgmsg_to_cv2(rgb, 'bgr8')
        filename = self.save_dir + '/%06d-color.jpg' % self.count
        if self.count % 10 == 0:
            cv2.imwrite(filename, im)
            print(filename)

            filename = self.save_dir + '/%06d-depth.png' % self.count
            cv2.imwrite(filename, depth_cv)
            print(filename)

            meta = {'intrinsic_matrix': self.intrinsic_matrix}
            # convert marker poses
            if self.with_marker:
                num = len(markers)
                for i in range(num):
                    marker = markers[i]
                    name = 'ar_marker_%02d' % (marker.id)
                    marker_pose = marker.pose.pose
                    pose = np.zeros((7, ), dtype=np.float32)
                    # translation
                    pose[0] = marker_pose.position.x
                    pose[1] = marker_pose.position.y
                    pose[2] = marker_pose.position.z
                    # quaternion
                    pose[3] = marker_pose.orientation.w
                    pose[4] = marker_pose.orientation.x
                    pose[5] = marker_pose.orientation.y
                    pose[6] = marker_pose.orientation.z
                    meta[name] = pose
                    print('%s: %f %f %f, %f %f %f %f' % (name, pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], pose[6]))

            filename = self.save_dir + '/%06d-meta.mat' % self.count
            print(filename)
            scipy.io.savemat(filename, meta, do_compression=True)  

        self.count += 1


if __name__ == '__main__':
    args = parse_args()

    # image listener
    listener = ImageListener(with_marker=not args.no_marker)
    try:  
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down"
