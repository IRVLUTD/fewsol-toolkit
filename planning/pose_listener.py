import rospy
import rosnode
import tf
import tf2_ros
import cv2
import numpy as np
import sys
import os
import tf.transformations as tra
from transforms3d.quaternions import mat2quat, quat2mat
from ar_track_alvar_msgs.msg import AlvarMarkers


def make_pose(tf_pose):
    """
    Helper function to get a full matrix out of this pose
    """
    trans, rot = tf_pose
    pose = tra.quaternion_matrix(rot)
    pose[:3, 3] = trans
    return pose


class PoseListener:

    """
    Listens on a particular message topic.
    """

    def __init__(self, topic_name='/ar_pose_marker', queue_size=1):
        self.topic_name = topic_name
        self.markers = None
        self.sub = rospy.Subscriber(self.topic_name, AlvarMarkers, self.callback, queue_size=queue_size)
        self.listener = tf.TransformListener()
        self.base_frame = 'measured/base_link'


    def callback(self, msg):
        self.markers = msg


    def get_tf_pose(self, target_frame, base_frame=None, is_matrix=False):
        if base_frame is None:
            base_frame = self.base_frame
        try:
            tf_pose = self.listener.lookupTransform(base_frame, target_frame, rospy.Time(0))
            if is_matrix:
                pose = make_pose(tf_pose)
            else:
                trans, rot = tf_pose
                qt = np.zeros((4,), dtype=np.float32)
                qt[0] = rot[3]
                qt[1] = rot[0]
                qt[2] = rot[1]
                qt[3] = rot[2]
                pose = np.zeros((7, ), dtype=np.float32)
                pose[:3] = trans
                pose[3:] = qt
        except (tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException):
            pose = None
        return pose


    def get_marker_poses(self):

        marker_names = []
        marker_poses = []

        if self.markers is None:
            return marker_names, marker_poses

        markers = self.markers.markers
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

            marker_names.append(name)
            marker_poses.append(pose)
            print('%s: %f %f %f, %f %f %f %f' % (name, pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], pose[6]))
        return marker_names, marker_poses
