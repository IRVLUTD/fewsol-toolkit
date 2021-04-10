import rospy
import numpy as np
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped

class JointListener:

    """
    Listens on a particular message topic.
    """

    def __init__(self, topic_name_joint='/robot/joint_states', topic_name_force='/franka_state_controller/F_ext', queue_size=100):
        self.topic_name_joint = topic_name_joint
        self.topic_name_force = topic_name_force
        self.robot_state = None
        self.joint_position = None
        self.robot_wrench = None
        self.robot_force = None
        self.robot_torque = None
        self._sub = rospy.Subscriber(self.topic_name_joint, JointState, self.robot_state_callback, queue_size=queue_size)
        self._sub_force = rospy.Subscriber(self.topic_name_force, WrenchStamped, self.robot_force_callback, queue_size=queue_size)

    def robot_state_callback(self, data):
        self.robot_state = data
        self.joint_position = np.array(data.position)

    def robot_force_callback(self, data):
        self.robot_wrench = data
        self.robot_force = np.array([data.wrench.force.x, data.wrench.force.y, data.wrench.force.z])
        self.robot_torque = np.array([data.wrench.torque.x, data.wrench.torque.y, data.wrench.torque.z])
