import rospy
import numpy as np
import message_filters
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo

class ImageListener:

    def __init__(self):

        self.cv_bridge = CvBridge()

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

        self.im = None
        self.depth = None


    def callback(self, rgb, depth):

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
        self.im = self.cv_bridge.imgmsg_to_cv2(rgb, 'bgr8')
        self.depth = depth_cv
