# Building a Large Scale Dataset for Few-shot Object Instance Recognition

## Installation

- install the [ar_track_alvar](http://wiki.ros.org/ar_track_alvar) ros package

- compile the utility functions with:
```
./build.sh
```

## AR tag tracking using ROS

- start realsense
```
roslaunch realsense2_camera rs_aligned_depth.launch tf_prefix:=measured/camera
```

- start rviz
```
rosrun rviz rviz -d ./ros/artag.rviz
```

- start AR tag tracking
```
roslaunch ar_track_alvar realsense_indiv.launch
roslaunch easy_handeye publish_eye_on_hand.launch
```
