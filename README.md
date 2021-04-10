# Building a Large Scale Dataset for Few-shot Object Instance Recognition

## Installation

- install the [ar_track_alvar](http://wiki.ros.org/ar_track_alvar) ros package

- install python packages
```Shell
pip install -r requirement.txt
```

- compile the utility functions with:
```
./build.sh
```

## Visualization of data

- View images
```
# $GPU_ID can be 0, 1; $FOLDER_PATH can be data/real_objects/0320T182600
./scripts/view_images.sh $GPU_ID $FOLDER_PATH
```

- View correspondences
```
./scripts/view_corr.sh $GPU_ID $FOLDER_PATH
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
