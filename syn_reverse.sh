#!/bin/bash

DST="/home/yuxiang/GitLab/few-shot-dataset"

cp $DST/README.md .
cp $DST/*.txt .
cp $DST/*.sh .

cp $DST/easy_handeye/*.yaml easy_handeye/
cp $DST/layers/*.py layers/
cp $DST/layers/*.cu layers/
cp $DST/layers/*.cpp layers/

cp $DST/planning/*.py planning/
cp $DST/ros/* ros/
cp $DST/scripts/* scripts/
cp $DST/tools/*.py tools/
cp $DST/tools/geo/*.py tools/geo
cp $DST/tools/pybullet_suncg/*.py tools/pybullet_suncg

