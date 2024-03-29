#!/bin/bash

# This bash script runs the data generation code.
# It runs it in intervals due to some memory leak in PyBullet

SUNCG_DIR=/capri/SUNCG/
SHAPENET_DIR=/capri/ShapeNetGrasp/objects/
TEXTURE_DIR=/capri/ShapeNetGrasp/random_texture_2/
COCO_DIR=/capri/coco/train2014/train2014/
SAVE_PATH=/home/yuxiang/GitLab/few-shot-dataset/data/scenes/

TABLETOP_TRAIN_START=0
TABLETOP_TRAIN_END=20000
INTERVAL=50 # Used for image generation

######## IMAGE GENERATION (sequential) ########

python3.6 tools/render_scene_descriptions_fewshot.py train fewshot $TABLETOP_TRAIN_START $TABLETOP_TRAIN_END \
       --suncg_dir $SUNCG_DIR \
       --shapenet_dir $SHAPENET_DIR --texture_dir $TEXTURE_DIR --coco_dir $COCO_DIR --save_path $SAVE_PATH

# Generate tabletop RGBD/Seg images w/ GUI sequentially
# for ((i = $TABLETOP_TRAIN_START; i < $TABLETOP_TRAIN_END; i += $INTERVAL)); do

#    batch_start=$i;
#    batch_end=$(($i + $INTERVAL));
#    if [ $batch_end -gt $TABLETOP_TRAIN_END ]; then
#    batch_end=$TABLETOP_TRAIN_END;
#    fi

#    python3.6 tools/render_scene_descriptions_fewshot.py train fewshot $batch_start $batch_end \
#           --suncg_dir $SUNCG_DIR \
#           --shapenet_dir $SHAPENET_DIR --texture_dir $TEXTURE_DIR --coco_dir $COCO_DIR --save_path $SAVE_PATH;

#done
