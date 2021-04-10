#!/bin/bash

# This bash script runs the data generation code.
# It runs it in intervals due to some memory leak in PyBullet

SUNCG_DIR=/capri/SUNCG/
SHAPENET_DIR=/capri/ShapeNetCore.v2/
TEXTURE_DIR=/capri/textures/
COCO_DIR=/capri/coco/train2014/train2014/
SAVE_PATH=/capri/tabletop_data/

TABLETOP_TRAIN_START=0
TABLETOP_TRAIN_END=40000

python3.6 tools/generate_scene_descriptions.py train tabletop $TABLETOP_TRAIN_START $TABLETOP_TRAIN_END \
    --suncg_dir $SUNCG_DIR \
    --shapenet_dir $SHAPENET_DIR --texture_dir $TEXTURE_DIR --coco_dir $COCO_DIR --save_path $SAVE_PATH \

python3.6 tools/render_scene_descriptions.py train tabletop $TABLETOP_TRAIN_START $TABLETOP_TRAIN_END \
    --suncg_dir $SUNCG_DIR \
    --shapenet_dir $SHAPENET_DIR --texture_dir $TEXTURE_DIR --coco_dir $COCO_DIR --save_path $SAVE_PATH;
