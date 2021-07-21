#!/bin/bash

# This bash script runs the data generation code.
# It runs it in intervals due to some memory leak in PyBullet

SUNCG_DIR=/capri/SUNCG/
SHAPENET_DIR=/capri/ShapeNetCore.v2/
GOOGLE_DIR=/capri/google_scan_selected/
TEXTURE_DIR=/capri/textures/
COCO_DIR=/capri/coco/train2014/train2014/
SAVE_PATH=/home/yuxiang/GitLab/few-shot-dataset/data/google_scenes/

TABLETOP_TRAIN_START=0
# TABLETOP_TRAIN_END=40000
TABLETOP_TRAIN_END=4

NUM_PROCESSES=10 # not used, but see PARALLEL_BATCH_SIZE
NUM_EXS_PER_CPU=1 # Number of scenes for each CPU process to handle
PARALLEL_BATCH_SIZE=$(($NUM_PROCESSES * $NUM_EXS_PER_CPU)) # 50 # make sure NUM_PROCESSES * NUM_EXS_PER_CPU = PARALLEL_BATCH_SIZE
INTERVAL=50 # Used for image generation

python3.6 tools/generate_scene_descriptions_google.py train fewshot $TABLETOP_TRAIN_START $TABLETOP_TRAIN_END \
    --suncg_dir $SUNCG_DIR --google_dir $GOOGLE_DIR \
    --shapenet_dir $SHAPENET_DIR --texture_dir $TEXTURE_DIR --coco_dir $COCO_DIR --save_path $SAVE_PATH \


######## SCENE DESCRIPTION GENERATION (parallel) ########

# Generate tabletop training scene descriptions in parallel
#for ((i = $TABLETOP_TRAIN_START; i < $TABLETOP_TRAIN_END; i += $PARALLEL_BATCH_SIZE)); do
#
#    batch_start=$i;
#    batch_end=$(($i + $PARALLEL_BATCH_SIZE));
#    if [ $batch_end -gt $TABLETOP_TRAIN_END ]; then
#      batch_end=$TABLETOP_TRAIN_END;
#    fi
#
#      for ((j = $batch_start; j < $batch_end; j += $NUM_EXS_PER_CPU)); do
#        cpu_start=$j;
#        cpu_end=$(($j + $NUM_EXS_PER_CPU));
#        if [ $cpu_end -gt $TABLETOP_TRAIN_END ]; then
#          cpu_end=$TABLETOP_TRAIN_END;
#        fi
#
#        python3.6 tools/generate_scene_descriptions_google.py train fewshot $cpu_start $cpu_end \
#               --suncg_dir $SUNCG_DIR \
#               --shapenet_dir $SHAPENET_DIR --texture_dir $TEXTURE_DIR --coco_dir $COCO_DIR --save_path $SAVE_PATH \
#               & # & turns it into a background process
#      done

    # Need to wait until the above is done
#    wait;
#done
