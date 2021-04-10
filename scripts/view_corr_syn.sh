#!/bin/bash
	
set -x
set -e

export PYTHONUNBUFFERED="True"
# export CUDA_VISIBLE_DEVICES=$1

time ./tools/visualize_corr_syn.py --gpu $1 \
  --imgdir $2 \
  --depth depth_*.png \
  --color rgb_*.jpg \
  --seg segmentation_*.png \
  --meta meta_*.mat

