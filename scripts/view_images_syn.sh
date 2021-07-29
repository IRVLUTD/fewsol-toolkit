#!/bin/bash
	
set -x
set -e

export PYTHONUNBUFFERED="True"
# export CUDA_VISIBLE_DEVICES=$1

time ./tools/view.py --gpu $1 \
  --imgdir $2 \
  --depth depth_*.png \
  --color rgb_*.jpg \
  --meta meta_*.mat

