#!/bin/bash
	
set -x
set -e

export PYTHONUNBUFFERED="True"
# export CUDA_VISIBLE_DEVICES=$1

time ./tools/visualize_corr.py --gpu $1 \
  --imgdir $2 \
  --depth *-depth.png \
  --color *-color.jpg \
  --meta *-meta.mat

