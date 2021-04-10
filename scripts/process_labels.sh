#!/bin/bash
	
set -x
set -e

export PYTHONUNBUFFERED="True"

time ./tools/process_clustering_labels.py --gpu $1 \
  --imgdir data/real_objects \
  --depth *-depth.png \
  --color *-color.jpg \
  --meta *-meta.mat

