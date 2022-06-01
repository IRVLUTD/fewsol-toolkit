#!/usr/bin/env python3
import sys
import os
import argparse
import glob
import cv2
import numpy as np
import copy
import shutil
import matplotlib.pyplot as plt
import scipy.io
from pathlib import Path
from PIL import Image
from mask import imread_indexed, visualize_segmentation


def list_dataset(ocid_object_path):
    data_path = Path(ocid_object_path)
    seqs = sorted(list(Path(data_path).glob('**/*seq*')))
    return seqs


if __name__ == "__main__":
    dataset_dir = '../data/OCID/'

    # list subdirs
    seqs = list_dataset(dataset_dir)
    num_seqs = len(seqs)
    print('%d seqs' % num_seqs)

    # for each sequence
    count_objects = 0
    for i in range(num_seqs):
        seq_name = seqs[i]

        # list images
        rgb_folder = os.path.join(seq_name, 'rgb')
        image_names = sorted(os.listdir(rgb_folder))

        object_id = 1
        for j in range(len(image_names)):
            # read image
            filename = os.path.join(seq_name, 'rgb', image_names[j])
            object_id += 1
            count_objects += 1
    print('%d objects' % count_objects)
