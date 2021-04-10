#!/usr/bin/env python3
import sys
import os
import argparse
import glob
import cv2
import numpy as np
import copy


if __name__ == "__main__":

    imgdir = '/capri/Fewshot_Dataset/real_objects'

    # list subdirs
    subdirs = os.listdir(imgdir)
    print(subdirs)

    for dirname in subdirs:

        # list images
        filename = os.path.join(imgdir, dirname, '*-color-masked.jpg')
        files = glob.glob(filename)
        for i in range(len(files)):
            filename = files[i]
            os.remove(filename)
            print(filename)

        # label
        filename = os.path.join(imgdir, dirname, '*-label-clustering.png')
        files = glob.glob(filename)
        for i in range(len(files)):
            filename = files[i]
            os.remove(filename)
            print(filename)
