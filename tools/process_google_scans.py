#!/usr/bin/env python3
import sys
import os
import argparse
import glob
import cv2
import numpy as np
import copy
import shutil

if __name__ == "__main__":
    model_dir = '../data/google_scan_selected/'

    # list subdirs
    subdirs = sorted(os.listdir(model_dir))
    print(subdirs)

    for dirname in subdirs:

        texture_file = os.path.join(model_dir, dirname, 'materials', 'textures', 'texture.png')
        new_file = os.path.join(model_dir, dirname, 'meshes', 'texture.png')
        shutil.move(texture_file, new_file)
        print(new_file)
