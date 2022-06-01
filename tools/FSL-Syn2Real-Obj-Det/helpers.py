import os
import cv2
import glob
import json
import numpy as np
from PIL import Image


def is_valid_dir(parser, directory):
    '''
    Check existence of a directory
    '''
    if not os.path.isdir(directory):
        parser.error(f"The directory {directory} does not exist!")
    else:
        return directory


def mkdirp(directory):
    '''
    mkdir -p equivalent
    '''
    if not os.path.isdir(directory):
        os.makedirs(directory) 
    return directory


def get_files_ending_with(path, extension):
    '''
    Return the path of files with a particular extension
    residing in path {directory}
    '''
    return glob.glob(f"{path}/*.{extension}")


def crop(mask, orig_img):
    '''
    Crop masked segment of original image
    '''
    mask = cv2.findNonZero(mask)
    xmin = min([x[0][0] for x in mask])
    ymin = min([x[0][1] for x in mask])
    xmax = max([x[0][0] for x in mask])
    ymax = max([x[0][1] for x in mask])
    cropped_image = orig_img[ymin:ymax, xmin:xmax]
    return cropped_image


def imread_indexed(filename):
    '''
    Load segmentation image (with palette) given filename.
    Borrowed from: https://github.com/yuxng/few-shot-dataset/blob/main/tools/mask.py
    E.g. read segmentation mask from *.png files
    '''
    im = Image.open(filename)
    annotation = np.array(im)
    return annotation


def hashify(str):
    '''
    Generate hash of a string
    '''
    return abs(hash(str)) % (10 ** 8)


def get_class_name(dir_path):
    '''
    Read name.txt containing class name
    '''
    with open(os.path.join(dir_path, "name.txt"), 'r') as class_name_file:
         # only single line exists
        return class_name_file.read().strip()\
                .replace(' ', '_').replace("\'","").replace('&','')


def save_mapper(out_dir, filename, obj):
    '''
    Save mapper file
    '''
    with open(os.path.join(out_dir, filename), 'w') as outfile:
        json.dump(obj, outfile)