import os
import cv2
import glob
import json
import numpy as np
from PIL import Image
from tqdm import tqdm


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


def create_test_support_set(real_obj_data_loc, out_dir):
    '''
    Process real objects dataset which will form the test (Support Set)
    '''
    link_map, link_map_file = {}, 'real_obj_dir_class_name_mapper.json'
    for dir in tqdm(os.listdir(real_obj_data_loc)):
        dir_path = os.path.join(real_obj_data_loc, dir)

        # Read name.txt containing class name
        class_name = get_class_name(dir_path)
        link_map[dir] = class_name

        # create required class directory to save cropped images
        out_support_dir = os.path.join(out_dir, class_name, "support")
        mkdirp(out_support_dir)

        # Get .jpg files from the folder (RGB)
        for jpg_img_path in get_files_ending_with(dir_path, 'jpg'):
            image_code = jpg_img_path.split("/")[-1].split("-")[0]
            
            # read the image mask
            mask_file = os.path.join(dir_path, f"{image_code}-label-binary.png")
            img_mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            orig_img = cv2.imread(jpg_img_path)
            cropped_img = crop(img_mask, orig_img)
            cropped_img_name = f"{class_name}-#{hashify(dir)}-{image_code}.png";
            cv2.imwrite(f"{os.path.join(out_support_dir, cropped_img_name)}", cropped_img)
    
    # Save mapper
    save_mapper(out_dir, link_map_file, link_map)