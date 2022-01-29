from cmath import log
import os
import cv2
import json
import argparse
import helpers
import scipy.io
import numpy as np
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.ERROR)

parser = argparse.ArgumentParser(description='Create test set using real objects and OCID datasets')
parser.add_argument('--real_obj_data_loc', help='Location of synthetic dataset',
                    required=True, type=lambda x: helpers.is_valid_dir(parser, x))
parser.add_argument('--ocid_data_loc', help='Location of Google scenes dataset',
                    required=True, type=lambda x: helpers.is_valid_dir(parser, x))
parser.add_argument('--out_dir', help='Location of output directory',
                    required=True, type=lambda x: helpers.mkdirp(x))
args = parser.parse_args()


# 1. Process real objects dataset which will form the test (Support Set)
link_map, link_map_file = {}, 'real_obj_dir_class_name_mapper.json'
for dir in tqdm(os.listdir(args.real_obj_data_loc)):
    dir_path = os.path.join(args.real_obj_data_loc, dir)

    # Read name.txt containing class name
    class_name = helpers.get_class_name(dir_path)
    link_map[dir] = class_name

    # create required class directory to save cropped images
    out_support_dir = os.path.join(args.out_dir, class_name, "support")
    helpers.mkdirp(out_support_dir)

    # Get .jpg files from the folder (RGB)
    for jpg_img_path in helpers.get_files_ending_with(dir_path, 'jpg'):
        image_code = jpg_img_path.split("/")[-1].split("-")[0]
        
        # read the image mask
        mask_file = os.path.join(dir_path, f"{image_code}-label-binary.png")
        img_mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        orig_img = cv2.imread(jpg_img_path)
        cropped_img = helpers.crop(img_mask, orig_img)
        cropped_img_name = f"{class_name.replace(' ', '_')}-#{helpers.hashify(dir)}-{image_code}.png";
        cv2.imwrite(f"{os.path.join(out_support_dir, cropped_img_name)}", cropped_img)

# Save mapper
helpers.save_mapper(args.out_dir, link_map_file, link_map)

# 2. Process OCID dataset which will form the test (Query Set)
cropped_img_parent_mapper = {}
for dir in tqdm(os.listdir(args.ocid_data_loc)):
    dir_path = os.path.join(args.ocid_data_loc, dir)
    
    # Get .mat files from the folder
    for mat_path in helpers.get_files_ending_with(dir_path, 'mat'):
        meta_data = scipy.io.loadmat(mat_path)
        image_code = mat_path.split("/")[-1].split(".")[0].split("meta")[0]
        
        # read the segmentation mask
        color_file = os.path.join(dir_path, f"{image_code}color.jpg")
        img_mask, orig_img = meta_data['label'], cv2.imread(color_file)
        
        # Read name.txt containing class name
        class_name = helpers.get_class_name(dir_path)
        
        # create required class directory to save cropped images
        out_query_dir = os.path.join(args.out_dir, class_name, "query")
        helpers.mkdirp(out_query_dir)
                    
        # keep mask of required object
        object_pixel_value = meta_data['object_id'][0][0]
        mask = np.where(img_mask == object_pixel_value, 1, 0)

        try:
            cropped_img = helpers.crop(mask, orig_img)
            cropped_img_name = f"{dir}-{class_name.replace(' ', '_')}-#{helpers.hashify(image_code)}.png";
            cropped_img_parent_mapper[cropped_img_name]=f"{meta_data['seq_name'][0]}/{meta_data['image_name'][0]}"
            cv2.imwrite(os.path.join(out_query_dir, cropped_img_name), cropped_img)
        except:
            # raise Exception(f"Object id {object_pixel_value} doesn't exist in the current mask")
            logging.debug(f"Object id {object_pixel_value} doesn't exist in the current mask")
            continue

# Save mapper
helpers.save_mapper(args.out_dir, "ocid_cropped_img_parent_mapper.json", cropped_img_parent_mapper)