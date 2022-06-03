from cmath import log
import os
import cv2
import argparse
import helpers
import scipy.io
import numpy as np
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.ERROR)

parser = argparse.ArgumentParser(
    description='Create test set using real objects and OCID datasets.\
                 OCID contains segmentation masks from any custom segmentation network.')
parser.add_argument('--real_obj_data_loc', help='Location of real objects\' dataset',
                    required=True, type=lambda x: helpers.is_valid_dir(parser, x))
parser.add_argument('--ocid_data_loc', help='Location of OCID segmentation dataset',
                    required=True, type=lambda x: helpers.is_valid_dir(parser, x))
parser.add_argument('--out_dir', help='Location of output directory',
                    required=True, type=lambda x: helpers.mkdirp(x))
args = parser.parse_args()


# 1. Process real objects dataset which will form the test (Support Set)
helpers.create_test_support_set(args.real_obj_data_loc, args.out_dir)

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
        mask, orig_img = meta_data['mask_pred'], cv2.imread(color_file)
        
        # Read name.txt containing class name
        class_name = helpers.get_class_name(dir_path)
        
        # create required class directory to save cropped images
        out_query_dir = os.path.join(args.out_dir, class_name, "query")
        helpers.mkdirp(out_query_dir)

        # check for empty mask
        try:
            cropped_img = helpers.crop(mask, orig_img)
            cropped_img_name = f"{dir}-{class_name}-#{helpers.hashify(image_code)}.png";
            cropped_img_parent_mapper[cropped_img_name] = \
                f"{meta_data['seq_name'][0]}/{meta_data['image_name'][0]}"
            cv2.imwrite(os.path.join(out_query_dir, cropped_img_name), cropped_img)
        except:
            continue

# Save mapper
helpers.save_mapper(args.out_dir, 
                    "ocid_cropped_img_parent_mapper.json", 
                    cropped_img_parent_mapper)
