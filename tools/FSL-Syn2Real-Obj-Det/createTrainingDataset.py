import os
import cv2
import json
import argparse
import helpers
import scipy.io
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Create training set using Synthetic and Google scenes datasets')
parser.add_argument('--syn_data_loc', help='Location of synthetic dataset',
                    required=True, type=lambda x: helpers.is_valid_dir(parser, x))
parser.add_argument('--google_scenes_data_loc', help='Location of Google scenes dataset',
                    required=True, type=lambda x: helpers.is_valid_dir(parser, x))
parser.add_argument('--out_dir', help='Location of output directory',
                    required=True, type=lambda x: helpers.mkdirp(x))
args = parser.parse_args()


# 1. Process Synthetic dataset which will form the training (Support Set)
link_map, link_map_file = {}, 'syn_google_scenes_data_mapper.json'
for dir in tqdm(os.listdir(args.syn_data_loc)):
    dir_path = os.path.join(args.syn_data_loc, dir)

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


# 2. Process Google Scenes dataset which will form the training (Query Set)
for dir in tqdm(os.listdir(args.google_scenes_data_loc)):
    dir_path = os.path.join(args.google_scenes_data_loc, dir)
    
    # Get .mat files from the folder
    for mat_path in helpers.get_files_ending_with(dir_path, 'mat'):
        meta_data = scipy.io.loadmat(mat_path)
        image_code = mat_path.split("/")[-1].split(".")[0].split("_")[1]
        
        # read the segmentation mask
        mask_file = os.path.join(dir_path, f"segmentation_{image_code}.png")
        jpg_file = os.path.join(dir_path, f"rgb_{image_code}.jpg")
        img_mask, orig_img = helpers.imread_indexed(mask_file), cv2.imread(jpg_file)
        
        for object_name in meta_data["object_names"]:
            object_name = object_name.strip()
            class_name = link_map[object_name.strip()]

            # create required class directory to save cropped images
            out_query_dir = os.path.join(args.out_dir, class_name, "query")
            helpers.mkdirp(out_query_dir)
                        
            # keep mask of required object
            object_pixel_value = meta_data[object_name][0][0]
            mask = np.where(img_mask == object_pixel_value, 1, 0)
            
            try:
                cropped_img = helpers.crop(mask, orig_img)
                cropped_img_name = f"{dir}-{class_name.replace(' ', '_')}-#{helpers.hashify(dir)}-{image_code}-id-{object_pixel_value}.png";
                cv2.imwrite(os.path.join(out_query_dir, cropped_img_name), cropped_img)
            except:
                # raise Exception(f"Object id {object_pixel_value} doesn't exist in the current mask")
                continue