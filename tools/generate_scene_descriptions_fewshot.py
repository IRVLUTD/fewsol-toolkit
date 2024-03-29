""" This code is a python script that essentially replicates the code in simulating_tabletop_data.ipynb

    As a script, this can be called many times with a bash script. This helps when I need to kill
    the process due to the weird memory leak in PyBullet.

    To see how to call this script, run the following:
    $:~ python generate_data.py -h

    Keep end_scene - start_scene <= 50 in order to not get hung by memory leak

    NOTE: THIS SCRIPT ONLY GENERATES SCENE DESCRIPTIONS, so that it can be run in parallel w/out GUI.
          render_scene_descriptions.py will read the scene descriptions and generate RGBD + Segmentation images.
"""

import time
import os, sys
import argparse
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2

# my libraries
import simulation_util as sim_util

# pybullet
import pybullet as p
import pybullet_data

# suncg
import pybullet_suncg.fewshot_simulator as fewshot_sim

# for reloading libraries and debugging
from importlib import reload

VIEWS_PER_SCENE = 7

def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('train_or_test', help="currently, MUST be 'train'")
    parser.add_argument('scenario', help="MUST be in ['tabletop', 'cabinet', 'kitchen', 'fewshot']")
    parser.add_argument('start_scene', help="scene to start generating from")
    parser.add_argument('end_scene', help="last scene to generate (exclusive)")
    parser.add_argument('-cd', '--cabinet_dir', dest='cab_dir', help="""directory to sektion cabinet model.
                        Only required if scenario = cabinet""")
    parser.add_argument('-kd', '--kitchen_dir', dest='kitchen_dir', help="""directory to kitchen model.
                        Only required if scenario = kitchen""")
    parser.add_argument('-lf', '--logfile', dest='logfile', default='log.txt', help="logfile")
    parser.add_argument('--suncg_dir', required=True, help="SUNCG v1 base directory")
    parser.add_argument('--shapenet_dir', required=True, help="ShapeNetCore v2 base directory")
    parser.add_argument('--texture_dir', required=True, help="texture image directory")
    parser.add_argument('--coco_dir', required=True, help="coco image directory")
    parser.add_argument('--save_path', required=True, help="""Save path. This directory MUST have
                        [training_suncg_houses.json,
                        training_shapenet_tables.json, training_shapenet_objects.json]""")
    parser.add_argument("--remove", action="store_true", help="remove examples")
    args = parser.parse_args()

    if args.train_or_test != 'train':
        raise Exception('Only <train> supported currently...')
    if args.scenario not in ['tabletop', 'cabinet', 'kitchen', 'fewshot']:
        raise Exception("MUST select <scenario> from ['tabletop', 'cabinet', 'kitchen', 'fewshot']...")
    if not os.path.exists(args.save_path + 'training_suncg_houses.json'):
        raise Exception("save_path is missing 'training_suncg_houses.json'...")

    return args

def get_simulation_params(args):

    ##### Load SUNCG stuff #####

    # House lists
    training_houses_filename = args.save_path + 'training_suncg_houses.json'
    train_houses = json.load(open(training_houses_filename))

    # Room types I'm considering
    valid_room_types = set(['Living_Room', 'Kitchen', 'Room', 'Dining Room', 'Office'])

    # Room objects to filter out
    nyuv2_40_classes_filter_list = ['desk', 'chair', 'table', 'person', 'otherstructure', 'otherfurniture']
    coarse_grained_classes_filter_list = ['desk', 'chair', 'table', 'person', 'computer', 'bench_chair', 
                                          'ottoman', 'storage_bench', 'pet']

    ##### load ShapeNet #####
    # list all subdirs
    subdirs = os.listdir(args.shapenet_dir)
    train_tables = []
    ycb_objects = []
    shapenet_objects = []
    for d in subdirs:
        if d.startswith('Table_'):
            train_tables.append(d)
        elif d.startswith('0'):
            if d != '051_large_clamp':
                ycb_objects.append(d)
        elif 'Chair' not in d:
            shapenet_objects.append(d)

    # List of texture images
    textures = []
    filename = os.path.join(args.texture_dir, '*.jpg')
    files = glob.glob(filename)
    for i in range(len(files)):
        filename = files[i]
        textures.append(filename)
    textures.sort()

    coco_images = []
    for filename in os.listdir(args.coco_dir):
        coco_images.append(os.path.join(args.coco_dir, filename))
    coco_images.sort()

    simulation_params = {
        
        # scene stuff
        'num_objects_in_scene' : 5, # generate this many objects
        'min_objects_in_scene' : 3, # must still have this many objects
        'simulation_steps' : 1000,

        # House stuff
        'house_ids' : train_houses, 

        # room stuff
        'valid_room_types' : valid_room_types,
        'min_xlength' : 3.0, # Note: I believe this is in meters
        'min_ylength' : 3.0, 

        # table stuff
        'valid_tables' : train_tables, 
        'max_table_height' : 1.0, # measured in meters
        'min_table_height' : 0.75, 
        'max_table_size' : 1.2,
        'min_table_size' : 0.8,
        'table_init_factor' : 0.9, # this multiplicative factor limits how close you can initialize to wall

        # cabinet stuff
        'cab_init_factor' : 0.9, # this multiplicative factor limits how close you can initialize to wall

        # object stuff
        'shapenet_object_ids' : shapenet_objects, 
        'ycb_object_ids' : ycb_objects, 
        'max_xratio' : 1/4,
        'max_yratio' : 1/4,
        'max_zratio' : 1/3,
        'delta' : 0.4, # Above the bottom kitchen sektion cabinet is another cabinet. 
                       # If delta is too high, it might end up there...
        'ycb_obj_percentage' : 0.0,
        'shapenet_obj_percentage' : 0.7,
        'cuboid_percentage' : 0.15,
        'cylinder_percentage' : 0.15,

        # textures
        'textures' : textures,
        'coco_images' : coco_images,

        # stuff
        'max_initialization_tries' : 100,

        # Camera/Frustum parameters
        'img_width' : 640, 
        'img_height' : 480,
        'near' : 0.01,
        'far' : 100,
        'fov' : 45, # vertical field of view in angles

        # other camera stuff
        'max_camera_rotation' : np.pi / 15., # Max rotation in radians
        
        # Predicate stuff
        'theta_predicte_lr_fb_ab' : np.pi / 12., # 15 degrees
        'occ_IoU_threshold' : 0.5,

        # other stuff
        'nyuv2_40_classes_filter_list' : nyuv2_40_classes_filter_list,
        'coarse_grained_classes_filter_list' : coarse_grained_classes_filter_list,   
    }

    return simulation_params

def printout(string, filename):
    print(string, file=open(filename, 'a'))


def check_dir(direc):
    return all([f'segmentation_{i:05d}.png' in direc for i in range(VIEWS_PER_SCENE)]) and \
           all([f'depth_{i:05d}.png' in direc for i in range(VIEWS_PER_SCENE)]) and \
           all([f'rgb_{i:05d}.jpg' in direc for i in range(VIEWS_PER_SCENE)]) and \
           all([f'meta_{i:05d}.mat' in direc for i in range(VIEWS_PER_SCENE)])


def main():

    args = parse_arguments()
    simulation_params = get_simulation_params(args)

    save_dir = args.save_path + 'train/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # Calculate missing scenes
    temp = os.listdir(save_dir)
    temp1 = [int(x.split('_')[1]) for x in temp]
    missing_scenes = set(range(int(args.start_scene), int(args.end_scene))).difference(set(temp1)) 
    missing_scenes = sorted(list(missing_scenes))
    print('missing_scenes: ', missing_scenes )

    # Calculate missing scenes
    # temp = sorted(os.listdir(save_dir))
    # missing_scenes = []
    # for scene_num in range(int(args.start_scene), int(args.end_scene)):
    #    direct = f'scene_{scene_num:05d}'
    #    if direct in temp:
    #        directory_contents = os.listdir(save_dir + direct)
    #        if not check_dir(directory_contents):
    #            missing_scenes.append(scene_num)
    # print('missing_scenes: ', missing_scenes, len(missing_scenes))

    #### Actual Data Generation #####
    for scene_num in missing_scenes:
        
        # Sample scene
        try:
            sim = fewshot_sim.FewshotSimulator(
                    mode='gui', 
                    suncg_data_dir_base=args.suncg_dir, 
                    shapenet_data_dir_base=args.shapenet_dir, 
                    params=simulation_params, 
                    verbose=False
                   )
        except TimeoutError as e: # Scene took longer than 75 seconds to generate, or errored out
            printout(f"Scene {scene_num} took to long to genrate...", args.logfile)
            printout(str(e), args.logfile)
            sim.disconnect()
            continue
        except Exception as e:
            printout(f"Scene {scene_num} failed to genrate...", args.logfile)
            printout("Errored out. Not due to timer, but something else...", args.logfile)
            printout(str(e), args.logfile)
            sim.disconnect()
            continue
        except:
            printout(f"Scene {scene_num} failed to genrate...", args.logfile)
            printout("Not a Python Error... what is it?", args.logfile)
            continue

        scene_description = sim.generate_scenes(1)[0]

        # Make directory
        scene_save_dir = save_dir + f"scene_{scene_num:05d}/"
        print(scene_save_dir)
        if not os.path.exists(scene_save_dir):
            os.makedirs(scene_save_dir)
        
        # Scene Description
        scene_description_filename = scene_save_dir + 'scene_description.txt'
        with open(scene_description_filename, 'w') as save_file:
            json.dump(scene_description, save_file)    

        printout(f"Generated scene {scene_num}!", args.logfile)
        sim.disconnect()

if __name__ == '__main__':
    main()
