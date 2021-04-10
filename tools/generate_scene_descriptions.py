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
import pybullet_suncg.tabletop_simulator as tabletop_sim
import pybullet_suncg.cabinet_simulator as cabinet_sim
import pybullet_suncg.kitchen_simulator as kitchen_sim

# for reloading libraries and debugging
from importlib import reload


def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('train_or_test', help="currently, MUST be 'train'")
    parser.add_argument('scenario', help="MUST be in ['tabletop', 'cabinet', 'kitchen']")
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
    if args.scenario not in ['tabletop', 'cabinet', 'kitchen']:
        raise Exception("MUST select <scenario> from ['tabletop', 'cabinet', 'kitchen']...")
    if not os.path.exists(args.save_path + 'training_suncg_houses.json'):
        raise Exception("save_path is missing 'training_suncg_houses.json'...")
    if not os.path.exists(args.save_path + 'training_shapenet_tables.json'):
        raise Exception("save_path is missing 'training_shapenet_tables.json'...")
    if not os.path.exists(args.save_path + 'training_shapenet_objects.json'):
        raise Exception("save_path is missing 'training_shapenet_objects.json'...")

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

    ##### Load ShapeNet stuff #####

    # Create a dictionary of name -> synset_id
    temp = json.load(open(args.shapenet_dir + 'taxonomy.json'))
    taxonomy_dict = {x['name'] : x['synsetId'] for x in temp}

    # weirdly, the synsets in the taxonomy file are not the same as what's in the ShapeNetCore.v2 directory. Filter this out
    synsets_in_dir = os.listdir(args.shapenet_dir)
    try:
        synsets_in_dir.remove('taxonomy.json')
    except ValueError:
        pass
    try:
        synsets_in_dir.remove('README.txt')
    except ValueError:
        pass
    try:
        synsets_in_dir.remove('paths.txt')
    except ValueError:
        pass

    taxonomy_dict = {k:v for (k,v) in taxonomy_dict.items() if v in synsets_in_dir}

    # List of train/test tables
    training_tables_filename = args.save_path + 'training_shapenet_tables.json'
    train_tables = json.load(open(training_tables_filename))

    # List of train/test object instances
    training_instances_filename = args.save_path + 'training_shapenet_objects.json'
    train_models = json.load(open(training_instances_filename))

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

    ##### Simulation Parameters #####
    house_ids = train_houses if args.train_or_test == 'train' else test_houses
    valid_tables = train_tables if args.train_or_test == 'train' else test_tables
    object_ids = train_models if args.train_or_test == 'train' else test_models

    simulation_params = {
        
        # scene stuff
        'num_objects_in_scene' : 25, # generate this many objects
        'min_objects_in_scene' : 5, # must still have this many objects
        'simulation_steps' : 1000,

        # House stuff
        'house_ids' : house_ids, 

        # room stuff
        'valid_room_types' : valid_room_types,
        'min_xlength' : 3.0, # Note: I believe this is in meters
        'min_ylength' : 3.0, 

        # table stuff
        'valid_tables' : valid_tables, 
        'max_table_height' : 1.0, # measured in meters
        'min_table_height' : 0.75, 
        'table_init_factor' : 0.9, # this multiplicative factor limits how close you can initialize to wall

        # cabinet stuff
        'cab_init_factor' : 0.9, # this multiplicative factor limits how close you can initialize to wall

        # object stuff
        'object_ids' : object_ids, 
        'max_xratio' : 1/4,
        'max_yratio' : 1/4,
        'max_zratio' : 1/3,
        'delta' : 0.4, # Above the bottom kitchen sektion cabinet is another cabinet. 
                       # If delta is too high, it might end up there...
        'shapenet_obj_percentage' : 0.7,
        'cuboid_percentage' : 0.15,
        'cylinder_percentage' : 0.15,

        # textures
        'textures' : textures,
        'coco_images' : coco_images,

        # stuff
        'max_initialization_tries' : 1000,

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
        'taxonomy_dict' : taxonomy_dict,
        'nyuv2_40_classes_filter_list' : nyuv2_40_classes_filter_list,
        'coarse_grained_classes_filter_list' : coarse_grained_classes_filter_list,                   
        
    }

    return simulation_params

def printout(string, filename):
    print(string, file=open(filename, 'a'))

def main():

    args = parse_arguments()
    simulation_params = get_simulation_params(args)

    save_dir = args.save_path + \
                ('training_set/' if args.train_or_test == 'train' else 'test_set/')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # Calculate missing scenes
    temp = os.listdir(save_dir)
    temp1 = [int(x.split('_')[1]) for x in temp]
    missing_scenes = set(range(int(args.start_scene), int(args.end_scene))).difference(set(temp1)) 
    missing_scenes = sorted(list(missing_scenes))
    print('missing_scenes: ', missing_scenes )

    #### Actual Data Generation #####
    for scene_num in missing_scenes:
        
        # Sample scene
        try:
            if args.scenario == 'tabletop':
                sim = tabletop_sim.TabletopSimulator(
                        mode='direct', 
                        suncg_data_dir_base=args.suncg_dir, 
                        shapenet_data_dir_base=args.shapenet_dir, 
                        params=simulation_params, 
                        verbose=False
                       )
            elif args.scenario == 'cabinet':
                sim = cabinet_sim.CabinetSimulator(
                        mode='direct', 
                        suncg_data_dir_base=args.suncg_dir, 
                        shapenet_data_dir_base=args.shapenet_dir, 
                        sektion_cabinet_dir_base=args.cab_dir,
                        params=simulation_params, 
                        verbose=False
                      )
            elif args.scenario == 'kitchen':
                sim = kitchen_sim.KitchenSimulator(
                        mode='direct',
                        suncg_data_dir_base=args.suncg_dir, 
                        shapenet_data_dir_base=args.shapenet_dir, 
                        kitchen_dir_base=args.kitchen_dir,
                        params=simulation_params, 
                        verbose=False                        
                      )
            scene_description = sim.generate_scenes(1)[0]
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

