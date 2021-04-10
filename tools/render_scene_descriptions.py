""" As a script, this can be called many times with a bash script. This helps when I need to kill
    the process due to the weird memory leak in PyBullet.

    To see how to call this script, run the following:
    $:~ python render_scene_descriptions.py -h

    Keep end_scene - start_scene <= 50 in order to not get hung by memory leak

    NOTE: THIS SCRIPT REQUIRES SCENE DESCRIPTIONS are already computed
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
from generate_scene_descriptions import get_simulation_params, parse_arguments

# pybullet
import pybullet as p
import pybullet_data

# suncg
import pybullet_suncg.tabletop_simulator as tabletop_sim
import pybullet_suncg.cabinet_simulator as cabinet_sim
import pybullet_suncg.kitchen_simulator as kitchen_sim

# for reloading libraries and debugging
from importlib import reload

VIEWS_PER_SCENE = 7 # 1 background, 1 background+table/cabinet, 5 for background+table/cabinet+objects


def save_img_dict(img_dict, view_num, save_dir):
    # RGB
    rgb_filename = save_dir + f"rgb_{view_num:05d}.jpeg"
    cv2.imwrite(rgb_filename, cv2.cvtColor(img_dict['rgb'], cv2.COLOR_RGB2BGR))

    # Depth
    depth_filename = save_dir + f"depth_{view_num:05d}.png"
    cv2.imwrite(depth_filename, sim_util.saveable_depth_image(img_dict['depth']))

    # Segmentation
    seg_filename = save_dir + f"segmentation_{view_num:05d}.png"
    sim_util.imwrite_indexed(seg_filename, img_dict['seg'].astype(np.uint8))

def printout(string, filename):
    print(string, file=open(filename, 'a'))

def check_dir(direc):
    return all([f'segmentation_{i:05d}.png' in direc for i in range(VIEWS_PER_SCENE)]) and \
           all([f'depth_{i:05d}.png' in direc for i in range(VIEWS_PER_SCENE)]) and \
           all([f'rgb_{i:05d}.jpeg' in direc for i in range(VIEWS_PER_SCENE)])

def main():

    args = parse_arguments()
    simulation_params = get_simulation_params(args)

    save_dir = args.save_path + \
                ('training_set/' if args.train_or_test == 'train' else 'test_set/')

    # Calculate missing scenes
    temp = sorted(os.listdir(save_dir))
    missing_scenes = []
    for scene_num in range(int(args.start_scene), int(args.end_scene)):
        direct = f'scene_{scene_num:05d}'
        if direct in temp:
            directory_contents = os.listdir(save_dir + direct)
            if not check_dir(directory_contents):
                missing_scenes.append(scene_num)
    print('missing_scenes: ', missing_scenes, len(missing_scenes))

    #### Actual Data Generation #####
    for scene_num in missing_scenes:

        # Load scene
        if args.scenario == 'tabletop':
            sim = tabletop_sim.TabletopSimulator(
                    mode='gui', 
                    suncg_data_dir_base=args.suncg_dir, 
                    shapenet_data_dir_base=args.shapenet_dir, 
                    params=simulation_params, 
                    verbose=False
                   )
        elif args.scenario == 'cabinet':
            sim = cabinet_sim.CabinetSimulator(
                    mode='gui', 
                    suncg_data_dir_base=args.suncg_dir, 
                    shapenet_data_dir_base=args.shapenet_dir, 
                    sektion_cabinet_dir_base=args.cab_dir,
                    params=simulation_params, 
                    verbose=False
                  )
        elif args.scenario == 'kitchen':
            sim = kitchen_sim.KitchenSimulator(
                    mode='gui',
                    suncg_data_dir_base=args.suncg_dir, 
                    shapenet_data_dir_base=args.shapenet_dir, 
                    kitchen_dir_base=args.kitchen_dir,
                    params=simulation_params, 
                    verbose=False                        
                  )
        sim.reset()

        scene_save_dir = save_dir + f"scene_{scene_num:05d}/"
        print(scene_save_dir)
        scene_description = json.load(open(scene_save_dir + "scene_description.txt"))

        # Dictionary to save views
        scene_description['views'] = {}

        # Background-only view
        if args.scenario in ['tabletop', 'cabinet']:
            sim.load_house_room(scene_description)
        
            valid_background_view = False
            num_tries = 0
            while not valid_background_view:
                
                if num_tries > simulation_params['max_initialization_tries']:
                    break # this will force the entire scene to start over
                num_tries += 1
                
                # Sample the view
                img_dict = sim.sample_room_view()

                # Make sure it's valid. MUST have at least 2 SUNCG objects (e.g. walls/floor/fridge/couch)
                unique_labels = np.unique(img_dict['orig_seg_img'])
                valid_background_view = unique_labels.shape[0] >= 2

            if not valid_background_view:
                printout(f"""Scene {scene_num} failed to render. 
                    No valid background views...""", args.logfile)
                sim.disconnect()
                continue
            else:
                save_img_dict(img_dict, 0, scene_save_dir)
                scene_description['views']['background'] = img_dict['view_params']
                print('background view finished')

        # Background-table/cabinet view
        if args.scenario == 'tabletop':
            sim.load_table(scene_description)
        elif args.scenario == 'cabinet':
            sim.load_cabinet(scene_description)
        elif args.scenario == 'kitchen':
            sim.load_kitchen()
            sim.load_cabinet_pose(scene_description)

        valid_view = False
        num_tries = 0
        while not valid_view:
            
            if num_tries > simulation_params['max_initialization_tries']:
                break # this will force the entire scene to start over
            num_tries += 1
            print('num_tries: ', num_tries)
            
            # Sample the view
            if args.scenario == 'tabletop':
                img_dict = sim.sample_table_view(compute_predicates=False)
            elif args.scenario in ['cabinet', 'kitchen']:
                img_dict = sim.sample_cabinet_view(compute_predicates=False)
            
            # Make sure it's valid. If you can see at least _ pixels of tabletop/cabinet-top, then valid
            unique_labels = np.unique(img_dict['seg'])
            valid_view = 1 in unique_labels and np.count_nonzero(img_dict['seg'] == 1) > sim.NUM_PIXELS_VISIBLE
            
        if not valid_view:
            printout(f"""Scene {scene_num} failed to render. 
                No valid {args.scenario} views...""", args.logfile)
            sim.disconnect()
            continue
        else:
            save_img_dict(img_dict, 1, scene_save_dir)
            scene_description['views'][f'background+{args.scenario}'] = img_dict['view_params']
            print('table view finished')
            
        # Sample background-table/cabinet-object views. compute predicates, and save 
        sim.load_objects(scene_description)
        scene_description['views'][f'background+{args.scenario}+objects'] = []
        valid_views = False
        view_num = 2; num_tries = 0
        while not valid_views:
            
            if num_tries > simulation_params['max_initialization_tries']:
                break # this will force the entire scene to start over
            num_tries += 1
            print('num_tries: ', num_tries)

            # Sample the view
            if args.scenario == 'tabletop':
                img_dict = sim.sample_table_view(compute_predicates=False)
            elif args.scenario in ['cabinet', 'kitchen']:
                img_dict = sim.sample_cabinet_view(compute_predicates=False)

            # Make sure it's valid
            unique_labels = np.unique(img_dict['seg'])
            unique_object_labels = set(unique_labels).difference(set(range(sim.OBJ_LABEL_START)))
            valid = (0 in unique_labels and # background is in view
                     1 in unique_labels and # tabletop/cabinet-top is in view
                     len(unique_object_labels) >= 2 # at least 2 objects in view
                    )
            for label in unique_object_labels: # Make sure these labels are large enough
                if np.count_nonzero(img_dict['seg'] == label) < sim.NUM_PIXELS_VISIBLE:
                    valid = False
            if not valid:
                continue # sample another scene

            ### Save stuff ###
            save_img_dict(img_dict, view_num, scene_save_dir)
            scene_description['views'][f'background+{args.scenario}+objects'].append(img_dict['view_params'])
            print('camera view %d finished' % view_num)
            
            # increment    
            view_num += 1
            if view_num >= VIEWS_PER_SCENE:
                valid_views = True
            
        if not valid_views:
            printout(f"""Scene {scene_num} failed to render. 
                Tried to sample view too many times...""", args.logfile)
            sim.disconnect()
            continue    
        
        # Scene Description
        scene_description_filename = scene_save_dir + 'scene_description.txt'
        with open(scene_description_filename, 'w') as save_file:
            json.dump(scene_description, save_file)    

        # increment
        printout(f"Generated scene {scene_num}!", args.logfile)
        sim.disconnect()


if __name__ == '__main__':
    main()
