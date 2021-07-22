#!/usr/bin/env python3
import os
import time
import json
import numpy as np

if __name__ == '__main__':

    dir_name = '/capri/google_scenes/train'
    subdirs = sorted(os.listdir(dir_name))
    num = len(subdirs)

    # load mesh names
    filename = '../data/synthetic_objects_folders.txt'
    meshes = []
    with open(filename) as f:
        for line in f:
            meshes.append(line.strip())
    count = np.zeros((len(meshes), ), dtype=np.int32)

    for i in range(num):
        filename = os.path.join(dir_name, subdirs[i], 'scene_description.txt')
        print(filename)

        scene_description = json.load(open(filename))
        print(scene_description.keys())

        objects = scene_description['object_descriptions']
        for obj in objects:
            mesh_filename = obj['mesh_filename']
            names = mesh_filename.split('/')
            obj_name = names[-3]

            for j in range(len(meshes)):
                if obj_name == meshes[j]:
                    print(obj_name, j)
                    count[j] += 1
                    break

    print(count, sum(count))
