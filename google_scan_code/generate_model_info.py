from pyassimp import load
import os
import numpy as np
import json

dataset_root_path = '/data2/GoogleScanned'
model_list_path = os.path.join(dataset_root_path, 'models', 'model_list.txt')
with open(model_list_path, 'r') as f:
    model_list = f.read().split('\n')

model_info = {}
for model_name in model_list:
    model_obj_path = os.path.join(dataset_root_path, 'models', model_name, 'meshes', 'model.obj')
    assert(os.path.exists(model_obj_path))
    model = load(model_obj_path)
    assert(len(model.meshes)==1)
    v = model.meshes[0].vertices
    minv = v.min(axis=0)
    maxv = v.max(axis=0)
    sizev = maxv-minv
    diameter = np.sqrt(sizev.dot(sizev)).item()
    min_x,min_y,min_z = minv.tolist()
    size_x,size_y,size_z = sizev.tolist()
    model_info[model_name] = {'diameter': diameter,
                              'min_x': min_x,
                              'min_y': min_y,
                              'min_z': min_z,
                              'size_x': size_x,
                              'size_y': size_y,
                              'size_z': size_z}

model_info_path = os.path.join(dataset_root_path, 'models', 'model_info.json')
with open(model_info_path, 'w') as f:
    json.dump(model_info, f, indent=2)
