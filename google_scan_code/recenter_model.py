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
    with open(model_obj_path, 'r') as f:
        model_obj = f.read().split('\n')
    model_obj = [x.split(' ') for x in model_obj]
    head = set([x[0] for x in model_obj])
    model_obj_dict = {}
    v = np.array([[float(x) for x in y[1:]] for y in model_obj if y[0] == 'v'])
    print(model_name, np.mean(v, axis=0))
    vertex_mean = np.mean(v, axis=0)
    new_model_obj = [t for t in model_obj]
    for i, x in enumerate(model_obj):
        if x[0] == 'v':
            for j in [1,2,3]:
                new_model_obj[i][j] = str(float(x[j])-vertex_mean[j-1])
    new_model_obj_str = '\n'.join([' '.join(x) for x in new_model_obj])
    new_model_obj_path = os.path.join(dataset_root_path, 'models', model_name, 'meshes', 'model_recenter.obj')
    with open(new_model_obj_path, 'w') as f:
        f.write(new_model_obj_str)




