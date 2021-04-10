import numpy as np
import os, sys
import pybullet as p
import subprocess as sp
import time
import glob

from collections import namedtuple
from itertools import groupby
from pyquaternion import Quaternion
from transforms3d.quaternions import quat2mat

import pybullet_suncg.simulator as sim
from geo.transform import Transform
from simulation_util import projection_to_intrinsics, view_to_extrinsics, tf_quat
import pybullet_suncg.house as suncg_house

# my libraries
sys.path.insert(0, os.path.abspath('..')) # Add parent path to the sys.path. should only be called ONCE
import simulation_util as sim_util

# handle to a simulated rigid body
Body = namedtuple('Body', ['id', 'bid', 'vid', 'cid', 'static'])

# a body-body contact record
Contact = namedtuple(
    'Contact', ['flags', 'idA', 'idB', 'linkIndexA', 'linkIndexB',
                'positionOnAInWS', 'positionOnBInWS',
                'contactNormalOnBInWS', 'distance', 'normalForce']
)

# ray intersection record
Intersection = namedtuple('Intersection', ['id', 'linkIndex', 'ray_fraction', 'position', 'normal'])

def sample_up_quaternion():

    # sample an axis
    index = np.random.randint(0, 3)
    if index == 0:
        axis = [1, 0, 0]
    elif index == 1:
        axis = [0, 1, 0]
    else:
        axis = [0, 0, 1]

    # sample an angle
    angles = [0, 90, 180, 270]
    angle = np.random.choice(angles)
    qt = Quaternion(axis=axis, angle=angle * np.pi / 180.0)
    return np.roll(qt.elements, -1) # x, y, z, w

class FewshotSimulator(sim.Simulator):
    """ Simulate SUNCG rooms with random ShapeNet table
    """

    def get_collision_list(self, obj_id, all_obj_ids):
        """ My own simple collision checking using axis-aligned bounding boxes

            @param obj_id: ID of query object
            @param all_obj_ids: list of IDs of objects you want to check collision with
        """
        obj_coords = self.get_object_bbox_coordinates(self._obj_id_to_body[obj_id].id)
        objects_in_collision = []

        for other_obj_id in all_obj_ids:
            if other_obj_id == obj_id:
                continue
            other_obj_coords = self.get_object_bbox_coordinates(self._obj_id_to_body[other_obj_id].id)
            
            collision = (min(obj_coords['xmax'], other_obj_coords['xmax']) - max(obj_coords['xmin'], other_obj_coords['xmin']) > 0) and \
                        (min(obj_coords['ymax'], other_obj_coords['ymax']) - max(obj_coords['ymin'], other_obj_coords['ymin']) > 0) and \
                        (min(obj_coords['zmax'], other_obj_coords['zmax']) - max(obj_coords['zmin'], other_obj_coords['zmin']) > 0)
            if collision:
                objects_in_collision.append(other_obj_id)

        return objects_in_collision

    def load_table(self, scene_description):
        """ Takes a scene descrption dictionary (as exported by self.export_scene_to_dictionary())
            and loads the table only

            @param scene_description: a scene description dictionary (schema can be found in 
                                                          self.export_scene_to_dictionary())
        """        
        table_description = scene_description['table']
        if not table_description['mesh_filename'].startswith(self._shapenet_data_dir_base):
            table_description['mesh_filename'] = self._shapenet_data_dir_base + table_description['mesh_filename']
        table_transform = Transform(translation=np.array(table_description['position']), 
                                    rotation=Quaternion(w=table_description['orientation'][0],
                                                        x=table_description['orientation'][1],
                                                        y=table_description['orientation'][2],
                                                        z=table_description['orientation'][3]),
                                    scale=np.ones(3) * table_description['scale'])
        table_obj_id = 'ShapeNet_table_0'
        self.add_mesh(table_obj_id, 
                     table_description['mesh_filename'], 
                     table_transform, 
                     table_description['mesh_filename'],
                     table_description['texture_filename'])
        self.table_stuff = {'obj_id' : table_obj_id,
                           'table_mesh_filename' : table_description['mesh_filename'],
                           'table_scale_factor' : table_description['scale'],
                          }

    def load_scene(self, scene_description):
        """ Takes a scene description dictionary (as exported by self.export_scene_to_dictionary())
            and loads it

            @param scene_description: a scene description dictionary (schema can be found in 
                                                          self.export_scene_to_dictionary())
        """
        # copy dictionary so we don't overwrite original
        import copy # use copy.deepcopy for nested dictionaries
        scene_description = copy.deepcopy(scene_description)

        # Reset the scene
        self.reset()

        # Load the room
        self.load_house_room(scene_description)
                                  
        # Load the table
        self.load_table(scene_description)
                                  
        # Load the objects
        self.load_objects(scene_description)




    ##### CODE TO SIMULATE SCENES #####

    def generate_random_table(self):
        """ Randomly generate a shapenet table that is standing up in loaded SUNCG room
        """

        room_coords = self.get_object_bbox_coordinates(self.loaded_room.body[0].id) # Get bbox coordinates of room mesh

        is_up = False
        num_tried_tables = 0
        while not is_up:
            if num_tried_tables > self.params['max_initialization_tries']:
                self.table_stuff = None
                return
            num_tried_tables += 1

            ### Select random table ###
            while True:
                model_dir = np.random.choice(self.params['valid_tables'])
                model_dir = os.path.join(self._shapenet_data_dir_base, model_dir)
                table_mesh_filename = model_dir + '/model_normalized.processed.obj'
                if os.path.exists(table_mesh_filename):
                    break

            # Select a texture image
            texture_filename = np.random.choice(self.params['textures'])

            ### Create table object in pybullet ### 
            # Note: up direction is +Y axis in ShapeNet models
            table_obj_id = 'ShapeNet_table_0'
            table_transform = Transform(rotation=Quaternion(axis=[1, 0, 0], angle=-np.pi/2))
            print('table mesh:', table_mesh_filename)
            table_body = self.add_mesh(table_obj_id, table_mesh_filename, table_transform, table_mesh_filename)

            # Re-scale table to appropriate height and load it right above ground
            table_coords = self.get_object_bbox_coordinates(table_obj_id)
            table_size = max(table_coords['xsize'], table_coords['zsize'])
            max_scale_factor = self.params['max_table_size'] / table_size
            min_scale_factor = self.params['min_table_size'] / table_size
            table_scale_factor = np.random.uniform(min_scale_factor, max_scale_factor)
            table_transform.rescale(table_scale_factor)

            # Reload the resacled table right above the ground
            self.remove(table_obj_id)
            table_body = self.add_mesh(table_obj_id, table_mesh_filename, table_transform, table_mesh_filename)
            table_coords = self.get_object_bbox_coordinates(table_obj_id) # scaled coordinates

            # List of pybullet object ids to check collsion
            room_obj_ids = [x.body.id for x in self.loaded_room.nodes if x.body is not None]

            # Walls id
            walls_id = self.loaded_room.body[0].id
            if len(self.loaded_room.body) < 2:
                self.remove(table_obj_id)
                continue
            floor_coords = self.get_object_bbox_coordinates(self.loaded_room.body[1].id)

            # Sample xz_location until it's not in collision
            in_collision_w_objs = True; in_collision_w_walls = True
            num_tries = 0
            while in_collision_w_objs or in_collision_w_walls:

                xmin = (room_coords['xmin'] - table_coords['xmin']) / self.params['table_init_factor']
                xmax = (room_coords['xmax'] - table_coords['xmax']) * self.params['table_init_factor']
                random_start_xpos = np.random.uniform(xmin, xmax)
                ypos = floor_coords['ymax'] - table_coords['ymin'] + np.random.uniform(0, 0.1) # add a bit of space so no collision w/ carpets...
                zmin = (room_coords['zmin'] - table_coords['zmin']) / self.params['table_init_factor']
                zmax = (room_coords['zmax'] - table_coords['zmax']) * self.params['table_init_factor']
                random_start_zpos = np.random.uniform(zmin, zmax)
                if (xmax < xmin) or (zmax < zmin): # table is too large. pick a new table
                    break
                random_start_pos = np.array([random_start_xpos, ypos, random_start_zpos])
                self.set_state(table_obj_id, random_start_pos, table_transform.rotation)

                if num_tries > self.params['max_initialization_tries']:
                    break # this will force code to pick a new table
                num_tries += 1

                # Check if it's in collision with anything
                in_collision_w_walls = self.get_closest_point(table_obj_id, walls_id).distance < 0
                in_collision_w_objs = len(self.get_collision_list(table_obj_id, room_obj_ids)) > 0 # Simpler coarse collision checking with these objects

            if in_collision_w_objs or in_collision_w_walls: # if still in collision, then it's because we exhausted number of tries
                self.remove(table_obj_id)
                continue

            # Let the table fall for about 1 second
            self.simulate(300)

            # Check if it fell down
            up_orientation = table_transform.rotation.elements # w,x,y,z
            is_up = np.allclose(self.get_state(table_obj_id)[1].elements, up_orientation, atol=1e-1)

            # Remove the table if it fell down or is in collision
            if not is_up:
                self.remove(table_obj_id)
                continue # yes, this is redundant
         
        # point debug camera at table
        if self._mode == 'gui':
            table_coords = self.get_object_bbox_coordinates(table_obj_id)
            table_pos = list(self.get_state(table_obj_id)[0])
            table_pos[1] = table_coords['ymax']
            p.resetDebugVisualizerCamera(cameraDistance=1.5,
                                         cameraYaw=45.0,
                                         cameraPitch=-30.0,
                                         cameraTargetPosition=table_pos,
                                         physicsClientId=self._pid)

        self.table_stuff = {'obj_id' : table_obj_id,
                            'table_mesh_filename' : table_mesh_filename,
                            'table_scale_factor' : table_scale_factor,
                            'texture_filename' : texture_filename,
                           }

         

    def generate_random_shapenet_models(self):
        """ Sample random ShapeNet models

            NOTE: This is to be called AFTER self.generate_random_table() has been called
        """

        ##### Sample random ShapeNet models #####
        obj_mesh_filenames = []
        obj_ids = []
        obj_scale_factors = []
        obj_texture_filenames = []

        ycb_obj_p = self.params['ycb_obj_percentage']
        sn_obj_p = self.params['shapenet_obj_percentage']
        cub_p = self.params['cuboid_percentage']
        cyl_p = self.params['cylinder_percentage']

        # Get max/min coordinates of table
        table_coords = self.get_object_bbox_coordinates(self.table_stuff['obj_id'])
        
        i = 0
        num_objects_for_scene = self.params['num_objects_in_scene']
        if self._verbose:
            print(f"Number of objects chosen for scene: {num_objects_for_scene}")
        while len(obj_ids) < num_objects_for_scene:
            obj_id = f'ShapeNet_obj_{i}'

            ### Sample the object ###
            obj_transform = Transform(rotation=Quaternion(axis=[1, 0, 0], angle=-np.pi/2))
            sample = np.random.rand()
            if sample < ycb_obj_p:
                obj_type = 'ycb'
                while True:
                    model_dir = np.random.choice(self.params['ycb_object_ids'])
                    model_dir = os.path.join(self._shapenet_data_dir_base, model_dir)
                    obj_mesh_filename = os.path.join(model_dir, 'model_normalized.obj')
                    if os.path.exists(obj_mesh_filename):
                        if obj_mesh_filename not in obj_mesh_filenames:
                            obj_mesh_filenames.append(os.path.join(model_dir, 'model_normalized.obj'))
                            break

                ### Create an object in pybullet ###
                print('ycb object:', obj_mesh_filename)
                obj_body = self.add_mesh(obj_id, obj_mesh_filename, obj_transform, obj_mesh_filename)

            elif sample < ycb_obj_p + sn_obj_p:
                obj_type = 'sn'
                while True:
                    model_dir = np.random.choice(self.params['shapenet_object_ids'])
                    model_dir = os.path.join(self._shapenet_data_dir_base, model_dir)
                    obj_mesh_filename = os.path.join(model_dir, 'model_normalized.processed.obj')
                    if os.path.exists(obj_mesh_filename):
                        obj_mesh_filenames.append(os.path.join(model_dir, 'model_normalized.processed.obj'))
                        break

                ### Create an object in pybullet ###
                obj_body = self.add_mesh(obj_id, obj_mesh_filename, obj_transform, obj_mesh_filename)

            elif sample < ycb_obj_p + sn_obj_p + cub_p: # cuboid
                obj_type = 'cub'
                obj_transform._r = Quaternion(x=-1, y=0, z=0, w=1) # y-axis up
                obj_body = self.add_cuboid(obj_id, obj_transform)
                temp = p.getVisualShapeData(self._obj_id_to_body[obj_id].bid)[0]
                half_extents = np.array(temp[3])/2
                rgba_color = temp[7]
                obj_mesh_filenames.append(str({'obj_type' : obj_type,
                                               'half_extents' : tuple(half_extents), 
                                               'rgba_color' : rgba_color}))
                # dictionary stored as a string. yeah, not the greatest idea

            elif sample < ycb_obj_p + sn_obj_p + cub_p + cyl_p: # cylinder
                obj_type = 'cyl'
                obj_transform._r = Quaternion(x=-1, y=0, z=0, w=1) # y-axis up
                obj_body = self.add_cylinder(obj_id, obj_transform)
                temp = p.getVisualShapeData(self._obj_id_to_body[obj_id].bid)[0]
                height, radius = temp[3][:2]
                rgba_color = temp[7]
                obj_mesh_filenames.append(str({'obj_type' : obj_type,
                                               'height' : height, 
                                               'radius' : radius, 
                                               'rgba_color' : rgba_color}))
                # dictionary stored as a string. yeah, not the greatest idea

            else:
                raise Exception("sum of [shapenet_obj_percentage, cuboid_percentage, cylinder_percentage] MUST = 1")

            # Sample a texture image
            if np.random.rand() < 0.8 and obj_type != 'ycb':
                texture_filename = np.random.choice(self.params['coco_images'])
            else:
                texture_filename = ''
            obj_texture_filenames.append(texture_filename)

            ### Sample scale of object ###
            canonical_obj_coords = self.get_object_bbox_coordinates(obj_id)

            # Compute scale factor
            if obj_type != 'ycb':
                xscale_factor = 1; yscale_factor = 1; zscale_factor = 1
                if canonical_obj_coords['xsize'] / table_coords['xsize'] > self.params['max_xratio']:
                    xscale_factor = self.params['max_xratio'] * table_coords['xsize'] / (canonical_obj_coords['xsize'])
                if canonical_obj_coords['ysize'] / table_coords['ysize'] > self.params['max_yratio']:
                    yscale_factor = self.params['max_yratio'] * table_coords['ysize'] / (canonical_obj_coords['ysize'])
                if canonical_obj_coords['zsize'] / table_coords['zsize'] > self.params['max_zratio']:
                    zscale_factor = self.params['max_zratio'] * table_coords['zsize'] / (canonical_obj_coords['zsize'])
                max_scale_factor = min(xscale_factor, yscale_factor, zscale_factor)
                obj_scale_factor = np.random.uniform(max_scale_factor * 0.75, max_scale_factor)
                obj_scale_factors.append(obj_scale_factor)
                obj_transform.rescale(obj_scale_factor)
            else:
                obj_scale_factors.append(1.0)

            ##### Sample random location/orientation for object #####

            # Make sure the object is not in collision with any other object
            in_collision = True
            num_tries = 0
            while in_collision:

                # Get all objects that are straight (these are the ones that could be supporting objects)
                straight_obj_ids = [x for x in obj_ids if np.allclose(self.get_state(x)[1].elements, np.array([1,0,0,0]))]

                # Sample a random starting orientation
                sample = np.random.rand()
                if sample < 0.4: # Simulate straight up

                    q = sample_up_quaternion()
                    if obj_type in ['cub', 'cyl']:
                        q = np.array([-1,0,0,1]) # y-axis up
                    extra_y = 0.
                    x_range_min = table_coords['xmin'] - canonical_obj_coords['xmin'] + table_coords['xsize'] * 0.2
                    x_range_max = table_coords['xmax'] - canonical_obj_coords['xmax'] - table_coords['xsize'] * 0.2
                    z_range_min = table_coords['zmin'] - canonical_obj_coords['zmin'] + table_coords['zsize'] * 0.2
                    z_range_max = table_coords['zmax'] - canonical_obj_coords['zmax'] - table_coords['zsize'] * 0.2

                elif sample < 0.8 and len(straight_obj_ids) >= 1: # put it one another object

                    q = sample_up_quaternion()
                    if obj_type in ['cub', 'cyl']:
                        q = np.array([-1,0,0,1]) # y-axis up
                    support_obj_id = np.random.choice(straight_obj_ids)
                    support_obj_coords = self.get_object_bbox_coordinates(support_obj_id)
                    extra_y = support_obj_coords['ysize'] + 1e-3 # 1mm for some extra wiggle room

                    # Select x,z coordinates to place it randomly on top
                    x_range_min = support_obj_coords['xmin'] - canonical_obj_coords['xmin']
                    x_range_max = support_obj_coords['xmax'] - canonical_obj_coords['xmax']
                    z_range_min = support_obj_coords['zmin'] - canonical_obj_coords['zmin']
                    z_range_max = support_obj_coords['zmax'] - canonical_obj_coords['zmax']

                    # If supporting object is too small, just place it in the middle
                    if x_range_min > x_range_max:
                        x_range_min = (support_obj_coords['xmin'] + support_obj_coords['xmax']) / 2.
                        x_range_max = x_range_min
                    if z_range_min > z_range_max:
                        z_range_min = (support_obj_coords['zmin'] + support_obj_coords['zmax']) / 2.
                        z_range_max = z_range_min

                else: # Simulate a random orientation

                    q = np.random.uniform(0, 2*np.pi, 3) # Euler angles
                    q = p.getQuaternionFromEuler(q)
                    extra_y = np.random.uniform(0, self.params['delta'])
                    x_range_min = table_coords['xmin'] - canonical_obj_coords['xmin'] + table_coords['xsize'] * 0.2
                    x_range_max = table_coords['xmax'] - canonical_obj_coords['xmax'] - table_coords['xsize'] * 0.2
                    z_range_min = table_coords['zmin'] - canonical_obj_coords['zmin'] + table_coords['zsize'] * 0.2
                    z_range_max = table_coords['zmax'] - canonical_obj_coords['zmax'] - table_coords['zsize'] * 0.2

                # Load this in and get axis-aligned bounding box
                self.remove(obj_id)
                obj_transform._r = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3]) # HACK. Not good to access "private" attribute _r, but whatevs
                if obj_type == 'sn' or obj_type == 'ycb':
                    obj_body = self.add_mesh(obj_id, obj_mesh_filename, obj_transform, obj_mesh_filename)
                elif obj_type == 'cub':
                    obj_body = self.add_cuboid(obj_id, obj_transform, half_extents=half_extents, rgba_color=rgba_color)
                elif obj_type == 'cyl':
                    obj_dy = self.add_cylinder(obj_id, obj_transform, height=height, radius=radius, rgba_color=rgba_color)
                obj_coords = self.get_object_bbox_coordinates(obj_id) # scaled coordinates

                # Sample a random starting location
                random_start_xpos = np.random.uniform(x_range_min, x_range_max)
                ypos = table_coords['ymax'] - obj_coords['ymin'] + extra_y
                random_start_zpos = np.random.uniform(z_range_min, z_range_max)
                random_start_pos = np.array([random_start_xpos, ypos, random_start_zpos])

                # Set position/orientation
                self.set_state(obj_id, random_start_pos, obj_transform._r) # HACK. Not good to access "private" attribute _r, but whatevs

                if num_tries > self.params['max_initialization_tries']:
                    break
                num_tries += 1

                # Check for collision
                in_collision = len(self.get_collision_list(obj_id, obj_ids)) > 0

            if in_collision: # if still in collision, then it's because we exhausted number of tries
                self.remove(obj_id)
                obj_mesh_filenames.pop(-1) # remove this since we aren't using this object
                obj_scale_factors.pop(-1) # remove this since we aren't using this object
                obj_texture_filenames.pop(-1)
                continue

            # If we get here, then object has successfully by initialized
            obj_ids.append(obj_id)
            i += 1


        self.shapenet_obj_stuff = {'obj_ids' : obj_ids,
                                   'obj_mesh_filenames': obj_mesh_filenames,
                                   'obj_scale_factors' : obj_scale_factors,
                                   'obj_texture_filenames': obj_texture_filenames,
                                  }

    def remove_fallen_objects(self):
        """ Remove any objects that have fallen (i.e. are lower than the table)
        """
        table_coords = self.get_object_bbox_coordinates(self.table_stuff['obj_id'])
        num_objects = len(self.shapenet_obj_stuff['obj_ids'])
        
        fallen_obj_ids = []
        for obj_id in self.shapenet_obj_stuff['obj_ids']:
            obj_coords = self.get_object_bbox_coordinates(obj_id)
            obj_ypos = (obj_coords['ymin'] + obj_coords['ymax']) / 2.
            if obj_ypos < table_coords['ymax']:
                fallen_obj_ids.append(obj_id)

        # This code actually removes the object from the scene
        for obj_id in fallen_obj_ids:
            self.remove(obj_id)

        # Update self.shapenet_obj_stuff dictionary
        valid_indices = []
        for i, obj_id in enumerate(self.shapenet_obj_stuff['obj_ids']):
            if obj_id not in fallen_obj_ids:
                valid_indices.append(i)
        self.shapenet_obj_stuff['obj_ids'] = [self.shapenet_obj_stuff['obj_ids'][i] 
                                              for i in range(num_objects) if i in valid_indices]
        self.shapenet_obj_stuff['obj_mesh_filenames'] = [self.shapenet_obj_stuff['obj_mesh_filenames'][i] 
                                                         for i in range(num_objects) if i in valid_indices]
        self.shapenet_obj_stuff['obj_scale_factors'] = [self.shapenet_obj_stuff['obj_scale_factors'][i] 
                                                        for i in range(num_objects) if i in valid_indices]
        self.shapenet_obj_stuff['obj_texture_filenames'] = [self.shapenet_obj_stuff['obj_texture_filenames'][i] 
                                                         for i in range(num_objects) if i in valid_indices]

    def export_scene_to_dictionary(self):
        """ Exports the PyBullet scene to a dictionary
        """

        # Initialize empty scene description
        scene_description = {}

        # House/Room description
        room_description = {'house_id' : self.loaded_room.house_id,
                            'room_id' : self.loaded_room.id}
        scene_description['room'] = room_description

        # Table description
        temp = self.get_state(self.table_stuff['obj_id'])
        table_description = {'mesh_filename' : self.table_stuff['table_mesh_filename'].replace(self._shapenet_data_dir_base, ''),
                             'texture_filename' : self.table_stuff['texture_filename'],
                             'position' : list(temp[0]),
                             'orientation' : list(temp[1].elements), # w,x,y,z quaternion
                             'scale' : self.table_stuff['table_scale_factor']}
        scene_description['table'] = table_description

        # Get descriptions of objects on table
        object_descriptions = []
        for i, obj_id in enumerate(self.shapenet_obj_stuff['obj_ids']):

            mesh_filename = self.shapenet_obj_stuff['obj_mesh_filenames'][i]
            texture_filename = self.shapenet_obj_stuff['obj_texture_filenames'][i]
            temp = self.get_state(obj_id)
            pos = list(temp[0])
            orientation = list(temp[1].elements) # w,x,y,z quaternion
            scale = self.shapenet_obj_stuff['obj_scale_factors'][i]

            obj_coords = self.get_object_bbox_coordinates(obj_id)
            bbox_center = list(sim_util.get_aligned_bbox3D_center(obj_coords))

            description = {'obj_id' : obj_id,
                           'mesh_filename' : mesh_filename.replace(self._shapenet_data_dir_base, ''),
                           'texture_filename' : texture_filename,
                           'position' : pos,
                           'orientation' : orientation,
                           'scale' : scale,
                           'axis_aligned_bbox3D_center' : bbox_center,
                          }
            object_descriptions.append(description)
            
        scene_description['object_descriptions'] = object_descriptions

        return scene_description

    @sim_util.timeout(75) # Limit this function to max of _ seconds
    def generate_scenes(self, num_scenes):

        scenes = []
        times = []
        while len(scenes) < num_scenes:

            start_time = time.time()

            self.reset()
            self.add_random_house_room(no_walls=False, no_ceil=False, no_floor=False, 
                                       use_separate_walls=False, only_architecture=False, 
                                       static=True)

            self.generate_random_table()

            if self.table_stuff is None: # This means we tried many tables and it didn't work
                continue # start over
            self.generate_random_shapenet_models()
            self.simulate(self.params['simulation_steps'])
            self.remove_fallen_objects()
            
            # Check some bad situations
            if len(self.shapenet_obj_stuff['obj_ids']) < self.params['min_objects_in_scene']: # Too many objects fell off
                continue # start over
            if self.get_state(self.table_stuff['obj_id'])[0][1] < -0.1: # table fell way way way down
                continue # start over

            ### Export scene to dictionary ###
            scene_description = self.export_scene_to_dictionary()
            scenes.append(scene_description)

            # End of while loop. timing stuff
            end_time = time.time()
            times.append(round(end_time - start_time, 3))

        if self._verbose:
            print("Time taken to generate scene: {0} seconds".format(sum(times)))
            print("Average time taken to generate scene: {0} seconds".format(np.mean(times)))

        return scenes




    ##### CODE TO RENDER SCENES #####

    def sample_table_view(self, compute_predicates=False):
        """ Sample a view near table
        """

        # Sample position on xz bbox of table
        table_coords = self.get_object_bbox_coordinates(self.table_stuff['obj_id'])

        # First, select a side
        xz_bbox_side_probs = np.array([table_coords['xsize'], # x side 1
                                       table_coords['zsize'], # z side 1
                                       table_coords['xsize'], # x side 2
                                       table_coords['zsize']] # z side 2
                                     )
        xz_bbox_side_probs = xz_bbox_side_probs / np.sum(xz_bbox_side_probs)
        side = np.random.choice(range(4), p=xz_bbox_side_probs)
        if side == 0: # x side 1
            p1 = np.array([table_coords['xmin'], table_coords['zmin']])
            p2 = np.array([table_coords['xmax'], table_coords['zmin']])
            side_length = np.linalg.norm(p2 - p1)
            other_side_length = np.linalg.norm(p2 - np.array([table_coords['xmax'], table_coords['zmax']]))
        elif side == 1: # z side 1
            p1 = np.array([table_coords['xmax'], table_coords['zmin']])
            p2 = np.array([table_coords['xmax'], table_coords['zmax']])
            side_length = np.linalg.norm(p2 - p1)
            other_side_length = np.linalg.norm(p2 - np.array([table_coords['xmin'], table_coords['zmax']]))
        elif side == 2: # x side 2
            p1 = np.array([table_coords['xmax'], table_coords['zmax']])
            p2 = np.array([table_coords['xmin'], table_coords['zmax']])
            side_length = np.linalg.norm(p2 - p1)
            other_side_length = np.linalg.norm(p2 - np.array([table_coords['xmin'], table_coords['zmin']]))
        elif side == 3: # z side 2
            p1 = np.array([table_coords['xmin'], table_coords['zmax']])
            p2 = np.array([table_coords['xmin'], table_coords['zmin']])
            side_length = np.linalg.norm(p2 - p1)
            other_side_length = np.linalg.norm(p2 - np.array([table_coords['xmax'], table_coords['zmin']]))

        # Select point on that side uniformly
        point = p1 + (p2 - p1) * np.random.uniform(0.4, 0.6)

        # Sample xz distance from that point
        dist_from_table = np.random.uniform(0.0, 0.15)
        theta = np.radians(-90)
        rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                               [np.sin(theta), np.cos(theta)]])
        away_from_table_direction = rot_matrix.dot ( (p2 - p1) / np.linalg.norm(p2 - p1) )
        camera_x, camera_z = point + dist_from_table * away_from_table_direction

        # Sample y distance
        # height_from_table = np.random.uniform(.5*table_coords['ysize'], 1.0*table_coords['ysize'])
        height_from_table = np.random.uniform(.5, 1.0) # anywhere from .5m to 1.2m above table
        camera_y = table_coords['ymax'] + height_from_table

        # Final camera position
        camera_pos = np.array([camera_x, camera_y, camera_z])

        # look at near the table center
        lookat_xmin = (table_coords['xmin'] + table_coords['xmax']) * 0.5 - table_coords['xsize'] * 0.2
        lookat_xmax = (table_coords['xmin'] + table_coords['xmax']) * 0.5 + table_coords['xsize'] * 0.2
        lookat_zmin = (table_coords['zmin'] + table_coords['zmax']) * 0.5 - table_coords['zsize'] * 0.2
        lookat_zmax = (table_coords['zmin'] + table_coords['zmax']) * 0.5 + table_coords['zsize'] * 0.2

        # Sample lookat position
        lookat_pos = np.array(self.get_state(self.table_stuff['obj_id'])[0])
        lookat_pos[0] = np.random.uniform(lookat_xmin, lookat_xmax)
        lookat_pos[1] = table_coords['ymax']
        lookat_pos[2] = np.random.uniform(lookat_zmin, lookat_zmax)

        if self._mode == 'gui':

            # Calculate yaw, pitch, direction for camera (Bullet needs this)
            # Equations for pitch/yaw is taken from:
            #     gamedev.stackexchange.com/questions/112565/finding-pitch-yaw-values-from-lookat-vector
            camera_direction = lookat_pos - camera_pos
            camera_distance = np.linalg.norm(camera_direction)
            camera_direction = camera_direction / camera_distance
            camera_pitch = np.arcsin(camera_direction[1]) * 180 / np.pi
            camera_yaw = np.arctan2(camera_direction[0], camera_direction[2]) * 180 / np.pi
            p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, lookat_pos)

        return self.get_camera_images(camera_pos, lookat_pos, compute_predicates=compute_predicates)

    def get_tabletop_mask(self, depth_img, table_mask, camera_pos, lookat_pos, camera_up_vector):
        # Filter out table labels to get tabletop ONLY
        H,W = self.params['img_height'], self.params['img_width']
        view_matrix = p.computeViewMatrix(camera_pos, lookat_pos, camera_up_vector)
        view_matrix = np.array(view_matrix).reshape(4,4, order='F')
        inv_cam_ext = np.linalg.inv(view_matrix) # Inverse camera extrinsics matrix

        # negative depth because OpenGL camera z-axis faces behind. 
        xyz_img = sim_util.compute_xyz(depth_img, self.params) # Shape: [H x W x 3]
        xyz_img[..., 2] = -1 * xyz_img[..., 2] # negate the depth to get OpenGL camera frame

        # Multiply each homogenous xyz point by inv camera extrinsics matrix to bring it back to world coordinate frame
        homogenous_xyz = np.concatenate([xyz_img, np.ones((H,W,1))], axis=2)
        world_frame_depth = inv_cam_ext.dot(homogenous_xyz.reshape(-1,4).T) 
        world_frame_depth = world_frame_depth.T.reshape((H,W,4)) # Shape: [H x W x 4]

        # Go from world frame coordinates to obj frame coordinates
        temp = self.get_state(self.table_stuff['obj_id'])
        table_pos = np.array(temp[0])
        table_quaternion = np.roll(temp[1].elements, -1)  # w,x,y,z -> x,y,z,w (which pybullet expects)
        table_orn = np.array(p.getMatrixFromQuaternion(table_quaternion)).reshape(3,3)
        transform_matrix = np.concatenate([table_orn, np.expand_dims(table_pos, axis=1)], axis=1)
        transform_matrix = np.concatenate([transform_matrix, np.array([[0,0,0,1]])], axis=0)
        transform_matrix = np.linalg.inv(transform_matrix)
        obj_frame_depth = transform_matrix.dot(world_frame_depth.reshape(-1,4).T)
        obj_frame_depth = obj_frame_depth.T.reshape((H,W,4))[..., :3] # Shape: [H x W x 3]

        # Get tabletop. Compute histogram of 1cm y-values and pick mode of histogram. 
        # It's kinda like RANSAC in 1 dimension, but using a discretization instead of randomness.
        highest_z_val = round(np.max(obj_frame_depth[table_mask, 2]) + 0.05, 2)
        if highest_z_val < 0:
            highest_z_val = 0.2
        bin_count, bin_edges = np.histogram(obj_frame_depth[table_mask, 2],
                                            bins=int(highest_z_val / .01), 
                                            range=(0,highest_z_val))
        bin_index = np.argmax(bin_count)
        tabletop_z_low = bin_edges[bin_index-1] # a bit less than lower part
        tabletop_z_high = bin_edges[bin_index + 2] # a bit more than higher part
        tabletop_mask = np.logical_and(obj_frame_depth[..., 2] >= tabletop_z_low, 
                                       obj_frame_depth[..., 2] <= tabletop_z_high)
        tabletop_mask = np.logical_and(tabletop_mask, table_mask) # Make sure tabletop_mask is subset of table

        return tabletop_mask

    def get_camera_images(self, camera_pos, lookat_pos, camera_up_vector=None, compute_predicates=False):
        """ Get RGB/Depth/Segmentation images
        """

        if camera_up_vector is None:
            camera_up_vector = self.sample_camera_up_vector(camera_pos, lookat_pos)

        # Compute view/projection matrices and get images
        aspect_ratio = self.params['img_width'] / self.params['img_height']
        view_matrix = p.computeViewMatrix(camera_pos, lookat_pos, camera_up_vector)
        proj_matrix = p.computeProjectionMatrixFOV(self.params['fov'], aspect_ratio, self.params['near'], self.params['far'])
        intrinsic_matrix = projection_to_intrinsics(proj_matrix, self.params['img_width'], self.params['img_height'])
        temp = p.getCameraImage(self.params['img_width'], self.params['img_height'], 
                                viewMatrix=view_matrix, projectionMatrix=proj_matrix,
                                renderer=p.ER_BULLET_HARDWARE_OPENGL) # tuple of: width, height, rgbPixels, depthPixels, segmentation

        # RGB image
        rgb_img = np.reshape(temp[2], (self.params['img_height'], self.params['img_width'], 4))[..., :3]

        # Depth image
        depth_buffer = np.array(temp[3]).reshape(self.params['img_height'],self.params['img_width'])
        depth_img = self.params['far'] * self.params['near'] / \
                    (self.params['far'] - (self.params['far'] - self.params['near']) * depth_buffer)
        # Note: this gives positive z values. this equation multiplies the actual negative z values by -1
        #       Negative z values are because OpenGL camera +z axis points away from image


        # Segmentation image
        seg_img = np.array(temp[4]).reshape(self.params['img_height'],self.params['img_width'])

        # Set near/far clipped depth values to 0. This is indicated by seg_img == -1
        depth_img[seg_img == -1] = 0.

        # Convert seg_img to background (0), table (1), objects (2+). near/far clipped values get background label
        bid_to_seglabel_mapping = {} # Mapping from bid to segmentation label

        # Table bullet ID
        if 'ShapeNet_table_0' in self._obj_id_to_body.keys():
            table_bid = self._obj_id_to_body['ShapeNet_table_0'].bid
            bid_to_seglabel_mapping[table_bid] = 1

        # Object bullet IDs
        # object_bids = [v.bid for k, v in sorted(self._obj_id_to_body.items(), key=lambda x:x[0])
        #               if 'ShapeNet' in k and 'table' not in k]

        object_bids = []
        object_names = []
        for k, v in self._obj_id_to_body.items():
            if 'ShapeNet' in k and 'table' not in k:
                object_bids.append(v.bid)
                index = self.shapenet_obj_stuff['obj_ids'].index(k)
                object_name = self.shapenet_obj_stuff['obj_mesh_filenames'][index]
                subnames = object_name.split('/')
                if len(subnames) > 1:
                    object_name = subnames[-2].strip()
                else:
                    object_name = subnames[-1].strip()
                object_names.append(object_name)

        meta = {'intrinsic_matrix': intrinsic_matrix}
        seglabel_id = self.OBJ_LABEL_START
        num = len(object_bids)
        for i in range(num):
            bid = object_bids[i]
            bid_to_seglabel_mapping[bid] = seglabel_id
            meta[object_names[i]] = seglabel_id
            seglabel_id += 1

        # Conversion happens here
        new_seg_img = np.zeros_like(seg_img)
        for bid, seg_label in bid_to_seglabel_mapping.items():
            mask = seg_img == bid
            if seg_label == 1 and np.count_nonzero(mask) > 0: # table
                mask = self.get_tabletop_mask(depth_img, mask, camera_pos, lookat_pos, camera_up_vector)
            new_seg_img[mask] = seg_label

        # construct meta data
        meta['image_width'] = self.params['img_width']
        meta['image_height'] = self.params['img_height']
        meta['object_names'] = object_names
        camera_pose = view_to_extrinsics(view_matrix)
        meta['camera_pose'] = camera_pose

        # object pose in world and in camera
        object_poses = np.zeros((4, 4, num), dtype=np.float32)
        objects_in_camera = np.zeros((4, 4, num), dtype=np.float32)
        for i in range(num):
            uid = object_bids[i]
            pos, orn = p.getBasePositionAndOrientation(uid, physicsClientId=self._pid)
            object_pose = np.eye(4, dtype=np.float32)
            object_pose[:3, :3] = quat2mat(tf_quat(orn))
            object_pose[:3, 3] = pos
            object_poses[:, :, i] = object_pose
            object_in_camera = np.matmul(camera_pose, object_pose)  # relative pose to camera
            objects_in_camera[:, :, i] = object_in_camera

        meta['object_poses'] = object_poses
        meta['objects_in_camera'] = objects_in_camera

        return {'rgb' : rgb_img,
                'depth' : depth_img,
                'seg' : new_seg_img,
                'orig_seg_img' : seg_img,
                'meta' : meta,
                'view_params' : {
                                'camera_pos' : camera_pos.tolist() if type(camera_pos) == np.ndarray else camera_pos,
                                'lookat_pos' : lookat_pos.tolist() if type(lookat_pos) == np.ndarray else lookat_pos,
                                'camera_up_vector' : camera_up_vector.tolist() if type(camera_up_vector) == np.ndarray else camera_up_vector,
                                }
                }
