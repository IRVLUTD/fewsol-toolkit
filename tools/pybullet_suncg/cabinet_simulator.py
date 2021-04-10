import numpy as np
import os, sys
import pybullet as p
import subprocess as sp
import time
import glob

from collections import namedtuple
from itertools import groupby
from pyquaternion import Quaternion

import pybullet_suncg.simulator as sim
from geo.transform import Transform
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


class CabinetSimulator(sim.Simulator):
    """ Simulate SUNCG rooms with sektion cabinet model
    """

    def __init__(self, mode='direct', bullet_server_binary=None, suncg_data_dir_base=None, 
                 shapenet_data_dir_base=None, sektion_cabinet_dir_base=None, 
                 params=dict(), verbose=False):
        super().__init__(mode=mode, 
                         bullet_server_binary=bullet_server_binary,
                         suncg_data_dir_base=suncg_data_dir_base,
                         shapenet_data_dir_base=shapenet_data_dir_base,
                         params=params,
                         verbose=verbose
                         )
        if sektion_cabinet_dir_base is not None:
            self._sektion_cabinet_dir_base = sektion_cabinet_dir_base
        else:
            raise Exception("MUST provide directory to sektion cabinet model...")

    def get_cab_bbox_coordinates(self):
        """ Same as above, but for cabinet coordinates, considering links (doors/drawers)

            Cabinet always faces in x direciton, so only look at that
        """
        cab_id = self.cabinet_stuff['obj_id']
        cab_coords = self.get_object_bbox_coordinates(cab_id)

        # Top drawer
        top_drawer_coords = self.get_object_bbox_coordinates(cab_id, linkIndex=10)
        cab_coords['xmin'] = min(cab_coords['xmin'], top_drawer_coords['xmin'])
        cab_coords['xmax'] = max(cab_coords['xmax'], top_drawer_coords['xmax'])

        # Bottom drawer
        bottom_drawer_coords = self.get_object_bbox_coordinates(cab_id, linkIndex=12)
        cab_coords['xmin'] = min(cab_coords['xmin'], bottom_drawer_coords['xmin'])
        cab_coords['xmax'] = max(cab_coords['xmax'], bottom_drawer_coords['xmax'])

        return cab_coords

    def get_collision_list(self, obj_id, all_obj_ids):
        """ My own simple collision checking using axis-aligned bounding boxes

            @param obj_id: ID of query object
            @param all_obj_ids: list of IDs of objects you want to check collision with
        """
        if obj_id == 'sektion_cabinet':
            obj_coords = self.get_cab_bbox_coordinates()
        else:
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


    def load_cabinet(self, scene_description):
        """ Load the cabinet. 
        """

        # Load cabinet in canonical position  
        self.load_cab_canonical()
        cab_id = self.cabinet_stuff['obj_id']
        cab_bid = self._obj_id_to_body[cab_id].bid

        # Put the joints in the position
        p.resetJointState(cab_bid, 
                          self.cabinet_stuff['jname_to_ind']['drawer_top'], 
                          scene_description['cabinet']['joint_states']['drawer_top']
                         )
        p.resetJointState(cab_bid, 
                          self.cabinet_stuff['jname_to_ind']['drawer_bottom'], 
                          scene_description['cabinet']['joint_states']['drawer_bottom']
                         )
        p.resetJointState(cab_bid, 
                          self.cabinet_stuff['jname_to_ind']['door_left'], 
                          scene_description['cabinet']['joint_states']['door_left']
                         )
        p.resetJointState(cab_bid, 
                          self.cabinet_stuff['jname_to_ind']['door_right'], 
                          scene_description['cabinet']['joint_states']['door_right']
                         )

        # Move it into the position
        self.set_state(cab_id, 
                       scene_description['cabinet']['position'],
                       Quaternion(w = scene_description['cabinet']['orientation'][0],
                                  x = scene_description['cabinet']['orientation'][1],
                                  y = scene_description['cabinet']['orientation'][2],
                                  z = scene_description['cabinet']['orientation'][3],
                                 )
                      )
        self.simulate(1) # HACK: weirdly, when setting cabinet state, the doors look funky. simulating 1 timestep fixes it.

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
                                  
        # Load the cabinet
        self.load_cabinet(scene_description)
                                  
        # Load the objects
        self.load_objects(scene_description)




    ##### CODE TO SIMULATE SCENES #####

    def load_cab_canonical(self):
        """ Load cabinet in canonical pose/orientation
        """
        cabinet_id = p.loadURDF(self._sektion_cabinet_dir_base + 'urdf/sektion_cabinet.urdf', 
                                basePosition=[0,0,0],
                                baseOrientation=[-1,0,0,1],
                               )

        self.cabinet_stuff = {'obj_id' : 'sektion_cabinet'}
        self.cabinet_stuff['jname_to_ind'] = {
            'drawer_top' : 10,
            'drawer_bottom' : 12,
            'door_right' : 2,
            'door_left' : 7
        }

        cabinet_body = Body(id=self.cabinet_stuff['obj_id'], bid=cabinet_id, vid=-1, cid=None, static=False)
        self._obj_id_to_body[cabinet_body.id] = cabinet_body
        self._bid_to_body[cabinet_body.bid] = cabinet_body

    def generate_cabinet_link_pose(self):
        """ Load cabinet, generate random pose. 
            Pose of cabinet links, i.e. drawers and doors

            link 2 is door_right_joint. revolute. in [pi, 2 * pi]
            link 7 is door_left_joint. revolute. in [0, pi]
            link 10 is drawer_top_joint. revolute. in [0, 0.5]
            link 12 is drawer_bottom_joint. revolute. in [0, 0.5]
        """
        cabinet_id = self._obj_id_to_body['sektion_cabinet'].bid

        # Sample random orientations for the links
        sample = np.random.rand()
        if sample < 0.4: # Top drawer open only
            joint_pos = np.random.uniform(0, 0.5)
            p.resetJointState(cabinet_id, 
                              self.cabinet_stuff['jname_to_ind']['drawer_top'], 
                              joint_pos
                              )
        elif sample < 0.8: # Bottom drawer open only
            joint_pos = np.random.uniform(0, 0.5)
            p.resetJointState(cabinet_id, 
                              self.cabinet_stuff['jname_to_ind']['drawer_bottom'], 
                              joint_pos
                              )            
        else: # neither top nor bottom drawer open
            pass

        # Sample drawer open or not
        door_right_open = np.random.randint(0, 2) # 0 = open, 1 = closed
        door_right_pos = (1 - door_right_open) * np.random.uniform(0, np.pi)
        door_left_open = np.random.randint(0, 2) # 0 = open, 1 = closed
        door_left_pos = (1 - door_left_open) * np.random.uniform(np.pi, 2*np.pi)
        p.resetJointState(cabinet_id,
                          self.cabinet_stuff['jname_to_ind']['door_right'],
                          door_right_pos
                         )
        p.resetJointState(cabinet_id,
                          self.cabinet_stuff['jname_to_ind']['door_left'],
                          door_left_pos
                         )

    def generate_cabinet_pos(self):
        """ After cabinet pose has been chosen, sample position in room

            Return True if successful cabinet placement
        """
        cab_id = self.cabinet_stuff['obj_id']

        room_coords = self.get_object_bbox_coordinates(self.loaded_room.body[0].id) # Get bbox coordinates of room mesh
        cab_coords = self.get_cab_bbox_coordinates() # Canonical Cabinet coordinates

        # List of pybullet object ids to check collsion
        room_obj_ids = [x.body.id for x in self.loaded_room.nodes if x.body is not None]

        # Walls id
        walls_id = self.loaded_room.body[0].id
        floor_coords = self.get_object_bbox_coordinates(self.loaded_room.body[1].id)

        # Sample xz_location until it's not in collision
        in_collision_w_objs = True; in_collision_w_walls = True
        num_tries = 0
        while in_collision_w_objs or in_collision_w_walls:

            xmin = (room_coords['xmin'] - cab_coords['xmin']) / self.params['cab_init_factor']
            xmax = (room_coords['xmax'] - cab_coords['xmax']) * self.params['cab_init_factor']
            random_start_xpos = np.random.uniform(xmin, xmax)
            ypos = floor_coords['ymax'] - cab_coords['ymin'] + np.random.uniform(0, 0.1) # add a bit of space so no collision w/ carpets...
            zmin = (room_coords['zmin'] - cab_coords['zmin']) / self.params['cab_init_factor']
            zmax = (room_coords['zmax'] - cab_coords['zmax']) * self.params['cab_init_factor']
            random_start_zpos = np.random.uniform(zmin, zmax)
            if (xmax < xmin) or (zmax < zmin): # cabinet is too large. pick a new room
                break
            random_start_pos = np.array([random_start_xpos, ypos, random_start_zpos])
            self.set_state(cab_id, random_start_pos, Quaternion(x=-1, y=0, z=0, w=1))
            self.simulate(1) # HACK: weirdly, when setting cabinet state, the doors look funky. simulating 1 timestep fixes it.

            if num_tries > self.params['max_initialization_tries']:
                break # this will force code to pick a new room
            num_tries += 1

            # Check if it's in collision with anything
            in_collision_w_walls = self.get_closest_point(cab_id, walls_id).distance < 0
            in_collision_w_objs = len(self.get_collision_list(cab_id, room_obj_ids)) > 0 # Simpler coarse collision checking with these objects

        if in_collision_w_objs or in_collision_w_walls: # if still in collision, then it's because we exhausted number of tries
            self.remove(cab_id)
            return False

        # Let the cabinet fall for about 1 second
        self.simulate(300)

        # Check if it fell down
        up_orientation = np.array([1,-1,0,0]) # w,x,y,z
        up_orientation = up_orientation / np.linalg.norm(up_orientation)
        is_up = np.allclose(self.get_state(cab_id)[1].elements, up_orientation, atol=1e-1)

        # Remove the cabinet if it fell down or is in collision
        if not is_up:
            self.remove(cab_id)
            return False
         
        # point debug camera at cabinet
        if self._mode == 'gui':
            cab_pos = list(self.get_state(cab_id)[0])
            p.resetDebugVisualizerCamera(cameraDistance=1.5,
                                         cameraYaw=-90.0,
                                         cameraPitch=-45.0,
                                         cameraTargetPosition=cab_pos,
                                         physicsClientId=self._pid)

        return True


    def generate_random_shapenet_models(self):
        """ Sample random ShapeNet models

            NOTE: This is to be called AFTER self.generate_cabinet_pos() has been called
        """

        ##### Sample random ShapeNet models #####
        obj_mesh_filenames = []
        obj_ids = []
        obj_scale_factors = []

        sn_obj_p = self.params['shapenet_obj_percentage']
        cub_p = self.params['cuboid_percentage']
        cyl_p = self.params['cylinder_percentage']

        # Get max/min coordinates of cabinet
        cab_coords = self.get_cab_bbox_coordinates()
        
        i = 0
        num_objects_for_scene = self.params['num_objects_in_scene']
        if self._verbose:
            print(f"Number of objects chosen for scene: {num_objects_for_scene}")
        while len(obj_ids) < num_objects_for_scene:
            obj_id = f'ShapeNet_obj_{i}'

            ### Sample the object ###
            obj_transform = Transform()
            sample = np.random.rand()
            if sample < sn_obj_p:
                obj_type = 'sn'

                synsets = list(self.params['object_ids'].keys())
                synset_to_sample = np.random.choice(synsets)
                model_dir = np.random.choice(self.params['object_ids'][synset_to_sample])
                model_dir = self._shapenet_data_dir_base + self.params['taxonomy_dict'][synset_to_sample] + '/' + model_dir + '/'
                obj_mesh_filename = model_dir + 'models/model_normalized.obj'
                obj_mesh_filenames.append(model_dir + 'models/model_normalized.obj')

                ### Create an object in pybullet ###
                obj_body = self.add_mesh(obj_id, obj_mesh_filename, obj_transform, obj_mesh_filename)

            elif sample < sn_obj_p + cub_p: # cuboid
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

            elif sample < sn_obj_p + cub_p + cyl_p: # cylinder
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

            ### Sample scale of object ###
            canonical_obj_coords = self.get_object_bbox_coordinates(obj_id)

            # Compute scale factor
            xscale_factor = 1; yscale_factor = 1; zscale_factor = 1
            if canonical_obj_coords['xsize'] / cab_coords['xsize'] > self.params['max_xratio']:
                xscale_factor = self.params['max_xratio'] * cab_coords['xsize'] / (canonical_obj_coords['xsize'])
            if canonical_obj_coords['ysize'] / cab_coords['ysize'] > self.params['max_yratio']:
                yscale_factor = self.params['max_yratio'] * cab_coords['ysize'] / (canonical_obj_coords['ysize'])
            if canonical_obj_coords['zsize'] / cab_coords['zsize'] > self.params['max_zratio']:
                zscale_factor = self.params['max_zratio'] * cab_coords['zsize'] / (canonical_obj_coords['zsize'])
            max_scale_factor = min(xscale_factor, yscale_factor, zscale_factor)
            obj_scale_factor = np.random.uniform(max_scale_factor * 0.75, max_scale_factor)
            obj_scale_factors.append(obj_scale_factor)
            obj_transform.rescale(obj_scale_factor)


            ##### Sample random location/orientation for object #####

            # Make sure the object is not in collision with any other object
            in_collision = True
            num_tries = 0
            while in_collision:

                # Get all objects that are straight (these are the ones that could be supporting objects)
                straight_obj_ids = [x for x in obj_ids if np.allclose(self.get_state(x)[1].elements, np.array([1,0,0,0]))]

                # Sample a random starting orientation
                sample = np.random.rand()
                if sample < 0.5: # Simulate straight up

                    q = np.array([0,0,0,1])
                    if obj_type in ['cub', 'cyl']:
                        q = np.array([-1,0,0,1]) # y-axis up
                    extra_y = 0.
                    x_range_min = cab_coords['xmin'] - canonical_obj_coords['xmin']
                    x_range_max = cab_coords['xmax'] - canonical_obj_coords['xmax']
                    z_range_min = cab_coords['zmin'] - canonical_obj_coords['zmin']
                    z_range_max = cab_coords['zmax'] - canonical_obj_coords['zmax']

                elif sample < 0.75 and len(straight_obj_ids) >= 1: # put it one another object

                    q = np.array([0,0,0,1])
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
                    x_range_min = cab_coords['xmin'] - canonical_obj_coords['xmin']
                    x_range_max = cab_coords['xmax'] - canonical_obj_coords['xmax']
                    z_range_min = cab_coords['zmin'] - canonical_obj_coords['zmin']
                    z_range_max = cab_coords['zmax'] - canonical_obj_coords['zmax']

                # Load this in and get axis-aligned bounding box
                self.remove(obj_id)
                obj_transform._r = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3]) # HACK. Not good to access "private" attribute _r, but whatevs
                if obj_type == 'sn':
                    obj_body = self.add_mesh(obj_id, obj_mesh_filename, obj_transform, obj_mesh_filename)
                elif obj_type == 'cub':
                    obj_body = self.add_cuboid(obj_id, obj_transform, half_extents=half_extents, rgba_color=rgba_color)
                elif obj_type == 'cyl':
                    obj_dy = self.add_cylinder(obj_id, obj_transform, height=height, radius=radius, rgba_color=rgba_color)
                obj_coords = self.get_object_bbox_coordinates(obj_id) # scaled coordinates

                # Sample a random starting location
                random_start_xpos = np.random.uniform(x_range_min, x_range_max)
                ypos = cab_coords['ymax'] - obj_coords['ymin'] + extra_y
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
                continue

            # If we get here, then object has successfully by initialized
            obj_ids.append(obj_id)
            i += 1


        self.shapenet_obj_stuff = {'obj_ids' : obj_ids,
                                   'obj_mesh_filenames': obj_mesh_filenames,
                                   'obj_scale_factors' : obj_scale_factors,
                                  }

    def remove_fallen_objects(self):
        """ Remove any objects that have fallen (i.e. are lower than the cabinet drawers)
        """
        bottom_drawer_coords = self.get_object_bbox_coordinates(self.cabinet_stuff['obj_id'], linkIndex=12)
        num_objects = len(self.shapenet_obj_stuff['obj_ids'])
        
        fallen_obj_ids = []
        for obj_id in self.shapenet_obj_stuff['obj_ids']:
            obj_coords = self.get_object_bbox_coordinates(obj_id)
            obj_ypos = (obj_coords['ymin'] + obj_coords['ymax']) / 2.
            if obj_ypos < bottom_drawer_coords['ymin']:
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

    def export_scene_to_dictionary(self):
        """ Exports the PyBullet scene to a dictionary
        """

        # Initialize empty scene description
        scene_description = {}

        # House/Room description
        room_description = {'house_id' : self.loaded_room.house_id,
                            'room_id' : self.loaded_room.id}
        scene_description['room'] = room_description

        # Cabinet description
        cab_id = self.cabinet_stuff['obj_id']
        cab_bid = self._obj_id_to_body[cab_id].bid
        temp = self.get_state(cab_id)
        cabinet_description = {
            'position' : list(temp[0]),
            'orientation' : list(temp[1].elements), # w,x,y,z quaternion
            'joint_states' : {},
        }

        drawer_b_pos = p.getJointState(cab_bid, self.cabinet_stuff['jname_to_ind']['drawer_bottom'])[0]
        cabinet_description['joint_states']['drawer_bottom'] = drawer_b_pos
        drawer_t_pos = p.getJointState(cab_bid, self.cabinet_stuff['jname_to_ind']['drawer_top'])[0]
        cabinet_description['joint_states']['drawer_top'] = drawer_t_pos
        door_l_pos = p.getJointState(cab_bid, self.cabinet_stuff['jname_to_ind']['door_left'])[0]
        cabinet_description['joint_states']['door_left'] = door_l_pos
        door_r_pos = p.getJointState(cab_bid, self.cabinet_stuff['jname_to_ind']['door_right'])[0]
        cabinet_description['joint_states']['door_right'] = door_r_pos

        scene_description['cabinet'] = cabinet_description

        # Get descriptions of objects on/in cabinet
        object_descriptions = []
        for i, obj_id in enumerate(self.shapenet_obj_stuff['obj_ids']):

            mesh_filename = self.shapenet_obj_stuff['obj_mesh_filenames'][i]
            temp = self.get_state(obj_id)
            pos = list(temp[0])
            orientation = list(temp[1].elements) # w,x,y,z quaternion
            scale = self.shapenet_obj_stuff['obj_scale_factors'][i]

            obj_coords = self.get_object_bbox_coordinates(obj_id)
            bbox_center = list(sim_util.get_aligned_bbox3D_center(obj_coords))

            description = {'obj_id' : obj_id,
                           'mesh_filename' : mesh_filename.replace(self._shapenet_data_dir_base, ''),
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

            # Generate cabinet drawer poses, and cabinet position
            self.load_cab_canonical()
            self.generate_cabinet_link_pose()
            cab_pos_success = self.generate_cabinet_pos()
            if not cab_pos_success:
                continue # start over

            # Objects
            self.generate_random_shapenet_models()
            self.simulate(self.params['simulation_steps'])
            self.remove_fallen_objects()
            
            # Check some bad situations
            if len(self.shapenet_obj_stuff['obj_ids']) < self.params['min_objects_in_scene']: # Too many objects fell off
                continue # start over
            if self.get_state(self.cabinet_stuff['obj_id'])[0][1] < -0.1: # cabinet fell way way way down
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

    def sample_cabinet_view(self, compute_predicates=False):
        """ Sample a view near cabinet
        """

        # Sample position on xz bbox of cabinet
        cab_coords = self.get_cab_bbox_coordinates()

        # First, select a side
        xz_bbox_side_probs = np.array([cab_coords['xsize'], # x side 1
                                       cab_coords['zsize'], # z side 1
                                       cab_coords['xsize'], # x side 2
                                       cab_coords['zsize']] # z side 2
                                     )
        xz_bbox_side_probs = xz_bbox_side_probs / np.sum(xz_bbox_side_probs)
        side = np.random.choice(range(4), p=xz_bbox_side_probs)
        if side == 0: # x side 1
            p1 = np.array([cab_coords['xmin'], cab_coords['zmin']])
            p2 = np.array([cab_coords['xmax'], cab_coords['zmin']])
            side_length = np.linalg.norm(p2 - p1)
            other_side_length = np.linalg.norm(p2 - np.array([cab_coords['xmax'], cab_coords['zmax']]))
        elif side == 1: # z side 1
            p1 = np.array([cab_coords['xmax'], cab_coords['zmin']])
            p2 = np.array([cab_coords['xmax'], cab_coords['zmax']])
            side_length = np.linalg.norm(p2 - p1)
            other_side_length = np.linalg.norm(p2 - np.array([cab_coords['xmin'], cab_coords['zmax']]))
        elif side == 2: # x side 2
            p1 = np.array([cab_coords['xmax'], cab_coords['zmax']])
            p2 = np.array([cab_coords['xmin'], cab_coords['zmax']])
            side_length = np.linalg.norm(p2 - p1)
            other_side_length = np.linalg.norm(p2 - np.array([cab_coords['xmin'], cab_coords['zmin']]))
        elif side == 3: # z side 2
            p1 = np.array([cab_coords['xmin'], cab_coords['zmax']])
            p2 = np.array([cab_coords['xmin'], cab_coords['zmin']])
            side_length = np.linalg.norm(p2 - p1)
            other_side_length = np.linalg.norm(p2 - np.array([cab_coords['xmax'], cab_coords['zmin']]))

        # Select point on that side uniformly
        point = p1 + (p2 - p1) * np.random.uniform(0,1)

        # Sample xz distance from that point
        dist_from_cab = np.random.uniform(0.0, 0.15)
        theta = np.radians(-90)
        rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                               [np.sin(theta), np.cos(theta)]])
        away_from_cab_direction = rot_matrix.dot ( (p2 - p1) / np.linalg.norm(p2 - p1) )
        camera_x, camera_z = point + dist_from_cab * away_from_cab_direction

        # Sample y distance
        height_from_cab = np.random.uniform(.5, 1.2) # anywhere from .5m to 1.2m above cabinet
        camera_y = cab_coords['ymax'] + height_from_cab

        # Final camera position
        camera_pos = np.array([camera_x, camera_y, camera_z])

        if side in [0,2]:
            lookat_xmin = max(point[0] - side_length*0.2, cab_coords['xmin'])
            lookat_xmax = min(point[0] + side_length*0.2, cab_coords['xmax'])
            if side == 0:
                lookat_zmin = point[1] + other_side_length*0.1
                lookat_zmax = point[1] + other_side_length*0.5
            else: # side == 2
                lookat_zmin = point[1] - other_side_length*0.5
                lookat_zmax = point[1] - other_side_length*0.1
        else: # side in [1,3]
            lookat_zmin = max(point[1] - side_length*0.2, cab_coords['zmin'])
            lookat_zmax = min(point[1] + side_length*0.2, cab_coords['zmax'])
            if side == 1:
                lookat_xmin = point[0] - other_side_length*0.5
                lookat_xmax = point[0] - other_side_length*0.1
            else: # side == 3
                lookat_xmin = point[0] + other_side_length*0.1
                lookat_xmax = point[0] + other_side_length*0.5

        # Sample lookat position
        lookat_pos = np.array(self.get_state(self.cabinet_stuff['obj_id'])[0])
        lookat_pos[0] = np.random.uniform(lookat_xmin, lookat_xmax)
        lookat_pos[1] = cab_coords['ymax']
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

    def get_cabinet_top_mask(self, depth_img, mask, camera_pos, lookat_pos, camera_up_vector):
        """ Compute cabinet top mask of mask ONLY

            @param depth_img: a [H x W x 3] numpy array
            @param mask: a [H x W] numpy bool array. We are looking for cabinet top of this mask
        """

        # Filter out cabinet labels to get cabinet top ONLY
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
        temp = self.get_state(self.cabinet_stuff['obj_id'])
        cab_pos = np.array(temp[0])
        cab_quaternion = np.roll(temp[1].elements, -1)  # w,x,y,z -> x,y,z,w (which pybullet expects)
        cab_orn = np.array(p.getMatrixFromQuaternion(cab_quaternion)).reshape(3,3)
        transform_matrix = np.concatenate([cab_orn, np.expand_dims(cab_pos, axis=1)], axis=1)
        transform_matrix = np.concatenate([transform_matrix, np.array([[0,0,0,1]])], axis=0)
        transform_matrix = np.linalg.inv(transform_matrix)
        obj_frame_depth = transform_matrix.dot(world_frame_depth.reshape(-1,4).T)
        obj_frame_depth = obj_frame_depth.T.reshape((H,W,4))[..., :3] # Shape: [H x W x 3]

        # But cab frame is z-axis up, so rotate to y-axis up
        euler_transform = p.getQuaternionFromEuler([-np.pi/2, 0,0])
        y_up_obj_frame_transform = p.getMatrixFromQuaternion(euler_transform) # yaw (z), pitch (y), roll (x)
        y_up_obj_frame_transform = np.array(y_up_obj_frame_transform).reshape(3,3)
        y_up_obj_frame_depth = y_up_obj_frame_transform.dot(obj_frame_depth.reshape(-1,3).T)
        y_up_obj_frame_depth = y_up_obj_frame_depth.T.reshape((H,W,3)) # Shape: [H x W x 3]

        # Get cabinet top. Compute histogram of 1cm y-values and pick mode of histogram. 
        # It's kinda like RANSAC in 1 dimension, but using a discretization instead of randomness.
        highest_y_val = round(np.max(y_up_obj_frame_depth[mask, 1]) + 0.05, 2)
        bin_count, bin_edges = np.histogram(y_up_obj_frame_depth[mask, 1], 
                                            bins=int(highest_y_val / .01), 
                                            range=(0,highest_y_val))
        bin_index = np.argmax(bin_count)
        cabtop_y_low = bin_edges[bin_index-1] # a bit less than lower part
        cabtop_y_high = bin_edges[bin_index + 2] # a bit more than higher part
        cabinet_top_mask = np.logical_and(y_up_obj_frame_depth[..., 1] >= cabtop_y_low, 
                                       y_up_obj_frame_depth[..., 1] <= cabtop_y_high)
        cabinet_top_mask = np.logical_and(cabinet_top_mask, mask) # Make sure cabinet_top_mask is subset of cabinet

        return cabinet_top_mask


    def get_camera_images(self, camera_pos, lookat_pos, camera_up_vector=None, compute_predicates=False):
        """ Get RGB/Depth/Segmentation images
        """

        if camera_up_vector is None:
            camera_up_vector = self.sample_camera_up_vector(camera_pos, lookat_pos)

        # Compute some stuff
        aspect_ratio = self.params['img_width']/self.params['img_height']
        e = 1/(np.tan(np.radians(self.params['fov']/2.)))
        t = self.params['near']/e; b = -t
        r = t*aspect_ratio; l = -r

        # Compute view/projection matrices and get images
        view_matrix = p.computeViewMatrix(camera_pos, lookat_pos, camera_up_vector)
        proj_matrix = p.computeProjectionMatrixFOV(self.params['fov'], aspect_ratio, self.params['near'], self.params['far'])
        temp = p.getCameraImage(self.params['img_width'], self.params['img_height'], 
                                viewMatrix=view_matrix, projectionMatrix=proj_matrix,
                                renderer=p.ER_BULLET_HARDWARE_OPENGL,
                                flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                               ) # tuple of: width, height, rgbPixels, depthPixels, segmentation

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
        objID_seg_img = seg_img & ((1 << 24) - 1) # Map seg_img to PyBullet object IDs
        objID_seg_img[objID_seg_img == 16777215] = -1 # Set -1 back to -1
        linkID_seg_img = (seg_img >> 24) - 1 # Map seg_img to PyBullet link IDs
        linkID_seg_img[linkID_seg_img == -2] = -1 # Set -1 back to -1

        # Set near/far clipped depth values to 0. This is indicated by objID_seg_img == -1
        depth_img[objID_seg_img == -1] = 0.

        # Convert objID_seg_img to background (0), cabinet (1), objects (2+). near/far clipped values get background label
        bid_to_seglabel_mapping = {} # Mapping from bid to segmentation label

        # Cabinet bullet ID
        #   link 2 is left door
        #   link 4 is left door knob
        #   link 7 is right door
        #   link 9 is right door knob
        #   link 10 is top drawer
        #   link 11 is top drawer handle
        #   link 12 is bottom drawer
        #   link 13 is bottom drawer handle
        link_to_seglabel_mapping = {}
        if 'sektion_cabinet' in self._obj_id_to_body.keys():
            cab_bid = self._obj_id_to_body['sektion_cabinet'].bid
            bid_to_seglabel_mapping[cab_bid] = 1
            # Top drawer
            link_to_seglabel_mapping[10] = 2
            link_to_seglabel_mapping[11] = 2
            # Bottom drawer
            link_to_seglabel_mapping[12] = 3
            link_to_seglabel_mapping[13] = 3

        # Object bullet IDs
        object_bids = [v.bid for k, v in sorted(self._obj_id_to_body.items(), key=lambda x:x[0]) 
                       if 'ShapeNet' in k]
        seglabel_id = self.OBJ_LABEL_START
        for bid in object_bids:
            bid_to_seglabel_mapping[bid] = seglabel_id
            seglabel_id += 1

        # Conversion happens here
        new_seg_img = np.zeros_like(objID_seg_img)
        for bid, seg_label in bid_to_seglabel_mapping.items():
            mask = objID_seg_img == bid
            if seg_label == 1: # cabinet
                mask = np.logical_and(mask, linkID_seg_img == -1) # Cabinet mask ONLY (no drawers/doors)
                if np.count_nonzero(mask) > 0: # if no cabinet in this image, the calling function will throw away the image
                    mask = self.get_cabinet_top_mask(depth_img, mask, camera_pos, lookat_pos, camera_up_vector)
            new_seg_img[mask] = seg_label
        for linkid, seg_label in link_to_seglabel_mapping.items():
            mask = (linkID_seg_img == linkid)
            new_seg_img[mask] = seg_label


        # Compute predicates w.r.t. this camera view
        predicates = []
        if compute_predicates: 
            pairwise_predicate_methods = [
                self.compute_left_right_predicates,
                self.compute_front_behind_predicates,
                self.compute_above_below_predicates,
                self.compute_occluded_predicates,
            ]
            single_arg_predicate_methods = [
                self.compute_inside_predicates,
            ]
            predicates = self.compute_predicates(camera_pos, 
                                                 lookat_pos, 
                                                 camera_up_vector, 
                                                 bid_to_seglabel_mapping,
                                                 new_seg_img,
                                                 pairwise_predicate_methods,
                                                 single_arg_predicate_methods,
                                                )
        else:
            predicates = []

        # Filter out some predictes
        #   if an object is in a drawer, it can't be left/right an object NOT in same drawer
        #   if an object is in a drawer, it can't be front/behind an object NOT in same drawer
        inside_drawer = {}
        for predicate in predicates:
            if predicate[4] == 'inside_of':
                inside_drawer[predicate[0]] = predicate[2] # e.g. ShapeNet_obj_2 : 'drawer_top'
        valid_indices = []
        for ind, predicate in enumerate(predicates):
            valid_indices.append(ind)
            obj1 = predicate[0]; obj2 = predicate[2]; pred = predicate[4]
            if obj1 in inside_drawer and pred in ['left', 'right', 'behind', 'front']:
                if not (obj2 in inside_drawer and \
                        inside_drawer[obj1] == inside_drawer[obj2]):
                    # print(f"{obj1} is inside {inside_drawer[obj1]}, obj2 is {'in ' + inside_drawer[obj2] if obj2 in inside_drawer else 'not'}")
                    valid_indices.pop(-1)
            if obj2 in inside_drawer and pred in ['left', 'right', 'behind', 'front']:
                if not (obj1 in inside_drawer and \
                        inside_drawer[obj1] == inside_drawer[obj2]):
                    # print(f"{obj2} is inside {inside_drawer[obj2]}, obj1 is {'in ' + inside_drawer[obj1] if obj1 in inside_drawer else 'not'}")
                    valid_indices.pop(-1)

        predicates = [predicates[i] for i in valid_indices]


        return {'rgb' : rgb_img,
                'depth' : depth_img,
                'seg' : new_seg_img,
                'orig_seg_img' : objID_seg_img,
                'view_params' : {
                                'camera_pos' : camera_pos.tolist() if type(camera_pos) == np.ndarray else camera_pos,
                                'lookat_pos' : lookat_pos.tolist() if type(lookat_pos) == np.ndarray else lookat_pos,
                                'camera_up_vector' : camera_up_vector.tolist() if type(camera_up_vector) == np.ndarray else camera_up_vector,
                                },
                'predicates' : predicates,
                }




    ##### CODE TO COMPUTE PREDICATES #####

    def compute_inside_predicates(self, obj_predicate_args, other_args):
        """ Compute whether o1 is inside top drawer or bottom drawer or None

            Relation rules:
                1) The lowest contact point with o1 and drawer MUST be in bottom half of drawer.
        """
        obj_id = obj_predicate_args['obj_id']
        self.simulate(5) # MUST run this for a few time steps before calling get_contacts()
        obj_contacts = self.get_contacts(obj_id, 'sektion_cabinet', only_closest_contact_per_pair=False)
        
        # if this is nonzero, there is contact with sektion cabinet
        if len(obj_contacts) > 0: 
            contact_list = obj_contacts[(obj_id, 'sektion_cabinet')]

            # top drawer
            dt_ind = self.cabinet_stuff['jname_to_ind']['drawer_top']
            drawer_contacts = list(filter(lambda x: x.linkIndexB == dt_ind, contact_list))
            if len(drawer_contacts) > 0: # if this is nonzero, there is contact with top drawer
                lowest_y_contact = min([x.positionOnBInWS[1] for x in drawer_contacts])
                drawer_coords = self.get_object_bbox_coordinates('sektion_cabinet', linkIndex=dt_ind)
                if drawer_coords['ymin'] <= lowest_y_contact <= (drawer_coords['ymin']+drawer_coords['ymax'])/2.:
                    other_args['predicates'].append((obj_id, 'drawer_top', 2, 'inside_of'))

            # bottom drawer
            db_ind = self.cabinet_stuff['jname_to_ind']['drawer_bottom']
            drawer_contacts = list(filter(lambda x: x.linkIndexB == db_ind, contact_list))
            if len(drawer_contacts) > 0: # if this is nonzero, there is contact with bottom drawer
                lowest_y_contact = min([x.positionOnBInWS[1] for x in drawer_contacts])
                drawer_coords = self.get_object_bbox_coordinates('sektion_cabinet', linkIndex=db_ind)
                if drawer_coords['ymin'] <= lowest_y_contact <= (drawer_coords['ymin']+drawer_coords['ymax'])/2.:
                    other_args['predicates'].append((obj_id, 'drawer_bottom', 3, 'inside_of'))








