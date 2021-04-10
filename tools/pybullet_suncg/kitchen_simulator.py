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


class KitchenSimulator(sim.Simulator):
    """ Simulate SUNCG rooms with sektion cabinet model
    """

    def __init__(self, mode='direct', bullet_server_binary=None, suncg_data_dir_base=None, 
                 shapenet_data_dir_base=None, kitchen_dir_base=None, 
                 params=dict(), verbose=False):
        super().__init__(mode=mode, 
                         bullet_server_binary=bullet_server_binary,
                         suncg_data_dir_base=suncg_data_dir_base,
                         shapenet_data_dir_base=shapenet_data_dir_base,
                         params=params,
                         verbose=verbose
                         )
        if kitchen_dir_base is not None:
            self._kitchen_dir_base = kitchen_dir_base
        else:
            raise Exception("MUST provide directory to kitchen model...")

    def get_cab_bbox_coordinates(self):
        """ Same as above, but for cabinet coordinates, considering links (doors/drawers)

            Cabinet always faces in x direciton, so only look at that
        """
        kitchen_id = self.kitchen_stuff['obj_id']
        cab_coords = self.get_object_bbox_coordinates(kitchen_id, 
                            linkIndex=self.kitchen_stuff['jname_to_ind']['sektion_cabinet'])

        # Top drawer
        top_drawer_coords = self.get_object_bbox_coordinates(kitchen_id, 
                                    linkIndex=self.kitchen_stuff['jname_to_ind']['drawer_top'])
        cab_coords['xmin'] = min(cab_coords['xmin'], top_drawer_coords['xmin'])
        cab_coords['xmax'] = max(cab_coords['xmax'], top_drawer_coords['xmax'])

        # Bottom drawer
        bottom_drawer_coords = self.get_object_bbox_coordinates(kitchen_id,
                                    linkIndex=self.kitchen_stuff['jname_to_ind']['drawer_bottom'])
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


    def load_cabinet_pose(self, scene_description):
        """ Load the cabinet. 
        """

        kitchen_id = self.kitchen_stuff['obj_id']
        kitchen_bid = self._obj_id_to_body[kitchen_id].bid

        # Load joint positions
        p.resetJointState(kitchen_bid, 
                          self.kitchen_stuff['jname_to_ind']['drawer_top'], 
                          scene_description['cabinet']['joint_states']['drawer_top']
                         )
        p.resetJointState(kitchen_bid, 
                          self.kitchen_stuff['jname_to_ind']['drawer_bottom'], 
                          scene_description['cabinet']['joint_states']['drawer_bottom']
                         )
        p.resetJointState(kitchen_bid, 
                          self.kitchen_stuff['jname_to_ind']['door_left'], 
                          scene_description['cabinet']['joint_states']['door_left']
                         )
        p.resetJointState(kitchen_bid, 
                          self.kitchen_stuff['jname_to_ind']['door_right'], 
                          scene_description['cabinet']['joint_states']['door_right']
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

        # Load the kitchen
        self.load_kitchen()

        # Load the cabinet
        self.load_cabinet_pose(scene_description)
                                  
        # Load the objects
        self.load_objects(scene_description)




    ##### CODE TO SIMULATE SCENES #####

    def load_kitchen(self):
        """ Load cabinet in canonical pose/orientation
        """
        kitchen_id = p.loadURDF(self._kitchen_dir_base + 'urdf/kitchen_part_right_gen_convex.urdf',
                          useFixedBase=True, 
                          basePosition=[0.,1.476,0.], baseOrientation=[-1,0,0,1]) # Y-axis up

        self.kitchen_stuff = {'obj_id' : 'kitchen'}
        self.kitchen_stuff['jname_to_ind'] = {
            'sektion_cabinet' : 43,
            'drawer_top' : 56,
            'drawer_bottom' : 58,
            'door_right' : 48,
            'door_left' : 53,
        }

        kitchen_body = Body(id=self.kitchen_stuff['obj_id'], bid=kitchen_id, vid=-1, cid=None, static=False)
        self._obj_id_to_body[kitchen_body.id] = kitchen_body
        self._bid_to_body[kitchen_body.bid] = kitchen_body

        # point debug camera at cabinet
        if self._mode == 'gui':
            cab_coords = self.get_cab_bbox_coordinates()
            cab_center = sim_util.get_aligned_bbox3D_center(cab_coords)
            cab_pos = [cab_center[0], cab_coords['ymax'], cab_center[2]]
            p.resetDebugVisualizerCamera(cameraDistance=1.5,
                                         cameraYaw=-90.0,
                                         cameraPitch=-45.0,
                                         cameraTargetPosition=cab_pos,
                                         physicsClientId=self._pid)

    def generate_cabinet_link_pose(self):
        """ Load cabinet, generate random pose. 
            Pose of cabinet links, i.e. drawers and doors

            link 43 indigo_transform. this is the entire (bottom right) sektion cabinet
            link 48 indigo_door_right_joint. revolute. in [0, 1/2*pi]
            link 53 indigo_door_left_joint. revolute. in [3/2*pi, 2*pi]
            link 56 indigo_drawer_top_joint. prismatic. in [0, 0.5]
            link 57 indigo_drawer_handle_top_joint
            link 58 indigo_drawer_bottom_joint. prismatic. in [0, 0.5]
            link 59 indigo_drawer_handle_bottom_joint
        """
        kitchen_id = self._obj_id_to_body['kitchen'].bid

        # Sample random orientations for the links
        sample = np.random.rand()
        if sample < 0.35: # Top drawer open only
            joint_pos = np.random.uniform(0, 0.5)
            p.resetJointState(kitchen_id, 
                              self.kitchen_stuff['jname_to_ind']['drawer_top'], 
                              joint_pos
                              )
        elif sample < 0.7: # Bottom drawer open only
            joint_pos = np.random.uniform(0, 0.5)
            p.resetJointState(kitchen_id, 
                              self.kitchen_stuff['jname_to_ind']['drawer_bottom'], 
                              joint_pos
                              )            
        else: # neither top nor bottom drawer open
            pass

        # link 48 indigo_door_right_joint. revolute. in [0, 1/2*pi]
        # link 53 indigo_door_left_joint. revolute. in [3/2*pi, 2*pi]

        # Sample drawer open or not
        door_right_open = np.random.binomial(1, 0.25) # 0 = open, 1 = closed
        door_right_pos = (1 - door_right_open) * np.random.uniform(0, 1/2*np.pi)
        door_left_open = np.random.binomial(1, 0.25) # 0 = open, 1 = closed
        door_left_pos = (1 - door_left_open) * np.random.uniform(3/2*np.pi, 2*np.pi)
        p.resetJointState(kitchen_id,
                          self.kitchen_stuff['jname_to_ind']['door_right'],
                          door_right_pos
                         )
        p.resetJointState(kitchen_id,
                          self.kitchen_stuff['jname_to_ind']['door_left'],
                          door_left_pos
                         )


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

                # Check for collision with other objects, and kitchen
                in_collision_w_objs = len(self.get_collision_list(obj_id, obj_ids)) > 0
                temp = self.get_closest_point(obj_id, self.kitchen_stuff['obj_id'])
                in_collision_w_walls = temp.linkIndexB != self.kitchen_stuff['jname_to_ind']['sektion_cabinet'] and temp.distance < 0
                in_collision = in_collision_w_objs or in_collision_w_walls

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
        bottom_drawer_coords = self.get_object_bbox_coordinates(self.kitchen_stuff['obj_id'], 
                                        linkIndex=self.kitchen_stuff['jname_to_ind']['drawer_bottom'])
        num_objects = len(self.shapenet_obj_stuff['obj_ids'])
        
        fallen_obj_ids = []
        for obj_id in self.shapenet_obj_stuff['obj_ids']:
            obj_coords = self.get_object_bbox_coordinates(obj_id)
            obj_ypos = (obj_coords['ymin'] + obj_coords['ymax']) / 2.
            if obj_ypos < bottom_drawer_coords['ymin']:
                fallen_obj_ids.append(obj_id)
            if obj_coords['ymin'] > 1.45: # Hack. object stuck in top right cabinet...
                print("Object stuck in top cabinet")
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
                                              for i in valid_indices]
        self.shapenet_obj_stuff['obj_mesh_filenames'] = [self.shapenet_obj_stuff['obj_mesh_filenames'][i] 
                                                         for i in valid_indices]
        self.shapenet_obj_stuff['obj_scale_factors'] = [self.shapenet_obj_stuff['obj_scale_factors'][i] 
                                                        for i in valid_indices]

    def export_scene_to_dictionary(self):
        """ Exports the PyBullet scene to a dictionary
        """

        # Initialize empty scene description
        scene_description = {}

        # Cabinet description
        cabinet_description = { 'joint_states' : {} }

        kitchen_id = self.kitchen_stuff['obj_id']
        kitchen_bid = self._obj_id_to_body[kitchen_id].bid
        drawer_b_pos = p.getJointState(kitchen_bid, self.kitchen_stuff['jname_to_ind']['drawer_bottom'])[0]
        cabinet_description['joint_states']['drawer_bottom'] = drawer_b_pos
        drawer_t_pos = p.getJointState(kitchen_bid, self.kitchen_stuff['jname_to_ind']['drawer_top'])[0]
        cabinet_description['joint_states']['drawer_top'] = drawer_t_pos
        door_l_pos = p.getJointState(kitchen_bid, self.kitchen_stuff['jname_to_ind']['door_left'])[0]
        cabinet_description['joint_states']['door_left'] = door_l_pos
        door_r_pos = p.getJointState(kitchen_bid, self.kitchen_stuff['jname_to_ind']['door_right'])[0]
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
            self.load_kitchen()

            # Generate cabinet drawer poses
            self.generate_cabinet_link_pose()

            # Objects
            self.generate_random_shapenet_models()
            self.simulate(self.params['simulation_steps'])
            self.remove_fallen_objects()
            
            # Check some bad situations
            if len(self.shapenet_obj_stuff['obj_ids']) < self.params['min_objects_in_scene']: # Too many objects fell off
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


        p1 = np.array([cab_coords['xmax'], cab_coords['zmin']])
        p2 = np.array([cab_coords['xmax'], cab_coords['zmax']])
        side_length = np.linalg.norm(p2 - p1)
        other_side_length = np.linalg.norm(p2 - np.array([cab_coords['xmin'], cab_coords['zmax']]))

        # Select point on that side uniformly
        point = p1 + (p2 - p1) * np.random.uniform(0,1)

        # Sample xz distance from that point
        dist_from_cab_range = [0.3, .7] # horizontal distance from cabinet in meters
        dist_from_cab = np.random.uniform(dist_from_cab_range[0], dist_from_cab_range[1])
        # print(f"horizontal dist: {dist_from_cab:.03f}. Range: {dist_from_cab_range}")
        theta = np.radians(-90)
        rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                               [np.sin(theta), np.cos(theta)]])
        away_from_cab_direction = rot_matrix.dot ( (p2 - p1) / np.linalg.norm(p2 - p1) )
        camera_x, camera_z = point + dist_from_cab * away_from_cab_direction

        # Sample y distance
        height_from_cab_range = [.4, .8] # vertical distance from cabinet countertop in meters
        height_from_cab = np.random.uniform(height_from_cab_range[0], height_from_cab_range[1]) 
        # print(f"vertical dist: {height_from_cab}. Range: {height_from_cab_range}")
        camera_y = cab_coords['ymax'] + height_from_cab

        # Final camera position
        camera_pos = np.array([camera_x, camera_y, camera_z])

        lookat_zmin = max(point[1] - side_length*0.2, cab_coords['zmin'])
        lookat_zmax = min(point[1] + side_length*0.2, cab_coords['zmax'])
        lookat_xmin = point[0] - other_side_length*0.5
        lookat_xmax = point[0] - other_side_length*0.0

        # Sample lookat position
        lookat_pos = np.zeros(3)
        lookat_pos[0] = np.random.uniform(lookat_xmin, lookat_xmax)
        # print(f"X lookat_pos: {lookat_pos[0]}. Range: {[lookat_xmin, lookat_xmax]}")
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

        # Get cabinet top
        cabinet_top_mask = np.isclose(world_frame_depth[..., 1], 0.925, atol=0.005) # in [0.92, .93]
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

        # Convert objID_seg_img to background (0), cabinet (1), top_drawer (2), bottom_drawer (3), objects (4+). 
        object_bids = [v.bid for k, v in sorted(self._obj_id_to_body.items(), key=lambda x:x[0]) 
                       if 'ShapeNet' in k]
        seglabel_id = self.OBJ_LABEL_START
        bid_to_seglabel_mapping = {} # Mapping from bid to segmentation label
        for bid in object_bids:
            bid_to_seglabel_mapping[bid] = seglabel_id
            seglabel_id += 1

        # Cabinet bullet ID
        #   link 43 is sektion cabinet
        #   link 56 is top drawer
        #   link 57 is top drawer handle
        #   link 58 is bottom drawer
        #   link 59 is bottom drawer handle
        link_to_seglabel_mapping = {} # Mapping from link to segmentation label
        # Sektion cabinet
        link_to_seglabel_mapping[43] = 1
        # Top drawer
        link_to_seglabel_mapping[56] = 2
        link_to_seglabel_mapping[57] = 2
        # Bottom drawer
        link_to_seglabel_mapping[58] = 3
        link_to_seglabel_mapping[59] = 3

        # Conversion happens here
        new_seg_img = np.zeros_like(objID_seg_img)
        for bid, seg_label in bid_to_seglabel_mapping.items():
            mask = objID_seg_img == bid
            new_seg_img[mask] = seg_label
        for linkid, seg_label in link_to_seglabel_mapping.items():
            mask = (linkID_seg_img == linkid)
            if seg_label == 1: # cabinet
                if np.count_nonzero(mask) > 0: # if no cabinet in this image, the calling function will throw away the image
                    mask = self.get_cabinet_top_mask(depth_img, mask, camera_pos, lookat_pos, camera_up_vector)
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
        obj_contacts = self.get_contacts(obj_id, 'kitchen', only_closest_contact_per_pair=False)
        
        # if this is nonzero, there is contact with sektion cabinet
        if len(obj_contacts) > 0: 
            contact_list = obj_contacts[(obj_id, 'kitchen')]

            # top drawer
            dt_ind = self.kitchen_stuff['jname_to_ind']['drawer_top']
            drawer_contacts = list(filter(lambda x: x.linkIndexB == dt_ind, contact_list))
            if len(drawer_contacts) > 0: # if this is nonzero, there is contact with top drawer
                lowest_y_contact = min([x.positionOnBInWS[1] for x in drawer_contacts])
                drawer_coords = self.get_object_bbox_coordinates('kitchen', linkIndex=dt_ind)
                if drawer_coords['ymin'] <= lowest_y_contact <= (drawer_coords['ymin']+drawer_coords['ymax'])/2.:
                    other_args['predicates'].append((obj_id, 'drawer_top', 2, 'inside_of'))

            # bottom drawer
            db_ind = self.kitchen_stuff['jname_to_ind']['drawer_bottom']
            drawer_contacts = list(filter(lambda x: x.linkIndexB == db_ind, contact_list))
            if len(drawer_contacts) > 0: # if this is nonzero, there is contact with bottom drawer
                lowest_y_contact = min([x.positionOnBInWS[1] for x in drawer_contacts])
                drawer_coords = self.get_object_bbox_coordinates('kitchen', linkIndex=db_ind)
                if drawer_coords['ymin'] <= lowest_y_contact <= (drawer_coords['ymin']+drawer_coords['ymax'])/2.:
                    other_args['predicates'].append((obj_id, 'drawer_bottom', 3, 'inside_of'))








