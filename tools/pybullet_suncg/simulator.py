import numpy as np
import os, sys
import pybullet as p
import subprocess as sp
import time
import glob
import pandas
import itertools

from collections import namedtuple
from itertools import groupby
from pyquaternion import Quaternion

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


class Simulator:
    """
        This code is based on the public pybullet-based SUNCG simulator from 
            github.com/msavva/pybullet_suncg. 
        I have extended this code to randomly generate a SUNCG room with ShapeNet objects.
        An instance of this class holds one PyBullet scene at a time. This scene is exported
            to a dictionary and the depth maps/labels/images are calculated and saved to disk.
    """
    def __init__(self, mode='direct', bullet_server_binary=None, suncg_data_dir_base=None, 
                 shapenet_data_dir_base=None, params=dict(), verbose=False):
        self._mode = mode
        self._verbose = verbose
        module_dir = os.path.dirname(os.path.abspath(__file__))

        # Initialize some path variables
        if suncg_data_dir_base:
            self._suncg_data_dir_base = suncg_data_dir_base
        else:
            self._suncg_data_dir_base = os.path.join(os.path.expanduser('~'), 'work', 'suncg')
        if shapenet_data_dir_base:
            self._shapenet_data_dir_base = shapenet_data_dir_base
        else:
            self._shapenet_data_dir_base = os.path.join(os.path.expanduser('~'), 'work', 'ShapeNetCore.v2')
        if bullet_server_binary:
            self._bullet_server_binary = bullet_server_binary
        else:
            self._bullet_server_binary = os.path.join(module_dir, '..', 'bullet_shared_memory_server')

        # Load object class mapping. Assues ModelCategoryMapping.csv lives in suncg_data_dir_base
        self.object_class_mapping = pandas.read_csv(suncg_data_dir_base + 'ModelCategoryMapping.csv')

        # Simulation parameters
        self.params = params.copy()

        # Filtered objects
        self._filtered_objects = []

        # Initialize other stuff
        self._obj_id_to_body = {}
        self._bid_to_body = {}
        self._pid = None
        self._bullet_server = None
        self.connect()

        # Miscellaneous
        self.NUM_PIXELS_VISIBLE = 100
        self._mesh_cache = {}

        # Object labels start here
        self.OBJ_LABEL_START = 2

    def connect(self):
        # disconnect and kill existing servers
        if self._pid is not None:
            p.disconnect(physicsClientId=self._pid)
            self._pid = None
        if self._bullet_server:
            print(f'Restarting by killing bullet server pid={self._bullet_server.pid}...')
            self._bullet_server.kill()
            time.sleep(1)  # seems necessary to prevent deadlock on re-connection attempt
            self._bullet_server = None

        # reconnect to appropriate server type
        if self._mode == 'gui':
            self._pid = p.connect(p.GUI)
        elif self._mode == 'direct':
            self._pid = p.connect(p.DIRECT)
        elif self._mode == 'shared_memory':
            print(f'Restarting bullet server process...')
            self._bullet_server = sp.Popen([self._bullet_server_binary])
            time.sleep(1)  # seems necessary to prevent deadlock on connection attempt
            self._pid = p.connect(p.SHARED_MEMORY)
        else:
            raise RuntimeError(f'Unknown simulator server mode={self._mode}')

        # reset and initialize gui if needed
        p.resetSimulation(physicsClientId=self._pid)
        if self._mode == 'gui':
            p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1, physicsClientId=self._pid)
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0, physicsClientId=self._pid)
            p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0, physicsClientId=self._pid)
            p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0, physicsClientId=self._pid)

        # Set the gravity
        p.setGravity(0, -10, 0, physicsClientId=self._pid)

        # Reset, just in case this was called incorrectly
        self.reset()

    def disconnect(self):
        if self._pid is not None:
            p.disconnect(physicsClientId=self._pid)
            self._pid = None
        if self._bullet_server:
            print(f'Disconnecting. Killing bullet server pid={self._bullet_server.pid}...')
            self._bullet_server.kill()
            time.sleep(1)  # seems necessary to prevent deadlock on re-connection attempt
            self._bullet_server = None


    def __del__(self):
        if self._bullet_server:
            print(f'Process terminating. Killing bullet server pid={self._bullet_server.pid}...')
            self._bullet_server.kill()

    def set_gui_rendering(self, room=None):
        if not self._mode == 'gui':
            return False
        center = np.array([0.0, 0.0, 0.0])
        num_obj = 0

        objs_to_avoid = []
        if room is not None: # don't factor these rooms into camera target calculation
            objs_to_avoid = [body.id for body in room.body]
        for obj_id in self._obj_id_to_body.keys():
            if obj_id in objs_to_avoid:
                continue
            pos, _ = self.get_state(obj_id)
            if not np.allclose(pos, [0, 0, 0]):  # try to ignore room object 'identity' transform
                num_obj += 1
                center += pos
        center /= num_obj
        p.resetDebugVisualizerCamera(cameraDistance=5.0,
                                     cameraYaw=45.0,
                                     cameraPitch=-30.0,
                                     cameraTargetPosition=center,
                                     physicsClientId=self._pid)
        # return enabled

    def add_cuboid(self, obj_id, transform, half_extents=None, rgba_color=None, texture_file=''):
        if half_extents is None:
            half_extents = np.random.uniform(0.5,1,[3])
        if rgba_color is None:
            rgba_color = sim_util.random_color()
        half_extents = half_extents * transform.scale
        cid = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        vid = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=rgba_color)
        rot_q = np.roll(transform.rotation.elements, -1)  # w,x,y,z -> x,y,z,w (which pybullet expects)
        bid = p.createMultiBody(baseMass=1,
                                baseCollisionShapeIndex=cid,
                                baseVisualShapeIndex=vid,
                                basePosition=transform.translation,
                                baseOrientation=rot_q,
                               )

        if texture_file != '':
            texture = p.loadTexture(texture_file)
            p.changeVisualShape(bid, -1, textureUniqueId=texture)

        body = Body(id=obj_id, bid=bid, vid=vid, cid=cid, static=False)
        self._obj_id_to_body[obj_id] = body
        self._bid_to_body[bid] = body
        return body

    def add_cylinder(self, obj_id, transform, height=None, radius=None, rgba_color=None, texture_file=''):
        if height is None:
            ratio = np.random.uniform(0.5, 5) # height / radius = ratio
            height = 1
        if radius is None:
            radius = height / ratio
        if rgba_color is None:
            rgba_color = sim_util.random_color()   
        radius = radius * transform.scale[0] # Assume transform.scale is c * [1,1,1]
        height = height * transform.scale[0] 
        cid = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
        vid = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=rgba_color)
        rot_q = np.roll(transform.rotation.elements, -1)  # w,x,y,z -> x,y,z,w (which pybullet expects)
        bid = p.createMultiBody(baseMass=1,
                                baseCollisionShapeIndex=cid,
                                baseVisualShapeIndex=vid,
                                basePosition=transform.translation,
                                baseOrientation=rot_q,
                               )

        if texture_file != '':
            texture = p.loadTexture(texture_file)
            p.changeVisualShape(bid, -1, textureUniqueId=texture)

        body = Body(id=obj_id, bid=bid, vid=vid, cid=cid, static=False)
        self._obj_id_to_body[obj_id] = body
        self._bid_to_body[bid] = body
        return body

    def add_mesh(self, obj_id, obj_file, transform, vis_mesh_file=None, texture_file='', static=False):

        if static:
            cid = p.createCollisionShape(p.GEOM_MESH, fileName=obj_file, meshScale=transform.scale,
                                         flags=p.GEOM_FORCE_CONCAVE_TRIMESH, physicsClientId=self._pid)
        else:
            cid = p.createCollisionShape(p.GEOM_MESH, fileName=obj_file, meshScale=transform.scale,
                                         physicsClientId=self._pid)

        vid = -1
        if vis_mesh_file:
            vid = p.createVisualShape(p.GEOM_MESH, fileName=vis_mesh_file, meshScale=transform.scale,
                                      physicsClientId=self._pid)

        rot_q = np.roll(transform.rotation.elements, -1)  # w,x,y,z -> x,y,z,w (which pybullet expects)
        mass = 0 if static else 1
        bid = p.createMultiBody(baseMass=mass,
                                baseCollisionShapeIndex=cid,
                                baseVisualShapeIndex=vid,
                                basePosition=transform.translation,
                                baseOrientation=rot_q,
                                physicsClientId=self._pid)

        if texture_file != '':
            texture = p.loadTexture(texture_file)
            p.changeVisualShape(bid, -1, textureUniqueId=texture)
        else:
            rgba_color = sim_util.random_color()
            p.changeVisualShape(bid, -1, rgbaColor=rgba_color)

        body = Body(id=obj_id, bid=bid, vid=vid, cid=cid, static=static)
        self._obj_id_to_body[obj_id] = body
        self._bid_to_body[bid] = body
        return body

    def add_box(self, obj_id, half_extents, transform, static=False):
        cid = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents, physicsClientId=self._pid)
        rot_q = np.roll(transform.rotation.elements, -1)  # w,x,y,z -> x,y,z,w (which pybullet expects)
        mass = 0 if static else 1
        bid = p.createMultiBody(baseMass=mass,
                                baseCollisionShapeIndex=cid,
                                basePosition=transform.translation,
                                baseOrientation=rot_q,
                                physicsClientId=self._pid)
        body = Body(id=obj_id, bid=bid, vid=-1, cid=cid, static=static)
        self._obj_id_to_body[obj_id] = body
        self._bid_to_body[bid] = body
        return body

    # House-specific functions
    def add_object(self, node, create_vis_mesh=False, static=False):
        model_id = node.modelId.replace('_mirror', '')
        object_dir = os.path.join(self._suncg_data_dir_base, 'object')
        basename = f'{object_dir}/{model_id}/{model_id}'
        vis_obj_filename = f'{basename}.obj' if create_vis_mesh else None
        col_obj_filename = f'{basename}.vhacd.obj' # if you've used the VHACD algorithm to compute better collision meshes than a convex hull
        if not os.path.exists(col_obj_filename):
            # print('WARNING: collision mesh {col_obj_filename} unavailable, using visual mesh instead.')
            col_obj_filename = f'{basename}.obj'
        return self.add_mesh(obj_id=node.id, obj_file=col_obj_filename, transform=Transform.from_node(node),
                             vis_mesh_file=vis_obj_filename, static=static)

    def add_wall(self, node):
        h = node['height']
        p0 = np.transpose(np.matrix(node['points'][0]))
        p1 = np.transpose(np.matrix(node['points'][1]))
        c = (p0 + p1) * 0.5
        c[1] = h * 0.5
        dp = p1 - p0
        dp_l = np.linalg.norm(dp)
        dp = dp / dp_l
        angle = np.arccos(dp[0])
        rot_q = Quaternion(axis=[0, 1, 0], radians=angle)
        half_extents = np.array([dp_l, h, node['depth']]) * 0.5
        return self.add_box(obj_id=node['id'], half_extents=half_extents,
                            transform=Transform(translation=c, rotation=rot_q), static=True)

    def add_room(self, node, wall=True, floor=True, ceiling=False):
        def add_architecture(n, obj_file, suffix):
            return self.add_mesh(obj_id=n.id + suffix, obj_file=obj_file, transform=Transform(), 
                                 vis_mesh_file=obj_file, static=True)
        room_id = node.modelId
        room_dir = os.path.join(self._suncg_data_dir_base, 'room')
        basename = f'{room_dir}/{node.house_id}/{room_id}'
        body_ids = []
        if wall:
            body_wall = add_architecture(node, f'{basename}w.obj', '')  # treat walls as room (=room.id, no suffix)
            body_ids.append(body_wall)
        if floor:
            body_floor = add_architecture(node, f'{basename}f.obj', 'f')
            body_ids.append(body_floor)
        if ceiling:
            body_ceiling = add_architecture(node, f'{basename}c.obj', 'c')
            body_ids.append(body_ceiling)
        return body_ids


    def add_random_house_room(self, no_walls=False, no_ceil=True, no_floor=False, 
                              use_separate_walls=False, only_architecture=False, static=True):
        """ Select a random room from a random house and load it

            @param house_ids: List of house IDs
            @param valid_room_types: List of valid room types
        """
        room = None
        while room is None:
            house_id = np.random.choice(self.params['house_ids'])
            print('house:', house_id)
            house = suncg_house.House(house_json=f'{self._suncg_data_dir_base}/house/{house_id}/house.json')

            for _room in house.rooms:
                valid_room_type = len(set(_room.roomTypes).intersection(self.params['valid_room_types'])) > 0
                room_xsize = _room.bbox['max'][0] - _room.bbox['min'][0]
                room_ysize = _room.bbox['max'][2] - _room.bbox['min'][2]
                valid_room_size = (room_xsize > self.params['min_xlength']) and \
                                  (room_ysize > self.params['min_ylength'])
                if valid_room_type and valid_room_size:
                    if self._verbose:
                        print(f"Using a {_room.roomTypes}")
                    room = _room
                    break

        # Print size of room
        room_xyz_center = np.array([room_xsize, 0, room_ysize]) / 2
        if self._verbose:
            print(f"Room xsize, zsize: {room_xsize}, {room_ysize}")
            print(f'Room xyz center: {room_xyz_center}')

        # Load the room
        self.add_house_room(house, room, no_walls=no_walls, no_ceil=no_ceil, no_floor=no_floor, 
                            use_separate_walls=use_separate_walls, only_architecture=only_architecture, 
                            static=static)

    def add_house_room(self, house, room, no_walls=False, no_ceil=True, no_floor=False, 
                       use_separate_walls=False, only_architecture=False, static=True):

        # Don't allow rendering. Speeds up the loading of the room
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=self._pid)

        room_node = [node for node in house.nodes if node.id == room.id]
        if len(room_node) < 1:
            raise Exception('Missing Room')
        if only_architecture:
            house.nodes = room_node
        else:
            house.nodes = [node for node in room.nodes]
            house.nodes.append(room_node[0])

        # First, filter out objects
        for node in house.nodes:
            if not node.valid:
                continue
            if node.type == 'Object':
                # Filtering out happens here
                classes = self.get_object_class(node)
                if classes['NYUv2'] in self.params['nyuv2_40_classes_filter_list'] or \
                   classes['Coarse'] in self.params['coarse_grained_classes_filter_list']:
                    if self._verbose:
                        print(f"Filtered a {classes['NYUv2']}, {classes['Coarse']}")
                    self._filtered_objects.append(node) # keep track of this so I can filter stuff that is on top

        # Now, add the meshes
        for node in house.nodes:
            if not node.valid:
                continue

            # Initiliaze the .body attribute
            if not hasattr(node, 'body'):
                node.body = None

            if node.type == 'Object':
                if node in self._filtered_objects:
                    continue
                if self.on_top_of_filtered_object(node):
                    classes = self.get_object_class(node)
                    if self._verbose:
                        print(f"Filtered a {classes['NYUv2']}, {classes['Coarse']} which was on top of a filtered object")
                    continue 
                # If not filtered, add the object to the scene
                node.body = self.add_object(node, create_vis_mesh=True, static=static)

            if node.type == 'Room':
                ceil = False if no_ceil else not (hasattr(node, 'hideCeiling') and node.hideCeiling == 1)
                wall = False if (no_walls or use_separate_walls) else not (hasattr(node, 'hideWalls') and node.hideWalls == 1)
                floor = False if no_floor else not (hasattr(node, 'hideFloor') and node.hideFloor == 1)
                node.body = self.add_room(node, wall=wall, floor=floor, ceiling=ceil)

            if node.type == 'Box':
                half_widths = list(map(lambda x: 0.5 * x, node.dimensions))
                node.body = self.add_box(obj_id=node.id, half_extents=half_widths, transform=Transform.from_node(node),
                                         static=static)

        if use_separate_walls and not no_walls:
            for wall in house.walls:
                wall['body'] = self.add_wall(wall)

        # Move room back to origin
        room_bbox = room.bbox
        for obj_id in self._obj_id_to_body.keys():
            pos, rot = self.get_state(obj_id)
            new_pos = list(np.array(pos) - np.array(room_bbox['min']))
            self.set_state(obj_id, new_pos, rot)
        self.set_gui_rendering(room=room)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self._pid)

        self.loaded_room = room

    def on_top_of_filtered_object(self, node):
        """ Check if this object is on top of a filtered object
        """
        on_top = False
        for obj_node in self._filtered_objects:
            inside_xz = node.bbox['min'][0] >= obj_node.bbox['min'][0] and \
                        node.bbox['min'][2] >= obj_node.bbox['min'][2] and \
                        node.bbox['max'][0] <= obj_node.bbox['max'][0] and \
                        node.bbox['max'][2] <= obj_node.bbox['max'][2]
            higher_y = node.bbox['min'][1] > (obj_node.bbox['min'][1] + obj_node.bbox['max'][1])/2
            if inside_xz and higher_y:
                on_top = True
                break
        return on_top

    def get_object_class(self, node):
        """ Get class w.r.t. NYUv2 mappings and coarse grained mappings (provided ny the SUNCG dataset)
        """
        mID = node.modelId
        if '_mirror' in mID: # weird corner case of mirrored objects
            mID = mID.split('_mirror')[0]
        nyuv2_40class = self.object_class_mapping[self.object_class_mapping['model_id'] == mID]['nyuv2_40class'].item()
        coarse_grained_class = self.object_class_mapping[self.object_class_mapping['model_id'] == mID]['coarse_grained_class'].item()
        return {'NYUv2' : nyuv2_40class,
                'Coarse' : coarse_grained_class}


    def remove(self, obj_id):
        body = self._obj_id_to_body[obj_id]
        p.removeBody(bodyUniqueId=body.bid, physicsClientId=self._pid)
        del self._obj_id_to_body[obj_id]
        del self._bid_to_body[body.bid]

    def remove_shapenet_obj(self, obj_id):
        if obj_id in self.shapenet_obj_stuff['obj_ids']:
            ind = self.shapenet_obj_stuff['obj_ids'].index(obj_id)
            self.shapenet_obj_stuff['obj_ids'].pop(ind)
            self.shapenet_obj_stuff['obj_mesh_filenames'].pop(ind)
            self.shapenet_obj_stuff['obj_scale_factors'].pop(ind)

    def set_state(self, obj_id, position, rotation_q):
        body = self._obj_id_to_body[obj_id]
        rot_q = np.roll(rotation_q.elements, -1)  # w,x,y,z -> x,y,z,w (which pybullet expects)
        p.resetBasePositionAndOrientation(bodyUniqueId=body.bid, posObj=position, ornObj=rot_q,
                                          physicsClientId=self._pid)

    def get_state(self, obj_id):
        body = self._obj_id_to_body[obj_id]
        pos, q = p.getBasePositionAndOrientation(bodyUniqueId=body.bid, physicsClientId=self._pid)
        rotation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        return pos, rotation

    def step(self):
        p.stepSimulation(physicsClientId=self._pid)

    def reset(self):
        p.resetSimulation(physicsClientId=self._pid)
        self._obj_id_to_body = {}
        self._bid_to_body = {}
        self._filtered_objects = []
        self._mesh_cache = {}
        p.setGravity(0, -10, 0, physicsClientId=self._pid)

    def get_closest_point(self, obj_id_a, obj_id_b, max_distance=np.inf):
        """
        Return record with distance between closest points between pair of nodes if within max_distance or None.
        """
        bid_a = self._obj_id_to_body[obj_id_a].bid
        bid_b = self._obj_id_to_body[obj_id_b].bid
        cps = p.getClosestPoints(bodyA=bid_a, bodyB=bid_b, distance=max_distance, physicsClientId=self._pid)
        cp = None
        if len(cps) > 0:
            closest_points = self._convert_contacts(cps)
            cp = min(closest_points, key=lambda x: x.distance)
        del cps  # NOTE force garbage collection of pybullet objects
        return cp

    def get_contacts(self, obj_id_a=None, obj_id_b=None, only_closest_contact_per_pair=True,
                     include_collision_with_static=True):
        """
        Return all current contacts. When include_collision_with_statics is true, include contacts with static bodies
        """
        bid_a = self._obj_id_to_body[obj_id_a].bid if obj_id_a else -1
        bid_b = self._obj_id_to_body[obj_id_b].bid if obj_id_b else -1
        cs = p.getContactPoints(bodyA=bid_a, bodyB=bid_b, physicsClientId=self._pid)
        contacts = self._convert_contacts(cs)
        del cs  # NOTE force garbage collection of pybullet objects

        if not include_collision_with_static:
            def not_contact_with_static(c):
                static_a = self._obj_id_to_body[c.idA].static
                static_b = self._obj_id_to_body[c.idB].static
                return not static_a and not static_b
            contacts = filter(not_contact_with_static, contacts)
            # print(f'#all_contacts={len(all_contacts)} to #non_static_contacts={len(non_static_contacts)}')

        if only_closest_contact_per_pair:
            def bid_pair_key(x):
                return str(x.idA) + '_' + str(x.idB)
            contacts = sorted(contacts, key=bid_pair_key)
            min_dist_contact_by_pair = {}
            for k, g in groupby(contacts, key=bid_pair_key):
                min_dist_contact = min(g, key=lambda x: x.distance)
                min_dist_contact_by_pair[k] = min_dist_contact
            contacts = min_dist_contact_by_pair.values()

        # convert into dictionary of form (id_a, id_b) -> Contact
        contacts_dict = {}
        for c in contacts:
            key = (c.idA, c.idB)
            if key in contacts_dict:
                contacts_dict[key].append(c)
            else:
                contacts_dict[key] = [c]

        return contacts_dict

    def _convert_contacts(self, contacts):
        out = []
        for c in contacts:
            bid_a = c[1]
            bid_b = c[2]
            if bid_a not in self._bid_to_body or bid_b not in self._bid_to_body:
                continue
            id_a = self._bid_to_body[bid_a].id
            id_b = self._bid_to_body[bid_b].id
            o = Contact(flags=c[0], idA=id_a, idB=id_b, linkIndexA=c[3], linkIndexB=c[4],
                        positionOnAInWS=c[5], positionOnBInWS=c[6], contactNormalOnBInWS=c[7],
                        distance=c[8], normalForce=c[9])
            out.append(o)
        return out

    def ray_test(self, from_pos, to_pos):
        hit = p.rayTest(rayFromPosition=from_pos, rayToPosition=to_pos, physicsClientId=self._pid)
        intersection = Intersection._make(*hit)
        del hit  # NOTE force garbage collection of pybullet objects
        if intersection.id >= 0:  # if intersection, replace bid with id
            intersection = intersection._replace(id=self._bid_to_body[intersection.id].id)
        return intersection




    ########## FUNCTIONS FOR LOADING SHAPENET MODELS INTO SIMULATION ##########



    ##### UTILITIES #####
    def get_object_bbox_coordinates(self, obj_id, linkIndex=-1):
        """ Return min/max coordinates of an encapsulating bounding box in x,y,z dims
        
            @param obj_id: ID of object
            @return: an np.array: [ [xmin, ymin, zmin],
                                    [xmax, ymax, zmax] ]
        """

        obj_min, obj_max = p.getAABB(self._obj_id_to_body[obj_id].bid, linkIndex=linkIndex, physicsClientId=self._pid)
          
        return {'xmin' : obj_min[0],
                'ymin' : obj_min[1],
                'zmin' : obj_min[2],
                'xmax' : obj_max[0],
                'ymax' : obj_max[1],
                'zmax' : obj_max[2],
                'xsize' : obj_max[0] - obj_min[0],
                'ysize' : obj_max[1] - obj_min[1],
                'zsize' : obj_max[2] - obj_min[2]
               }

    def get_tight_cf_object_bbox_coords(self, obj_id, view_matrix):
        """ PyBullet's p.getAABB() method is a bit loose. Get tighter camera-frame axis-aligned 
            coordinates by consulting original vertices
            MUST be looking at ShapeNet objects only

            @param obj_id: a string
            @param view_matrix: a [4 x 4] numpy array that represents camera extrinsics 
                                (also called view matrix in OpenGL)
        """

        # Load the mesh. Cache them, as loading the mesh is slow
        ind = self.shapenet_obj_stuff['obj_ids'].index(obj_id)
        if obj_id in self._mesh_cache:
            mesh = self._mesh_cache[obj_id]
        else:
            meshfile = self.shapenet_obj_stuff['obj_mesh_filenames'][ind]
            if meshfile.startswith('{'): # randomly generated cuboid/cylinder
                obj_dict = eval(meshfile)
                obj_type = obj_dict['obj_type']
                all_corners_mat = sim_util.all_corners_matrix()
                if obj_type == 'cub':
                    v = np.array(obj_dict['half_extents'])
                elif obj_type == 'cyl':
                    v = np.array([obj_dict['radius'], obj_dict['radius'], obj_dict['height']/2])
                mesh = {'vertices' : all_corners_mat * v}
            else: # normal mesh file from ShapeNet object
                mesh = sim_util.load_mesh(meshfile)
            self._mesh_cache[obj_id] = mesh

        # Scale the vertices 
        vertices = np.array(mesh['vertices']) * self.shapenet_obj_stuff['obj_scale_factors'][ind]

        # Transform vertices to world frame (using homogeneous coordinates)
        vertices = np.concatenate([vertices, np.ones((vertices.shape[0],1))], axis=1) # Shape: [num_vertices x 4]
        temp = self.get_state(obj_id)
        table_pos = np.array(temp[0])
        table_quaternion = np.roll(temp[1].elements, -1)  # w,x,y,z -> x,y,z,w (which pybullet expects)
        table_orn = np.array(p.getMatrixFromQuaternion(table_quaternion)).reshape(3,3)
        transform_matrix = np.concatenate([table_orn, np.expand_dims(table_pos, axis=1)], axis=1)
        transform_matrix = np.concatenate([transform_matrix, np.array([[0,0,0,1]])], axis=0)
        vertices = transform_matrix.dot(vertices.T)
        vertices = vertices.T # Shape: [num_vertices x 4]

        # Transform vertices to camera frame 
        vertices = view_matrix.dot(vertices.T).T[..., :3] # Shape: [num_vertices x 3]
        vertices[:,2] = -1 * vertices[:,2] # # negate z to get the left-hand system, z-axis pointing forward

        # Get corners
        corners = np.array([[vertices[:,0].min(), vertices[:,1].min(), vertices[:,2].min()],
                            [vertices[:,0].max(), vertices[:,1].min(), vertices[:,2].min()],
                            [vertices[:,0].min(), vertices[:,1].max(), vertices[:,2].min()],
                            [vertices[:,0].max(), vertices[:,1].max(), vertices[:,2].min()],
                            [vertices[:,0].min(), vertices[:,1].min(), vertices[:,2].max()],
                            [vertices[:,0].max(), vertices[:,1].min(), vertices[:,2].max()],
                            [vertices[:,0].min(), vertices[:,1].max(), vertices[:,2].max()],
                            [vertices[:,0].max(), vertices[:,1].max(), vertices[:,2].max()],
                           ], dtype=np.float32).T

        # Put it into coords format
        tight_bbox_coords = {
            'xmin' : corners[0,:].min(),
            'ymin' : corners[1,:].min(),
            'zmin' : corners[2,:].min(),
            'xmax' : corners[0,:].max(),
            'ymax' : corners[1,:].max(),
            'zmax' : corners[2,:].max(),
        }
        tight_bbox_coords['xsize'] = tight_bbox_coords['xmax'] - tight_bbox_coords['xmin']
        tight_bbox_coords['ysize'] = tight_bbox_coords['ymax'] - tight_bbox_coords['ymin']
        tight_bbox_coords['zsize'] = tight_bbox_coords['zmax'] - tight_bbox_coords['zmin']

        return tight_bbox_coords

    def simulate(self, timesteps):
        """ Simulate dynamics. Don't allow rendering, which speeds up the process
        """
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=self._pid)
        for i in range(timesteps): 
            p.stepSimulation(physicsClientId=self._pid)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self._pid)

    def load_house_room(self, scene_description):
        """ Takes a scene descrption dictionary (as exported by self.export_scene_to_dictionary())
            and loads the house only

            NOTE: This MUST be called before load_cabinet() or load_objects()

            @param scene_description: a scene description dictionary (schema can be found in 
                                                          self.export_scene_to_dictionary())
        """

        house = suncg_house.House(house_json=f"{self._suncg_data_dir_base}/house/{scene_description['room']['house_id']}/house.json")
        room = [r for r in house.rooms if r.id == scene_description['room']['room_id']][0]
        self.add_house_room(house, room, no_walls=False, no_ceil=True, no_floor=False, use_separate_walls=False,
                           only_architecture=False, static=True)

    def load_objects(self, scene_description):
        """ Takes a scene descrption dictionary (as exported by self.export_scene_to_dictionary())
            and loads the objects only

            @param scene_description: a scene description dictionary (schema can be found in 
                                                          self.export_scene_to_dictionary())
        """                
        object_descriptions = scene_description['object_descriptions']
                                  
        self.shapenet_obj_stuff = {
            'obj_ids' : [],
            'obj_mesh_filenames' : [],
            'obj_scale_factors' : [],
            'obj_texture_filenames' : [],
        }

        for i, obj in enumerate(object_descriptions):
            print('%d:%d' % (i, len(object_descriptions)), obj)
            if obj['mesh_filename'].startswith('{'):
                self.load_cub_cyl(obj, f'ShapeNet_obj_{i}')
            else:
                self.load_sn_obj(obj, f'ShapeNet_obj_{i}')

    def load_cub_cyl(self, object_desc, potential_obj_id):
        """ Load cuboid/cylinder. Helper function for self.load_objects()
        """
        if 'obj_id' not in object_desc:
            obj_id = potential_obj_id
            object_desc['obj_id'] = obj_id
        else:
            obj_id = object_desc['obj_id']

        obj_transform = Transform(translation=np.array(object_desc['position']),
                                  rotation=Quaternion(w=object_desc['orientation'][0],
                                                      x=object_desc['orientation'][1],
                                                      y=object_desc['orientation'][2],
                                                      z=object_desc['orientation'][3]),
                                  scale=np.ones(3) * object_desc['scale']
                                 )

        obj_dict = eval(object_desc['mesh_filename'])
        obj_type = obj_dict['obj_type']
        if obj_type == 'cub':
            half_extents = np.array(obj_dict['half_extents'])
            rgba_color = np.array(obj_dict['rgba_color'])
            self.add_cuboid(obj_id, obj_transform, 
                            half_extents=half_extents, rgba_color=rgba_color, texture_file=object_desc['texture_filename'])
        elif obj_type == 'cyl':
            radius = obj_dict['radius']
            height = obj_dict['height']
            rgba_color = np.array(obj_dict['rgba_color'])
            self.add_cylinder(obj_id, obj_transform, radius=radius, 
                              height=height, rgba_color=rgba_color, texture_file=object_desc['texture_filename'])

        self.shapenet_obj_stuff['obj_ids'].append(obj_id)
        self.shapenet_obj_stuff['obj_mesh_filenames'].append(object_desc['mesh_filename'])
        self.shapenet_obj_stuff['obj_scale_factors'].append(object_desc['scale'])
        self.shapenet_obj_stuff['obj_texture_filenames'].append(object_desc['texture_filename'])


    def load_sn_obj(self, object_desc, potential_obj_id):
        """ Load a single object. Helper function for self.load_objects()
        """
        if not object_desc['mesh_filename'].startswith(self._shapenet_data_dir_base):
            object_desc['mesh_filename'] = self._shapenet_data_dir_base + object_desc['mesh_filename']
        if 'obj_id' not in object_desc:
            obj_id = potential_obj_id
            object_desc['obj_id'] = obj_id
        else:
            obj_id = object_desc['obj_id']
        obj_transform = Transform(translation=np.array(object_desc['position']),
                                  rotation=Quaternion(w=object_desc['orientation'][0],
                                                      x=object_desc['orientation'][1],
                                                      y=object_desc['orientation'][2],
                                                      z=object_desc['orientation'][3]),
                                  scale=np.ones(3) * object_desc['scale']
                                 )
        self.add_mesh(obj_id,
                     object_desc['mesh_filename'],
                     obj_transform,
                     object_desc['mesh_filename'],
                     object_desc['texture_filename']
                    )

        self.shapenet_obj_stuff['obj_ids'].append(obj_id)
        self.shapenet_obj_stuff['obj_mesh_filenames'].append(object_desc['mesh_filename'])
        self.shapenet_obj_stuff['obj_scale_factors'].append(object_desc['scale'])
        self.shapenet_obj_stuff['obj_texture_filenames'].append(object_desc['texture_filename'])



    ##### CODE TO RENDER SCENES #####

    def sample_room_view(self):
        """ Sample a view inside room
        """

        walls_coords = self.get_object_bbox_coordinates(self.loaded_room.body[0].id)

        # Sample anywhere inside room
        camera_pos = np.random.uniform(0,1, size=[3])
        camera_pos[0] = np.random.uniform(walls_coords['xmin'] + .25 * walls_coords['xsize'], 
                                          walls_coords['xmax'] - .25 * walls_coords['xsize'])
        camera_pos[1] = np.random.uniform(max(walls_coords['ymin'] + .25 * walls_coords['ysize'], 1.0), # minimum y height is 1.0 meters
                                          walls_coords['ymax'] - .25 * walls_coords['ysize'])
        camera_pos[2] = np.random.uniform(walls_coords['zmin'] + .25 * walls_coords['zsize'], 
                                          walls_coords['zmax'] - .25 * walls_coords['zsize'])

        # Sample a "lookat" position. Take the vector [0,0,1], rotate it on xz plane (about y-axis), then on yz plane (about x-axis)
        xz_plane_theta = np.random.uniform(0, 2*np.pi)   # horizontal rotation. rotate on xz plane, about y-axis
        yz_plane_theta = np.random.uniform(0, np.pi / 6) # up-down rotation. rotate on yz plane, about x-axis

        # Compose the two extrinsic rotations
        quat = p.multiplyTransforms(np.array([0,0,0]),
                                    p.getQuaternionFromEuler(([0,xz_plane_theta,0])), 
                                    np.array([0,0,0]),
                                    p.getQuaternionFromEuler(([yz_plane_theta,0,0]))
                                   )[1]

        direction = np.array([0,0,1])
        direction = np.asarray(p.getMatrixFromQuaternion(quat)).reshape(3,3).dot(direction)
        lookat_pos = camera_pos + direction

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

        return self.get_camera_images(camera_pos, lookat_pos)


    def sample_camera_up_vector(self, camera_pos, lookat_pos):

        # To do this, I first generate the camera view with [0,1,0] as the up-vector, then sample a rotation from the camera x-y axis and apply it
        # truncated normal sampling. clip it to 2*sigma range, meaning ~5% of samples are maximally rotated
        theta = np.random.normal(0, self.params['max_camera_rotation'] / 2, size=[1])
        theta = theta.clip(-self.params['max_camera_rotation'], self.params['max_camera_rotation'])[0]
        my_rot_mat = np.array([[np.cos(theta), -np.sin(theta), 0],
                               [np.sin(theta), np.cos(theta), 0],
                               [0, 0, 1]])
        y_axis = np.array([0,1,0])
        camera_up_vector = my_rot_mat.dot(y_axis) # this is in camera coordinate frame. pybullet needs this in world coordinate frame
        camera_rotation_matrix = np.asarray(p.computeViewMatrix(camera_pos, lookat_pos, y_axis)).reshape(4,4, order='F')[:3,:3].T # Note that transpose is the inverse since it's orthogonal. this is camera rotation matrix
        camera_up_vector = camera_rotation_matrix.dot(camera_up_vector) # camera up vector in world coordinate frame

        return camera_up_vector




    ##### CODE TO COMPUTE PREDICATES #####

    def compute_predicates(self, camera_pos, lookat_pos, camera_up_vector, 
                           bid_to_seglabel_mapping, seg_img,
                           pairwise_predicate_methods,
                           single_arg_predicate_methods):
        """ Compute a set of predicates w.r.t. camera transformation
    
            @param camera_pos: 
            @param lookat_pos: 
            @param camera_up_vector: 
            @param bid_to_seglabel_mapping: a Python dictionary mapping PyBullet ID's to segmentation labels
            @param seg_img: 
            @param pairwise_predicate_methods: 
            @param single_arg_predicate_methods: 
        """

        # Helper functions
        def camera_frame_bbox_coords(obj_id):
            cf_coords = self.get_tight_cf_object_bbox_coords(obj_id, predicate_vm)
            cf_bbox_corners = sim_util.get_array_of_corners(cf_coords)
            cf_center = sim_util.get_aligned_bbox3D_center(cf_coords)
            cf_bbox_corners = np.concatenate([cf_bbox_corners, np.expand_dims(cf_center, axis=1)], axis=1)
            return cf_bbox_corners

        def is_visibile(obj_id):
            label = bid_to_seglabel_mapping[self._obj_id_to_body[obj_id].bid]
            visible = label in seg_img
            visible = visible and np.count_nonzero(seg_img == label) >= self.NUM_PIXELS_VISIBLE
            return visible

        # special view matrix for predicates where z-axis points horizontally instead of down at cabinet
        straight_lookat_pos = lookat_pos.copy()
        straight_lookat_pos[1] = camera_pos[1]
        predicate_vm = p.computeViewMatrix(camera_pos, straight_lookat_pos, [0,1,0]) 
        predicate_vm = np.array(predicate_vm).reshape(4,4, order='F')

        # Predicates list
        predicates = []
        other_args = { # Used for predicate functions
            'predicates' : predicates,
            'camera_pos' : camera_pos,
            'lookat_pos' : lookat_pos,
            'camera_up_vector' : camera_up_vector,
        }

        for obj_id in self.shapenet_obj_stuff['obj_ids']:

            cf_bbox_corners = camera_frame_bbox_coords(obj_id)
            visible = is_visibile(obj_id)
            for method in single_arg_predicate_methods:
                if visible:
                    method({'obj_id' : obj_id, 'cf_bbox_corners' : cf_bbox_corners}, 
                           other_args)

        for pair in itertools.combinations(self.shapenet_obj_stuff['obj_ids'], 2):

            # Get two objects
            obj1_id = pair[0]
            obj2_id = pair[1]

            # Get bbox coordinates
            cf_o1_bbox_corners = camera_frame_bbox_coords(obj1_id)
            cf_o2_bbox_corners = camera_frame_bbox_coords(obj2_id)

            # Get visibility
            o1_visible = is_visibile(obj1_id)
            o2_visible = is_visibile(obj2_id)

            ### Compute the predicates ###
            o1_predicate_args = {
                'obj_id' : obj1_id,
                'cf_bbox_corners' : cf_o1_bbox_corners, 
            }
            o2_predicate_args = {
                'obj_id' : obj2_id,
                'cf_bbox_corners' : cf_o2_bbox_corners,
            }
            if o1_visible and o2_visible:
                for method in pairwise_predicate_methods:
                    method(o1_predicate_args, o2_predicate_args, other_args)
                    method(o2_predicate_args, o1_predicate_args, other_args)


        # Each predicate shall be returned as a tuple: (o1_id, o1_label_id, o2_id, o2_label_id, rel_string)
        for ind, predicate in enumerate(predicates):
            # Note: predicate is a tuple

            i = 0
            while i < len(predicate):
                if predicate[i] in self._obj_id_to_body and \
                   self._obj_id_to_body[predicate[i]].bid in bid_to_seglabel_mapping:
                    label_id = bid_to_seglabel_mapping[self._obj_id_to_body[predicate[i]].bid]
                    predicate = predicate[:i+1] + (label_id,) + predicate[i+1:]
                    i += 1
                i += 1

            predicates[ind] = predicate

        return predicates

    def compute_left_right_predicates(self, o1_predicate_args, o2_predicate_args, other_args):
        """ Compute left-right predicates.
            Use camera frame coordinates.

            Relation rules:
                1) o1 center MUST be in half-space defined by o2 UPPER corner and theta (xz plane)
                2) o1 center MUST be in half-space defined by o2 LOWER corner and theta (xz plane)
                3) do same as 1) for xy
                4) do same as 2) for xy
                5) o1 center MUST be to left of all o2 corners
                6) All o1 corners MUST be to the left of o2 center
        """

        def left_of(cf_o1_bbox_corners, cf_o2_bbox_corners):

            # Check xz
            o1_xz_center = cf_o1_bbox_corners[[0,2], 8] # [x,z]

            # Get camera-frame axis-aligned bbox corners for o2. Only for x-z plane, and two left-most corners. 
            o2_upper_corner = np.array([cf_o2_bbox_corners[0,:8].min(), cf_o2_bbox_corners[2,:8].max()]) # [x,z]
            o2_lower_corner = np.array([cf_o2_bbox_corners[0,:8].min(), cf_o2_bbox_corners[2,:8].min()]) # [x,z]

            # Upper half-space defined by p'n + d = 0
            upper_normal = sim_util.rotate_2d(np.array([-1,0]), np.pi/2. - self.params['theta_predicte_lr_fb_ab'])
            upper_d = -1 * o2_upper_corner.dot(upper_normal)
            first_rule = o1_xz_center.dot(upper_normal) + upper_d >= 0

            # Lower half-space defined by p'n + d = 0
            lower_normal = sim_util.rotate_2d(np.array([0,1]), self.params['theta_predicte_lr_fb_ab'])
            lower_d = -1 * o2_lower_corner.dot(lower_normal)
            second_rule = o1_xz_center.dot(lower_normal) + lower_d >= 0

            xz_works = first_rule and second_rule

            # Check xy
            o1_xy_center = cf_o1_bbox_corners[[0,1], 8] # [x,y]

            # Get camera-frame axis-aligned bbox corners for o2. Only for x-y plane, and two left-most corners. 
            o2_upper_corner = np.array([cf_o2_bbox_corners[0,:8].min(), cf_o2_bbox_corners[1,:8].max()]) # [x,y]
            o2_lower_corner = np.array([cf_o2_bbox_corners[0,:8].min(), cf_o2_bbox_corners[1,:8].min()]) # [x,y]

            # Upper half-space defined by p'n + d = 0
            upper_normal = sim_util.rotate_2d(np.array([-1,0]), np.pi/2. - self.params['theta_predicte_lr_fb_ab'])
            upper_d = -1 * o2_upper_corner.dot(upper_normal)
            third_rule = o1_xy_center.dot(upper_normal) + upper_d >= 0

            # Lower half-space defined by p'n + d = 0
            lower_normal = sim_util.rotate_2d(np.array([0,1]), self.params['theta_predicte_lr_fb_ab'])
            lower_d = -1 * o2_lower_corner.dot(lower_normal)
            fourth_rule = o1_xy_center.dot(lower_normal) + lower_d >= 0

            xy_works = third_rule and fourth_rule

            # All corners check
            fifth_rule = np.all(o1_xz_center[0] <= cf_o2_bbox_corners[0,:8].min())

            # o1 right corners check
            sixth_rule = np.all(cf_o1_bbox_corners[0, :8].max() <= cf_o2_bbox_corners[0,8])

            return xz_works and xy_works and fifth_rule and sixth_rule

        obj1_id = o1_predicate_args['obj_id']
        obj2_id = o2_predicate_args['obj_id']

        # For symmetry, check if o1 is left of o2, and if o2 is right of o1
        cf_o1_bbox_corners = o1_predicate_args['cf_bbox_corners'].copy()
        cf_o2_bbox_corners = o2_predicate_args['cf_bbox_corners'].copy()
        o1_left_of_o2 = left_of(cf_o1_bbox_corners, cf_o2_bbox_corners)

        cf_o1_bbox_corners[0,:] = cf_o1_bbox_corners[0,:] * -1
        cf_o2_bbox_corners[0,:] = cf_o2_bbox_corners[0,:] * -1
        o2_right_of_o1 = left_of(cf_o2_bbox_corners, cf_o1_bbox_corners)

        if o1_left_of_o2 or o2_right_of_o1:
           other_args['predicates'].append((obj1_id, obj2_id, 'left'))
           other_args['predicates'].append((obj2_id, obj1_id, 'right'))            


    def compute_front_behind_predicates(self, o1_predicate_args, o2_predicate_args, other_args):
        """ Compute front-behind predicates.
            Use camera frame coordinates.

            Relation rules:
                1) o1 center MUST be in half-space defined by o2 LEFT corner and theta (xz plane)
                2) o1 center MUST be in half-space defined by o2 RIGHT corner and theta (xz plane)
                3) do same as 1) for yz
                4) do same as 2) for yz
                5) o1 center MUST be behind all o2 corners
                6) All o1 corners MUST be behind o2 center
        """

        def behind(cf_o1_bbox_corners, cf_o2_bbox_corners):

            # Check xz
            o1_xz_center = cf_o1_bbox_corners[[0,2], 8] # [x,z]

            # Get camera-frame axis-aligned bbox corners for o2. Only for x-z plane, and two left-most corners. 
            o2_left_corner = np.array([cf_o2_bbox_corners[0,:8].min(), cf_o2_bbox_corners[2,:8].max()]) # [x,z]
            o2_right_corner = np.array([cf_o2_bbox_corners[0,:8].max(), cf_o2_bbox_corners[2,:8].max()]) # [x,z]

            # Left half-space defined by p'n + d = 0
            left_normal = sim_util.rotate_2d(np.array([1,0]), self.params['theta_predicte_lr_fb_ab'])
            left_d = -1 * o2_left_corner.dot(left_normal)
            first_rule = o1_xz_center.dot(left_normal) + left_d >= 0

            # Right half-space defined by p'n + d = 0
            right_normal = sim_util.rotate_2d(np.array([0,1]), np.pi/2. - self.params['theta_predicte_lr_fb_ab'])
            right_d = -1 * o2_right_corner.dot(right_normal)
            second_rule = o1_xz_center.dot(right_normal) + right_d >= 0

            xz_works = first_rule and second_rule

            # Check yz
            o1_yz_center = cf_o1_bbox_corners[[1,2], 8] # [y,z]

            # Get camera-frame axis-aligned bbox corners for o2. Only for x-z plane, and two left-most corners. 
            o2_left_corner = np.array([cf_o2_bbox_corners[1,:8].min(), cf_o2_bbox_corners[2,:8].max()]) # [y,z]
            o2_right_corner = np.array([cf_o2_bbox_corners[1,:8].max(), cf_o2_bbox_corners[2,:8].max()]) # [y,z]

            # Left half-space defined by p'n + d = 0
            left_normal = sim_util.rotate_2d(np.array([1,0]), self.params['theta_predicte_lr_fb_ab'])
            left_d = -1 * o2_left_corner.dot(left_normal)
            third_rule = o1_yz_center.dot(left_normal) + left_d >= 0

            # Right half-space defined by p'n + d = 0
            right_normal = sim_util.rotate_2d(np.array([0,1]), np.pi/2. - self.params['theta_predicte_lr_fb_ab'])
            right_d = -1 * o2_right_corner.dot(right_normal)
            fourth_rule = o1_yz_center.dot(right_normal) + right_d >= 0

            yz_works = third_rule and fourth_rule

            # All corners check
            fifth_rule = np.all(o1_xz_center[1] >= cf_o2_bbox_corners[2,:8].max())

            # o1 near corners check
            sixth_rule = np.all(cf_o1_bbox_corners[2, :8].min() >= cf_o2_bbox_corners[2,8])

            return xz_works and yz_works and fifth_rule and sixth_rule

        obj1_id = o1_predicate_args['obj_id']
        obj2_id = o2_predicate_args['obj_id']

        # For symmetry, check if o1 is behind of o2, and if o2 is in front of o1
        cf_o1_bbox_corners = o1_predicate_args['cf_bbox_corners'].copy()
        cf_o2_bbox_corners = o2_predicate_args['cf_bbox_corners'].copy()
        o1_behind_o2 = behind(cf_o1_bbox_corners, cf_o2_bbox_corners)

        cf_o1_bbox_corners[2,:] = cf_o1_bbox_corners[2,:] * -1
        cf_o2_bbox_corners[2,:] = cf_o2_bbox_corners[2,:] * -1
        o2_in_front_of_o1 = behind(cf_o2_bbox_corners, cf_o1_bbox_corners)        

        if o1_behind_o2 or o2_in_front_of_o1:
            other_args['predicates'].append((obj1_id, obj2_id, 'behind'))
            other_args['predicates'].append((obj2_id, obj1_id, 'front'))

    def compute_above_below_predicates(self, o1_predicate_args, o2_predicate_args, other_args):
        """ Compute above-below predicates.
            Use camera frame coordinates.

            Relation rules:
                1) o1 center MUST be in half-space defined by o2 LEFT corner and theta (xy plane)
                2) o1 center MUST be in half-space defined by o2 RIGHT corner and theta (xy plane)
                3) do same as 1) for zy
                4) do same as 2) for zy
                5) o1 has overlap with o2 in xz plane
                6) o1 center MUST be above all o2 corners
                7) All o1 corners MUST be above o2 center

                rule = ((1 & 2 & 3 & 4) || 5) & 6 & 7
        """

        def above(cf_o1_bbox_corners, cf_o2_bbox_corners):

            # Check xy
            o1_xy_center = cf_o1_bbox_corners[[0,1], 8] # [x,y]

            # Get camera-frame axis-aligned bbox corners for o2
            o2_left_corner = np.array([cf_o2_bbox_corners[0,:8].min(), cf_o2_bbox_corners[1,:8].max()]) # [x,y]
            o2_right_corner = np.array([cf_o2_bbox_corners[0,:8].max(), cf_o2_bbox_corners[1,:8].max()]) # [x,y]

            # Left half-space defined by p'n + d = 0
            left_normal = sim_util.rotate_2d(np.array([1,0]), self.params['theta_predicte_lr_fb_ab'])
            left_d = -1 * o2_left_corner.dot(left_normal)
            first_rule = o1_xy_center.dot(left_normal) + left_d >= 0

            # Right half-space defined by p'n + d = 0
            right_normal = sim_util.rotate_2d(np.array([0,1]), np.pi/2. - self.params['theta_predicte_lr_fb_ab'])
            right_d = -1 * o2_right_corner.dot(right_normal)
            second_rule = o1_xy_center.dot(right_normal) + right_d >= 0

            xy_works = first_rule and second_rule

            # Check zy
            o1_zy_center = cf_o1_bbox_corners[[2,1], 8] # [z,y]

            # Get camera-frame axis-aligned bbox corners for o2 
            o2_left_corner = np.array([cf_o2_bbox_corners[2,:8].min(), cf_o2_bbox_corners[1,:8].max()]) # [z,y]
            o2_right_corner = np.array([cf_o2_bbox_corners[2,:8].max(), cf_o2_bbox_corners[1,:8].max()]) # [z,y]

            # Left half-space defined by p'n + d = 0
            left_normal = sim_util.rotate_2d(np.array([1,0]), self.params['theta_predicte_lr_fb_ab'])
            left_d = -1 * o2_left_corner.dot(left_normal)
            third_rule = o1_zy_center.dot(left_normal) + left_d >= 0

            # Right half-space defined by p'n + d = 0
            right_normal = sim_util.rotate_2d(np.array([0,1]), np.pi/2. - self.params['theta_predicte_lr_fb_ab'])
            right_d = -1 * o2_right_corner.dot(right_normal)
            fourth_rule = o1_zy_center.dot(right_normal) + right_d >= 0

            zy_works = third_rule and fourth_rule

            # All corners check
            fifth_rule = np.all(o1_xy_center[1] >= cf_o2_bbox_corners[1,:8].max())

            # o1 bottom corners check
            sixth_rule = np.all(cf_o1_bbox_corners[1, :8].min() >= cf_o2_bbox_corners[1,8])

            return xy_works and zy_works and fifth_rule and sixth_rule

        obj1_id = o1_predicate_args['obj_id']
        obj2_id = o2_predicate_args['obj_id']

        # For symmetry, check if o1 is above o2, and if o2 is below o1
        cf_o1_bbox_corners = o1_predicate_args['cf_bbox_corners'].copy()
        cf_o2_bbox_corners = o2_predicate_args['cf_bbox_corners'].copy()
        o1_above_o2 = above(cf_o1_bbox_corners, cf_o2_bbox_corners)

        cf_o1_bbox_corners[1,:] = cf_o1_bbox_corners[1,:] * -1
        cf_o2_bbox_corners[1,:] = cf_o2_bbox_corners[1,:] * -1
        o2_below_o1 = above(cf_o2_bbox_corners, cf_o1_bbox_corners)            

        if o1_above_o2 or o2_below_o1:
            other_args['predicates'].append((obj1_id, obj2_id, 'above'))
            other_args['predicates'].append((obj2_id, obj1_id, 'below'))

    def compute_occluded_predicates(self, o1_predicate_args, o2_predicate_args, other_args):
        """ Compute if obj2 is occluded obj1. 

            Relation rules:
                1) obj1's mask MUST be occluded by obj2 by X IoU of true mask. 
        """
        obj1_id = o1_predicate_args['obj_id']
        obj2_id = o2_predicate_args['obj_id']

        view_matrix = p.computeViewMatrix(other_args['camera_pos'], 
                                          other_args['lookat_pos'], 
                                          other_args['camera_up_vector'])
        proj_matrix = p.computeProjectionMatrixFOV(self.params['fov'], 
                                                   self.params['img_width']/self.params['img_height'], 
                                                   self.params['near'], 
                                                   self.params['far'])

        # Take photo while o2 is in
        temp = p.getCameraImage(self.params['img_width'], self.params['img_height'], 
                                viewMatrix=view_matrix, projectionMatrix=proj_matrix,
                                renderer=p.ER_BULLET_HARDWARE_OPENGL)
        seg_img = np.array(temp[4]).reshape(self.params['img_height'],self.params['img_width'])
        mask_w_obj2 = seg_img == self._obj_id_to_body[obj1_id].bid

        # Move o2 away (behind camera), take a photo
        obj_pos, obj_orn = self.get_state(obj2_id)
        obj_pos = np.array(obj_pos); 
        camera_pos = np.array(other_args['camera_pos'])
        pos_diff = np.array(obj_pos) - camera_pos
        new_obj_pos = camera_pos - pos_diff
        self.set_state(obj2_id, new_obj_pos, obj_orn) # move it away

        temp = p.getCameraImage(self.params['img_width'], self.params['img_height'], 
                                viewMatrix=view_matrix, projectionMatrix=proj_matrix,
                                renderer=p.ER_BULLET_HARDWARE_OPENGL)
        seg_img = np.array(temp[4]).reshape(self.params['img_height'],self.params['img_width'])
        mask_wout_obj2 = seg_img == self._obj_id_to_body[obj1_id].bid

        self.set_state(obj2_id, obj_pos, obj_orn) # move it back


        # obj2_desc = [x for x in self.export_scene_to_dictionary()['object_descriptions'] if x['obj_id'] == obj2_id][0] # HACK
        # self.remove(obj2_id)
        # self.remove_shapenet_obj(obj2_id) # load_obj will add this back. if this isn't called, info will be duplicated
        # temp = p.getCameraImage(self.params['img_width'], self.params['img_height'], 
        #                         viewMatrix=view_matrix, projectionMatrix=proj_matrix,
        #                         renderer=p.ER_BULLET_HARDWARE_OPENGL)
        # seg_img = np.array(temp[4]).reshape(self.params['img_height'],self.params['img_width'])
        # mask_wout_obj2 = seg_img == self._obj_id_to_body[obj1_id].bid

        # self.load_obj(obj2_desc, '')

        occluded_IoU = sim_util.mask_IoU(np.logical_and(~mask_w_obj2, mask_wout_obj2), mask_wout_obj2)
        if occluded_IoU >= self.params['occ_IoU_threshold']:
            other_args['predicates'].append((obj1_id, obj2_id, 'is_occluded'))
            other_args['predicates'].append((obj2_id, obj1_id, 'occludes'))
