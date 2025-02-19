import rl_utils as _utils
from configuration import Configuration
import math
import sys
import os
import numpy as np


instance_name, home_path, local_path, data_path, models_path = _utils.setup(levels_in=1)

# defaults
device = 'cuda:0'
if instance_name in ['ace']:
	device = 'cuda:2'
	
# path to airsim release file - precompiled binary
airsim_map = 'Blocks'

# set path to read rooftops from to determine collidable surface positions
rooftops_path = f'{data_path}rooftops/{airsim_map}.p' # match to map or use voxels if not available

# path to get sensor observations from
observations_path = f'{data_path}observations/'

# path to make queries to
queries_path = home_path + '/'.join(['hosts', airsim_map, 'queries']) + '/'

# bounds drone can move in
x_steps = [x for x in range(-60, 101)]
y_steps = [y for y in range(-140, 141)]
z_steps = [z for z in range(-4, -3)]
yaw_steps = [0*np.pi/2, 1*np.pi/2, 2*np.pi/2, 3*np.pi/2]

# write run files to this directory
working_directory = f'{home_path}data/{airsim_map}/'

# setup for run, set system vars and prepare file system
_utils.setup(
	working_directory,
	)

# make data controller to teleport to each specified point and capture all data from given sensors
sensors = [ # we will create these sensor components below (pass in string names here, this is to handle connection priority later)
	'ClientSensor',
	] # fill this array with name of desired sensors to capture data from (create components below)
from controllers.data import Data
controller = Data(
	drone_component = 'Drone', # drone to issue commands to move around map
	sensors_components = sensors, # list of sensors to fetch data
	map_component = 'Map', # map to move around in
	points = [], # list of points to visit
	write_all=False,
	crash_handler=False,
)

# set meta data (anything you want here, just writes to config file as a dict)
meta = {
	'instance_name': instance_name,
	}

## make a new configuration file to add components to 
configuration = Configuration(meta, controller)

# create map object
from maps._etherial import _Etherial
_Etherial(
	rooftops_component='Rooftops',
	name = 'Map',
	)
# read rooftops data struct to determine z-height of nearest collidable object below or above x,y coords
from others.rooftops import Rooftops
rooftops = Rooftops(
	read_path = rooftops_path,
	name = 'Rooftops',
)

# drone controller component - we will use AirSim
	# this can also be real world drone controller like Tello
from drones._etherial import _Etherial
_Etherial(
	map_component = 'Map',
	name = 'Drone',
	)

from sensors._client import _Client
_Client(
	sensor_name = 'FlattenedDepthV1',
	map_name = airsim_map,
	observations_path = observations_path,
	queries_path = queries_path,
	name = 'ClientSensor',
	)

# CONNECT COMPONENTS
configuration.connect_all()

fetch_points = []
fetch_states = []
for x in x_steps:
	for y in y_steps:
		for z in z_steps:
			in_object = rooftops.in_object(x, y, z)
			if in_object: # check if point in object
				continue
			for yaw in yaw_steps:
				point = [x, y, z, yaw]
				fetch_points.append(point)
				fetch_states.append({
					'drone_position':[x,y,z],
					'yaw':yaw,
				})
controller.points = fetch_points # random_points
controller.states = fetch_states # random_points

# WRITE CONFIGURATION
configuration.save()

# RUN CONTROLLER
configuration.controller.run()

# done
_utils.speak('learning loop terminated. Now exiting...')
configuration.controller.stop()
