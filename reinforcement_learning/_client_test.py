import rl_utils as _utils
from configuration import Configuration
import math
import sys
import os
import numpy as np


instance_name, home_path, local_path, data_path, models_path = _utils.db_setup(levels_in=1)

# mandatory params
airsim_map = sys.argv[1]
motion = sys.argv[2]
project_name = sys.argv[3]
experiment_name = sys.argv[4]
trial_name = sys.argv[5]
run_name = sys.argv[6]
test_name = sys.argv[7]
astar_name = sys.argv[8]
model_name = sys.argv[9]
start_level = int(sys.argv[10])
max_level = int(sys.argv[11])
num_per = int(sys.argv[12])
overwrite = sys.argv[13] in ['1', 'True', 'true']
device = sys.argv[14]

# set params for all classes
_utils.set_local_parameter('airsim_map', airsim_map)
_utils.set_local_parameter('motion', motion)
_utils.set_local_parameter('project_name', project_name)
_utils.set_local_parameter('experiment_name', experiment_name)
_utils.set_local_parameter('trial_name', trial_name)
_utils.set_local_parameter('run_name', run_name)
_utils.set_local_parameter('test_name', test_name)
_utils.set_local_parameter('astar_name', astar_name)
_utils.set_local_parameter('model_name', model_name)
_utils.set_local_parameter('start_level', start_level)
_utils.set_local_parameter('max_level', max_level)
_utils.set_local_parameter('num_per', num_per)
_utils.set_local_parameter('overwrite', overwrite)
_utils.set_local_parameter('device', device)

# file paths
astar_paths_file = f'{data_path}{astar_name}/{airsim_map}_{motion}_test.p'
rooftops_path = f'{data_path}rooftops/{airsim_map}.p' # match to map or use voxels i
read_dir = models_path + '/'.join([project_name, experiment_name, trial_name, run_name])+'/'
write_dir = f'{read_dir}{test_name}/'
working_directory = write_dir 
read_model_file = f'{read_dir}modeling/{model_name}.zip'
read_config_file = read_dir + 'configuration.json'
job_name = f'testNavi_{airsim_map}_{motion}_{project_name}_{experiment_name}_{trial_name}_{run_name}_{test_name}_{astar_name}_{model_name}_{start_level}_{max_level}_{num_per}_{overwrite}'
if not overwrite and os.path.exists(working_directory + 'evaluation.json'):
	_utils.progress(job_name, f'already complete')
	print('experiment completed already!')
	sys.exit(1)
if not os.path.exists(astar_paths_file) or not os.path.exists(rooftops_path) or not os.path.exists(read_model_file) or not os.path.exists(read_config_file):
	_utils.progress(job_name, f'failed missing needed files')
	print('missing neeed files.')
	sys.exit(1)
    

# bounds drone can move in
region = 'all'
z_bounds = [-40, 0]
if airsim_map in ['AirSimNH']:
	if region in ['all']:
		x_bounds = [-240, 240]
		y_bounds = [-240, 240]
	elif region in ['train']: 
		x_bounds = [-10, 240]
		y_bounds = [-240, 240]
	elif region in ['test']:
		x_bounds = [-240, 10]
		y_bounds = [-240, 240]
if airsim_map in ['Blocks']:
	if region in ['all']:
		x_bounds = [-120, 100]
		y_bounds = [-140, 140]
	elif region in ['train']:
		x_bounds = [-60, 100]
		y_bounds = [-140, 140]
	elif region in ['test']:
		x_bounds = [-120, 0]
		y_bounds = [-140, 140]


# read astars paths for curriculum learning... 
astar_paths = _utils.pk_read(astar_paths_file)
num_sublevels = np.sum([len(astar_paths['levels'][level]) for level in range(start_level,max_level+1)])
num_episodes = int(num_per * num_sublevels)

# setup for run, set system vars and prepare file system
_utils.setup(
	working_directory,
	)

## CONTROLLER
from controllers.test import Test
controller = Test(
		environment_component = 'Environment', # environment to run test in
		model_component = 'Model', # used to make predictions
		results_directory = working_directory,
		num_episodes = num_episodes,
		job_name = job_name,
	)
# SET META DATA (anything you want here, just writes to config file as a dict)
meta = {
	'notes': '',
}

## read old CONFIGURATION 
configuration = Configuration.load(
	read_config_file, # read all components in this config file
	controller, # evaluate
	read_modifiers=False, # no modifiers from train data - we will make new ones
	skip_components = [
		'Map', 
		'Spawner', 
		'Rooftops',
		'Drone',
		], # change map we launch in, change how we spawn
	change_params={
		'device':device,
		# 'x_bounds':x_bounds,
		# 'y_bounds':y_bounds,
		# 'z_bounds':z_bounds,
		}
	)
configuration.update_meta(meta)

# create map object
from maps._etherial import _Etherial
_Etherial(
	rooftops_component='Rooftops',
	x_bounds = x_bounds,
	y_bounds = y_bounds,
	z_bounds = z_bounds,
	name = 'Map',
	)
# read rooftops data struct to determine z-height of nearest collidable object below or above x,y coords
from others.rooftops import Rooftops
Rooftops(
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

# SPAWNER - spawn drone at first point in each astar path with target of last point
from spawners.levels import Levels
Levels(
	drone_component = 'Drone',
	levels_path = astar_paths_file,
	paths_per_sublevel = num_per,
	start_level = start_level,
	max_level = max_level,
	name = 'Spawner',
)
# SAVERS - save observations and states at each step
from modifiers.saver import Saver
# save Train states and observations
Saver(
	base_component = 'Environment',
	parent_method = 'end',
	track_vars = [
				#'observations', 
				'states',
				],
	write_folder = working_directory + 'states/',
	order = 'post',
	save_config = False,
	save_benchmarks = False,
	frequency = num_episodes,
	name='Saver',
)
# MODEL - change read path
model_component = configuration.get_component('Model')
model_component.read_model_path = read_model_file

# CONNECT COMPONENTS
configuration.connect_all()

# WRITE CONFIGURATION
configuration.save()

# RUN CONTROLLER
configuration.controller.run()

# done
_utils.speak('Evaluations done!')
configuration.disconnect_all()
_utils.progress(job_name, f'complete')