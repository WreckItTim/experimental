import rl_utils as _utils
from configuration import Configuration
import math
import sys
import os
import numpy as np
import modules

# params set from arguments passed in python call
config_path = sys.argv[1] # direct path to configuration.json file to load rl components
model_path = sys.argv[2] # direct path to model.zip file to read with neural network
working_directory = _utils.fix_directory(sys.argv[3]) # directory path to write test results

assert os.path.exists(config_path), f'configuration path DNE at {config_path}'
assert os.path.exists(model_path), f'model path DNE at {model_path}'

# global params for you to set
device = 'cuda:0' # device to load pytorch model on
data_path = '/home/tim/Dropbox/data/' # CHANGE to your path -- parent directory for all reading
start_level, max_level = 0, 5 # start_level, max_level to test on
num_evals_per_sublevel = 10

# setup for run, set system vars and prepare file system
_utils.setup(working_directory)

## read old CONFIGURATION 
# SET META DATA (anything you want here for notes, just writes to config file as a dict)
meta = {}
configuration = Configuration.load(
	config_path, # read all components in this config file
	read_modifiers=False, # do not load modifiers used in train configuration - we will make new ones
	skip_components = [ # do not load these components because we will overwrite them for testing
		'Spawner', # change how we spawn
		],
	change_params={ # change parameters in components to desired value
		'device':device, # load model onto specificed pytorch device
		'read_model_path':model_path, # specify where to load model
		}
	)
configuration.update_meta(meta)

# parameters set from config file
airsim_map = configuration.parameters['airsim_map']
motion = configuration.parameters['motion']
rooftops_path = f'{data_path}rooftops/v1/{airsim_map}.p' # match to map or use voxels i
x_bounds, y_bounds, z_bounds = modules.bounds_v1(airsim_map, motion) # bounds that drone can move around map in
astar_paths_file = f'{data_path}astar_paths/v1/{airsim_map}_{motion}_test.p'
astar_paths = _utils.pk_read(astar_paths_file)
num_sublevels = np.sum([len(astar_paths['levels'][level]) for level in range(start_level, max_level+1)])
num_episodes = int(num_evals_per_sublevel * num_sublevels)

## CONTROLLER -- we will test on config
from controllers.test import Test
controller = Test(
		environment_component = 'Environment', # environment to run test in
		model_component = 'Model', # used to make predictions
		results_directory = working_directory,
		num_episodes = num_episodes,
	)
configuration.set_controller(controller)
    

# **** new components to create for testing, to add to those found in the configuration file

# SPAWNER - spawn drone at first point in each astar path with target of last point
from spawners.levels import Levels
Levels(
	drone_component = 'Drone',
	levels_path = astar_paths_file,
	paths_per_sublevel = num_evals_per_sublevel,
	start_level = start_level,
	max_level = max_level,
	name = 'Spawner',
)
# SAVERS - save observations and/or states at each step
from modifiers.saver import Saver
# save Train states and observations
Saver(
	base_component = 'Environment',
	parent_method = 'end',
	track_vars = [
				#'observations', # uncomment this to write observations to file (can take alot of disk space)
				'states',
				],
	write_folder = working_directory + 'states/',
	order = 'post',
	save_config = False,
	save_benchmarks = False,
	frequency = num_episodes,
	name='Saver',
)

# CONNECT COMPONENTS
configuration.connect_all()

# WRITE CONFIGURATION
configuration.save()

# RUN CONTROLLER
configuration.controller.run()

# done
_utils.speak('Evaluations done!')
configuration.disconnect_all()