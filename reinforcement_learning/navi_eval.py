# ** insert this head at top of all main files with your proper paths **
local_dir = '/home/tim/local/' # any local files outside of repo
home_dir = '/home/tim/Dropbox/experimental/' # home directory of repo
import os
os.chdir(home_dir) # TODO: uncomment this
import sys
sys.path.append(home_dir)
import utils.global_methods as gm
import map_data.map_methods as mm

# local imports
from configuration import Configuration

# params set from arguments passed in python call
job_name = None # optional param for tracking outside of this program
config_path = 'null' # set with args, file path to configuration.json to load train components and parameters
model_path = 'null' # set with args, file path to model.zip to load sb3 actor model to evaluate
output_dir = 'null' # set with args, directory path to folder to write results
split_name = 'null' # set with args, i.e. train, val, test -- which portion of paths_split to use from astar path
device = 'cuda:0' # device to load pytorch model on
min_level, max_level = 0, 7 # range of curric difficulty levels to test on
overwrite_directory = False
random_seed = 42
save_states = True

# read params from command line
arguments = gm.parse_arguments(sys.argv[1:])
locals().update(arguments)
gm.set_global('job_name', job_name)
gm.set_global('random_seed', random_seed)
gm.set_global('home_dir', home_dir)
gm.set_global('local_dir', local_dir)

assert os.path.exists(config_path), f'config_path DNE at {config_path}'
assert os.path.exists(model_path), f'model_path DNE at {model_path}'
assert output_dir!='null', f'output_dir not passed as arg'

# setup for run, set system vars and prepare file system
gm.setup_output_dir(output_dir, overwrite_directory)

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
astar_dir = f'{home_dir}map_data/astar_paths/'
rooftops_dir = f'{home_dir}map_data/rooftops/'
observations_dir = f'{home_dir}map_data/observations/'
airsim_map = configuration.get_parameter('airsim_map')
motion = configuration.get_parameter('motion')
rooftops_version = configuration.get_parameter('rooftops_version')
astar_version = configuration.get_parameter('astar_version')
region = configuration.get_parameter('region')
path_splits = configuration.get_parameter('path_splits')
astar_name = configuration.get_parameter('astar_name')
configuration.set_parameter('random_seed', random_seed)
rooftops_path = f'{rooftops_dir}{rooftops_version}/{airsim_map}.p' # match to map or use voxels i
x_bounds, y_bounds, z_bounds = mm.get_bounds(airsim_map, region, motion)
astar_paths_file = f'{astar_dir}{astar_version}/{airsim_map}_{motion}_{astar_name}.p'

## CONTROLLER -- we will test on config
from controllers.test import Test
controller = Test(
		environment_component = 'Environment', # environment to run test in
		model_component = 'Model', # used to make predictions
		spawner_component = 'Spawner', # used to make predictions
		results_directory = output_dir,
	)
configuration.set_controller(controller)
    

# **** new components to create for testing, to add to those found in the configuration file

# SPAWNER - spawn drone at first point in each astar path with target of last point
from spawners.levels import Levels
Levels(
	drone_component = 'Drone',
	levels_path = astar_paths_file,
	min_level = min_level,
	max_level = max_level,
	path_splits = path_splits,
	split_name = split_name,
	name = 'Spawner',
)
# SAVERS - save observations and/or states at each step
from modifiers.saver import Saver
# save Train states and observations
if save_states:
	Saver(
		base_component = 'Environment',
		parent_method = 'end',
		track_vars = [
					#'observations', # uncomment this to write observations to file (can take alot of disk space)
					'states',
					],
		write_folder = output_dir + 'states/',
		order = 'post',
		save_config = False,
		save_benchmarks = False,
		frequency = 999_999,
		name='Saver',
	)

# CONNECT COMPONENTS
configuration.connect_all()

# WRITE CONFIGURATION
configuration.save()

# RUN CONTROLLER
gm.progress(job_name, f'started')
configuration.controller.run()

# done
gm.progress(job_name, f'complete')
#gm.speak('Evaluations done!')
configuration.disconnect_all()