import rl_utils as _utils
from configuration import Configuration
import math
import sys
import os
import numpy as np

overwrite_directory = False # True False # WARNING: overwrite_directory will clear all old model data in that working directory
data_path = '/home/tim/Dropbox/data' # CHANGE to your path
models_path = '/home/tim/Dropbox/models' # CHANGE to your path

airsim_map = sys.argv[1] # Block or AirSimNH
motion = sys.argv[2] # 2d or 3d motino
version = sys.argv[3] # the version is used to compare different hyper parameters using notebook_evals.ipynb (user defined)
case = sys.argv[4] # the case is used as a higher level to change hypers (user defined)
run_name = sys.argv[5] # for repeat random runs with same model hypers
device = sys.argv[6] # load DNN models on cuda:x or cpu device

# logging vars
astar_name = '32m' # read astar paths for training/testing from this subdirectory in data/astar_paths/ parent directory
astar_dir = f'{data_path}/astar_paths/{astar_name}'
project_name = 'Navi' # highest level for specific project (ie Navi NaviSlim NaviSplit etc)
experiment_name = 'DQN_v0' # second level for a specific experiment (ie changing model structure)
trial_name = f'{airsim_map}_{motion}_{version}_{case}'
job_name = f'trainNavi_{trial_name}_{run_name}' # logging purposes
working_directory = models_path + '/'.join([project_name, experiment_name, trial_name, run_name])+'/'

_utils.set_local_parameter('airsim_map', airsim_map)
_utils.set_local_parameter('motion', motion)
_utils.set_local_parameter('project_name', project_name)
_utils.set_local_parameter('experiment_name', experiment_name)
_utils.set_local_parameter('trial_name', trial_name)
_utils.set_local_parameter('job_name', job_name)
_utils.set_local_parameter('run_name', run_name)
_utils.set_local_parameter('astar_name', astar_name)
_utils.set_local_parameter('device', device)

# case based vars
astar_multiplier = 2 # determines max length of an episode
start_level, max_level = 0, 5 # index range of path difficulties to train and evaluate on, inclusive
total_timesteps = int(2e6) # maximum number of timesteps to train on
	# SB3 default is 1e6, Microsoft uses 5e5
buffer_size = int(1e5) # number of recent steps (observation, action, reward) tuples to store in memory to train on -- takes up memory
	# ^^ SB3 default is 1e6, Microsoft uses 5e5, I typically have to use less because of memory constraints
#exploration_fraction =  0.1
stop_annealing = int(1e4) # number of steps to stop annealing the exploration rate at
exploration_fraction = stop_annealing / total_timesteps
	# SB3 default is 0.1*total_timesteps, Microsoft uses 5e4
depth_sensor = 'DepthV2' # forward sensor to use as input to navigation model
actions = [
	'RotateRight90',
	'RotateLeft90',
	'MoveForward2',
	'MoveForward4',
	'MoveForward8',
	'MoveForward16',
	'MoveForward32',
	#'PivotRight2',
	#'PivotLeft2',
]
if case in ['', 'a']:
	pass

# version based vars
if version in ['', 'v1']:
	pass

# write run files to this directory
if os.path.exists(working_directory + 'completed.p'):
	_utils.progress(job_name, f'already complete')
	print('experiment completed already!')
	sys.exit(1)

# file IO
astar_train_paths_file = f'{astar_dir}{airsim_map}_{motion}_train.p'
astar_test_paths_file = f'{astar_dir}{airsim_map}_{motion}_test.p'
rooftops_path = f'{data_path}rooftops/{airsim_map}.p' # match to map or use voxels if not available
observations_path = f'{data_path}observations/'
queries_path = None
# assuming forward and downward sensor dimensions are same
sensor_info = _utils.read_json(f'{observations_path}{depth_sensor}/info.json')
image_bands, image_height, image_width = sensor_info['array_size']

# bounds drone can move in
region = 'all'
map_resolution_x = 2
map_resolution_y = 2
map_resolution_z = 4
z_bounds = [-4, -3]
if motion in '3d':
	z_bounds = [-16, 0]
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
x_vals = [x for x in range(x_bounds[0], x_bounds[1]+1, map_resolution_x)]
y_vals = [y for y in range(y_bounds[0], y_bounds[1]+1, map_resolution_y)]
z_vals = [z for z in range(z_bounds[0], z_bounds[1]+1, map_resolution_z)]
yaw_vals = [0, 1, 2, 3] # what yaws are accessible by drone
id_name = 'alpha' # when reading in observation data, which ID key words to use

# check if we continue training from previous run
continue_training = False
if os.path.exists(working_directory + 'configuration.json') and not overwrite_directory:
	continue_training = True

# setup for run, set system vars and prepare file system
_utils.setup(
	working_directory, 
	project_name=project_name,
	experiment_name=experiment_name,
	trial_name=trial_name,
	run_name=run_name,
	overwrite_directory=overwrite_directory,
	)

# make controller to run configuration on (we will train a model)
from controllers.train import Train
controller = Train(
	model_component = 'Model',
	environment_component = 'Environment',
	continue_training = continue_training,
	total_timesteps = total_timesteps,
	)

# continue training will load runs folder and pick up where it left off
if continue_training:
	# load configuration file and create object to save and connect components
	read_configuration_path = working_directory + 'configuration.json'
	configuration = Configuration.load(read_configuration_path, controller)
	meta = configuration.meta
	meta['continued_training'] = True
	# read model and/or replay buffer
	# get highest level complete
	modeling_dir = f'{working_directory}modeling/'
	fnames = os.listdir(modeling_dir)
	model_path = None
	buffer_path = f'{modeling_dir}replay_buffer.zip'
	if not os.path.exists(buffer_path):
		buffer_path = None
	highest_level = -1
	for fname in fnames:
		if 'model' in fname:
			parts = fname.split('.')[0].split('_')
			level = int(parts[2])
			if level > highest_level:
				highest_level = level
				model_path = f'{modeling_dir}{fname}.zip'
	model_component = configuration.get_component('Model')
	model_component.read_model_path = model_path
	model_component.read_replay_buffer_path = buffer_path
	_utils.speak('continuing training...')

# if not continuing training then make a brand spaking new config
else:
	# set meta data (anything you want here, just writes to config file as a dict)
	meta = {
		}

	## make a new configuration file to add components to 
		# this obj will be used to serialize components, and log experiment history
		# any components created after making this configuration file will auto be added to it
		# components use the name of other components which will be created and connected later
		# this is done to handle different objects which need to be connected to eachother and connected in different orders
		# there is a baked in priority que for connecting components if you need to handle this differently
	configuration = Configuration(
		meta, 
		controller,
		)

	## create environment component to handle step() and reset() for DRL model training
	from environments.goalenv import GoalEnv
	GoalEnv(
		drone_component = 'Drone', 
		actor_component = 'Actor', 
		observer_component = 'Observer', 
		rewarder_component = 'Rewarder',
		model_component = 'Model',
		map_component = 'Map',
		spawner_component = 'Spawner',
		crash_handler=False,
		name = 'Environment',
		)

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

	## REWARD FUNCTION
	# we will reward moving closer, reward reaching goal, penalize too many steps, and penalize collisions
	rewards = []
	reward_weights = []
	# heavy penalty for out of bounds
	from rewards.bounds import Bounds
	Bounds(
		drone_component = 'Drone',
		x_bounds = x_bounds,
		y_bounds = y_bounds,
		z_bounds = z_bounds,
		name = 'BoundsReward',
		)
	rewards.append('BoundsReward')
	reward_weights.append(1)
	# heavy penalty for collision
	from rewards.collision import Collision
	Collision(
		drone_component = 'Drone',
		name = 'CollisionReward',
		)
	rewards.append('CollisionReward')
	reward_weights.append(1)
	# heavy reward for reaching goal
	from rewards.goal import Goal
	Goal(
		drone_component = 'Drone',
		goal_component = 'Spawner',
		include_z = True if motion in '3d' else False, # includes z in distance calculations
		tolerance = 2, # must reach goal within this many meters
		terminate = True, # we are terminating this example when the drone realizes it reached the goal, collides, or reaches max
		name = 'GoalReward',
		)
	rewards.append('GoalReward')
	reward_weights.append(10)
	# heavy penalty for using too many steps
	from rewards.maxsteps import MaxSteps
	MaxSteps(
		spawner_component = 'Spawner',
		max_max = 200, # absolute max number of steps, regardless of scaling from further goals
		use_astar = True, # bases max steps based on astar length
		astar_multiplier = astar_multiplier, # max step size is this many times the astar length
		name = 'MaxStepsReward',
		)
	rewards.append('MaxStepsReward')
	reward_weights.append(1)
	# intermediate penalty for using more steps
	from rewards.steps import Steps
	Steps(
		name = 'StepsReward',
		)
	rewards.append('StepsReward')
	reward_weights.append(.1)
	# intermediate reward for approaching goal
	from rewards.distance import Distance
	Distance(
		drone_component = 'Drone',
		goal_component = 'Spawner',
		include_z = True if motion in '3d' else False, # includes z in distance calculations
		name = 'DistanceReward',
		)
	rewards.append('DistanceReward')
	reward_weights.append(.1)
	# REWARDER
	from rewarders.schema import Schema
	Schema(
		rewards_components = rewards,
		reward_weights = reward_weights, 
		name = 'Rewarder',
		)

	## ACTION SPACE
	# we will just move forward and rotate for this example
	from actions.fixedrotate import FixedRotate 
	FixedRotate(
		drone_component = 'Drone',  
		yaw_diff = math.pi/2, # can rotate at 90 deg increments
		name = 'RotateRight90',
		)
	FixedRotate(
		drone_component = 'Drone',  
		yaw_diff = -1*math.pi/2, # can rotate at 90 deg increments
		name = 'RotateLeft90',
		)
	from actions.fixedmove import FixedMove
	FixedMove(
		drone_component = 'Drone', 
		x_distance = 2,
		adjust_for_yaw = True, # this adjusts movement based on current yaw for relative moves
		name = 'MoveForward2',
		)
	FixedMove(
		drone_component = 'Drone', 
		x_distance = 4,
		adjust_for_yaw = True, # this adjusts movement based on current yaw for relative moves
		name = 'MoveForward4',
		)
	FixedMove(
		drone_component = 'Drone', 
		x_distance = 6,
		adjust_for_yaw = True, # this adjusts movement based on current yaw for relative moves
		name = 'MoveForward6',
		)
	FixedMove(
		drone_component = 'Drone', 
		x_distance = 8,
		adjust_for_yaw = True, # this adjusts movement based on current yaw for relative moves
		name = 'MoveForward8',
		)
	FixedMove(
		drone_component = 'Drone', 
		x_distance = 10,
		adjust_for_yaw = True, # this adjusts movement based on current yaw for relative moves
		name = 'MoveForward10',
		)
	FixedMove(
		drone_component = 'Drone', 
		x_distance = 16,
		adjust_for_yaw = True, # this adjusts movement based on current yaw for relative moves
		name = 'MoveForward16',
		)
	FixedMove(
		drone_component = 'Drone', 
		x_distance = 32,
		adjust_for_yaw = True, # this adjusts movement based on current yaw for relative moves
		name = 'MoveForward32',
		)
	FixedMove(
		drone_component = 'Drone', 
		y_distance = 2,
		adjust_for_yaw = True, # this adjusts movement based on current yaw for relative moves
		name = 'PivotRight2',
		)
	FixedMove(
		drone_component = 'Drone', 
		y_distance = -2,
		adjust_for_yaw = True, # this adjusts movement based on current yaw for relative moves
		name = 'PivotLeft2',
		)

	if motion in ['3d']:
		FixedMove(
			drone_component = 'Drone', 
			z_distance = 4,
			name = 'MoveDownward4',
			)
		actions.append('MoveDownward4')
		FixedMove(
			drone_component = 'Drone', 
			z_distance = -4,
			name = 'MoveUpward4',
			)
		actions.append('MoveUpward4')

	## ACTOR
	from actors.teleporterdiscrete import TeleporterDiscrete
	# we use a teleporter here because it is quicker and more stable
		# it will check collisions between current point and telported point then move directly to that location
	TeleporterDiscrete(
		drone_component = 'Drone',
		actions_components = actions,
		discretize=True,
		name = 'Actor',
		)

	## OBSERVATION SPACE
	# we will use the relative displacement between drone and goal, and front-facing depth maps
	# we will use the T0many most recent observations concatenated toghter, for this example T = 4
	# TRANSFORMERS
	from transformers.normalize import Normalize
	Normalize(
		min_input = -1*math.pi, # min angle
		max_input = math.pi, # max angle
		name = 'NormalizeOrientation',
		)
	Normalize(
		min_input = 1, # in front of sensor
		max_input = 100, # horizon
		left = 0,
		name = 'NormalizeDistance',
		)
	
	# SENSORS
	vector_sensors = []
	vector_length_forget = 0
	vector_length_remember = 0
	img_sensors = []
	image_bands_forget = 0
	image_bands_remember = 0
	# forward depth sensor
	from others.datadict import DataDict
	DataDict(
		data_dir = f'{observations_path}{depth_sensor}/{airsim_map}/',
		id_name=id_name, x_vals=x_vals, y_vals=y_vals, z_vals=z_vals, yaw_vals=yaw_vals,
		name = 'DataDictForward',
	)
	from sensors._client import _Client
	_Client(
		datadict_component = 'DataDictForward',
		drone_component = 'Drone',
		sensor_name = depth_sensor,
		map_name = airsim_map,
		observations_path = observations_path,
		queries_path = queries_path,
		transformers_components=[],
		name = depth_sensor,
	)
	img_sensors.append(depth_sensor)
	image_bands_remember += image_bands
	# sense horz distance to goal
	from sensors.distance import Distance
	Distance(
		misc_component = 'Drone',
		misc2_component = 'Spawner',
		include_x = True,
		include_y = True,
		include_z = False,
		prefix = 'drone_to_goal',
		transformers_components = [
			'NormalizeDistance',
			], 
		name = 'GoalDistanceXY',
		)
	vector_sensors.append('GoalDistanceXY')
	vector_length_remember += 1
	# sense yaw difference to goal 
	from sensors.orientation import Orientation
	Orientation(
		misc_component = 'Drone',
		misc2_component = 'Spawner',
		prefix = 'drone_to_goal',
		transformers_components = [
			'NormalizeOrientation',
			],
		name = 'GoalOrientation',
		)
	vector_sensors.append('GoalOrientation')
	vector_length_remember += 1
	from sensors.distancebounds import DistanceBounds
	DistanceBounds(
		drone_component = 'Drone',
		x_bounds = x_bounds,
		y_bounds = y_bounds,
		z_bounds = z_bounds,
		include_z = True if motion in '3d' else False,
		transformers_components = [
			'NormalizeDistance',
			],
		name = 'DistanceBounds',
	)
	vector_sensors.append('DistanceBounds')
	vector_length_remember += 1
	# sense vert distance to goal
	if motion in '3d':
		Distance(
			misc_component = 'Drone',
			misc2_component = 'Spawner',
			include_x = False,
			include_y = False,
			include_z = True,
			prefix = 'drone_to_goal',
			transformers_components = [
				'NormalizeDistance',
				], 
			name = 'GoalDistanceZ',
			)
		vector_sensors.append('GoalDistanceZ')
		vector_length_remember += 1
	# OBSERVER
	# currently must count vector size of sensor output
	from observers.single import Single
	Single(
		map_component = 'Map',
		sensors_components = vector_sensors,
		vector_length_forget = vector_length_forget,
		vector_length_remember = vector_length_remember,
		nPast = nPast,
		null_if_in_obj = True, # inside an object
		null_if_oob = True, # out of bounds
		name = 'VecObserver',
		)
	Single(
		sensors_components = img_sensors, 
		is_image = True,
		image_height = image_height, 
		image_width = image_width,
		image_bands_forget = image_bands_forget,
		image_bands_remember = image_bands_remember,
		nPast = nPast,
		null_if_in_obj = True, # inside an object
		null_if_oob = True, # out of bounds
		name = 'ImgObserver',
		)
	from observers.multi import Multi
	Multi(
		vector_observer_component = 'VecObserver',
		image_observer_component = 'ImgObserver',
		name = 'Observer',
		)




	## MODEL
		# we will use a TD3 algorithm from SB3
	from sb3models.dqn import DQN
	DQN(
		environment_component = 'Environment',
		policy = 'MultiInputPolicy',
		buffer_size = buffer_size,
		device = device,
		exploration_fraction = exploration_fraction,
		name = 'Model',
	)

	# SPAWNER
		# moves drone to desired starting location
		# sets the target goal since it is typically dependent on the start location
	from spawners.levels import Levels
	Levels(
		drone_component = 'Drone',
		levels_path = astar_train_paths_file,
		start_level = start_level,
		max_level = max_level,
		random_path = True,
		name = 'Spawner',
	)

	## MODIFIERS
		# modifiers are like wrappers, and will add functionality before or after any component
	# CURRICULUM LEARNING
		# this modifier will be called at the end of every episode to see the percent of succesfull paths
		# if enough paths were succesfull then this will level up to harder goal
	from modifiers.curriculum import Curriculum
	Curriculum(
		base_component = 'Environment',
		parent_method = 'end',
		order = 'post',
		spawner_component = 'Spawner', # which component to level up
		level_up_criteria = 0.9, # percent of succesfull paths needed to level up
		level_up_buffer = 100, # number of previous episodes to look at to measure level_up_criteria
		start_level = start_level,
		max_level = max_level, # can level up this many times after will terminate DRL learning loop
		terminate_on_max = terminate_on_max,
		frequency = 1, # check every this many episodes
		eval_on_levelup=True, # save model everytime we level up
		model_component='Model',
		name = 'Curriculum',
	)
	# SAVERS
		# these will save any intermediate data we want during the training process
	from modifiers.saver import Saver
	checkpoint_freq = total_timesteps # set to a high number to just save at end, curriclum saves on level up as well
	Saver(
		base_component = 'Model',
		parent_method = 'end',
		track_vars = [
					'model', 
					#'replay_buffer', # this can cost alot of memory
					],
		write_folder = working_directory + 'modeling/',
		save_config = False,
		save_log = False,
		order = 'post',
		frequency = checkpoint_freq,
		name = 'ModelingSaver',
	)
	# # save Train states and observations
	# saver = Saver(
	# 	base_component = 'Environment',
	# 	parent_method = 'end',
	# 	track_vars = [
	# 				'observations', 
	# 				'states',
	# 				],
	# 	write_folder = working_directory + 'states/',
	# 	order = 'post',
	# 	save_config = False,
	# 	save_benchmarks = False,
	# 	frequency = checkpoint_freq,
	# 	name='Saver',
	# )


# CONNECT COMPONENTS
configuration.connect_all()

# WRITE CONFIGURATION
configuration.save()

# WRITE CONTROLLER
controller.save(working_directory + 'train_controller.json')

# RUN CONTROLLER
configuration.controller.run()

# done
_utils.speak('training complete!')
model = configuration.get_component('Model')
model_write_path = f'{working_directory}modeling/model_final.zip'
model.save_model(model_write_path)
configuration.save()
curriculum = configuration.get_component('Curriculum')
environment = configuration.get_component('Environment')
_utils.evaluate2(
	f'{working_directory}configuration.json',
	f'{working_directory}modeling/model_final.zip',
	f'{working_directory}test_final/',
)
_utils.pk_write(_utils.get_timestamp(), working_directory + 'completed.p')
_utils.print_local_log()
configuration.disconnect_all()
_utils.progress(job_name, f'complete')