# ** insert this head at top of all main files with your proper paths **
local_dir = '/home/tim/local/' # any local files outside of repo
airsim_maps_dir = f'{local_dir}airsim_maps/'
home_dir = '/home/tim/Dropbox/' # home directory of repo
data_dir = f'{home_dir}data/'
observations_dir = f'{data_dir}observations/'
rooftops_dir = f'{data_dir}rooftops/'
astar_dir = f'{data_dir}astar_paths/'
models_dir = f'{home_dir}models/'
contexts_dir = f'{home_dir}contexts/'
import sys
sys.path.append(home_dir)
from methods import *

# local imports
from configuration import Configuration
import math

# default parametrs
job_name = 'null_' + random.randint(0, 1_000_000)
run_directory = job_name + '/'
device = 'cuda:0'
continue_training = False
contine_model_at = None # specify model path to continue training from
contine_config_at = None # specify configuration path to continue training from
contine_buffer_at = None
airsim_map = 'AirSimNH'
overwrite_directory = False # True will erase all files at otuput working directory
sensor_name = 'DepthV2' # forward sensor to use as input to navigation model
rooftops_version = 'v1'
astar_version = 'v1'
context_name = 'airsim'
level_up_criteria = 0.9 # percent of succesfull paths needed to level up
level_up_buffer = 100
frequency = 1
goal_tolerance = 2
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
nPast = 4
motion = '2d'
region = 'all'
map_resolution_x, map_resolution_y, map_resolution_z = 2, 2, 4
id_name = 'alpha' # when reading in observation data, which ID key words to use
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
if motion == '3d':
	actions = actions + [
		'MoveDownward4',
		'MoveUpward4',
	]

# read params from command line
arguments = parse_arguments(sys.argv[1:])
locals().update()
set_global_parameter('job_name', job_name)
context = __import__(f'contexts.{context_name}', fromlist=[''])
set_global_parameter('context_name', context_name)
set_global_parameter('context', context)

# set variable subpaths from root directories and params set above
working_directory = f'{models_dir}{run_directory}'
astar_train_paths_file = f'{astar_dir}{astar_version}/{airsim_map}_{motion}_train.p'
astar_test_paths_file = f'{astar_dir}{astar_version}/{airsim_map}_{motion}_test.p'
rooftops_path = f'{rooftops_dir}{rooftops_version}/{airsim_map}.p' # match to map or use voxels if not available
sensor_dir = data_dir
sensor_info = read_json(f'{observations_dir}{sensor_name}/info.json')
image_bands, image_height, image_width = sensor_info['array_size']
setup(working_directory, overwrite_directory)

# bounds drone can move in
x_bounds, y_bounds, z_bounds = context.get_bounds(airsim_map, region, motion)
x_vals = [x for x in range(x_bounds[0], x_bounds[1]+1, map_resolution_x)]
y_vals = [y for y in range(y_bounds[0], y_bounds[1]+1, map_resolution_y)]
z_vals = [z for z in range(z_bounds[0], z_bounds[1]+1, map_resolution_z)]
yaw_vals = [0, 1, 2, 3] # what yaws are accessible by drone


# COMPONENTS

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
	if contine_config_at is None:
		contine_config_at = working_directory + 'configuration.json'
	configuration = Configuration.load(contine_config_at, controller)
	meta = configuration.meta
	meta['continued_training'] = True
	# read model and/or replay buffer
	# get highest level complete
	modeling_dir = f'{working_directory}modeling/'
	if contine_buffer_at is None:
		contine_buffer_at = f'{modeling_dir}replay_buffer.zip'
	if contine_model_at is None:
		final_model_path = f'{modeling_dir}model_final.zip'
		if os.path.exists(final_model_path):
			contine_model_at = final_model_path
		else:
			highest_level = -1
			fnames = os.listdir(modeling_dir)
			for fname in fnames:
				if 'model' in fname:
					parts = fname.split('.')[0].split('_')
					level = int(parts[2])
					if level > highest_level:
						highest_level = level
						contine_model_at = f'{modeling_dir}{fname}'
	model_component = configuration.get_component('Model')
	model_component.read_model_path = contine_model_at
	model_component.read_replay_buffer_path = contine_buffer_at
	speak('continuing training...')

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
	configuration = Configuration(meta)
	configuration.set_controller(controller)

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
		crash_handler = False,
		name = 'Environment',
		)

	# create map object
	from maps.etherial import Etherial
	Etherial(
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
	from drones.etherial import Etherial
	Etherial(
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
		tolerance = goal_tolerance, # must reach goal within this many meters
		terminate = True, # we are terminating this example when the drone realizes it reached the goal, collides, or reaches max
		name = 'GoalReward',
		)
	rewards.append('GoalReward')
	reward_weights.append(10)
	# heavy penalty for using too many steps
	from rewards.maxsteps import MaxSteps
	MaxSteps(
		spawner_component = 'Spawner',
		max_max = 50, # absolute max number of steps, regardless of scaling from further goals
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
	FixedMove(
		drone_component = 'Drone', 
		z_distance = 4,
		name = 'MoveDownward4',
		)
	FixedMove(
		drone_component = 'Drone', 
		z_distance = -4,
		name = 'MoveUpward4',
		)

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
		data_dir = f'{observations_dir}{sensor_name}/{airsim_map}/',
		id_name=id_name, x_vals=x_vals, y_vals=y_vals, z_vals=z_vals, yaw_vals=yaw_vals,
		name = 'DataDictForward',
	)
	from sensors.cache import Cache
	Cache(
		datadict_component = 'DataDictForward',
		drone_component = 'Drone',
		sensor_name = sensor_name,
		sensor_dir = f'{observations_dir}{sensor_name}',
		map_name = airsim_map,
		transformers_components=[],
		name = sensor_name,
	)
	img_sensors.append(sensor_name)
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
		level_up_criteria = level_up_criteria, # percent of succesfull paths needed to level up
		level_up_buffer = level_up_buffer, # number of previous episodes to look at to measure level_up_criteria
		start_level = start_level,
		max_level = max_level, # can level up this many times after will terminate DRL learning loop
		terminate_on_max = False,
		frequency = frequency, # check every this many episodes
		eval_on_levelup = True, # save model everytime we level up
		model_component = 'Model',
		update_progress = True,
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
		save_config = True,
		save_log = False,
		order = 'post',
		frequency = checkpoint_freq,
		name = 'ModelingSaver',
	)


# CONNECT COMPONENTS
configuration.connect_all()

# WRITE CONFIGURATION
configuration.save()

# WRITE CONTROLLER
controller.save(working_directory + 'train_controller.json')

# RUN CONTROLLER
configuration.controller.run()

# done
speak('training complete!')
configuration.save()
evaluate_navi(
	f'{run_directory}configuration.json',
	f'{run_directory}modeling/model_final.zip',
	f'{run_directory}test_final/',
)
#print_local_log()
configuration.disconnect_all()
progress(job_name, f'complete')