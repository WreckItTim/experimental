import utils.global_methods as gm
from configuration import Configuration
import math
import os

# v1 modules used in SmartDepth paper (similar to NaviSlim and NaviSplit)

def continue_training_v1(output_dir):
	# load configuration file and create object to save and connect components
	read_configuration_path = output_dir + 'configuration.json'
	configuration = Configuration.load(read_configuration_path)
	meta = configuration.meta
	meta['continued_training'] = True
	# read model and/or replay buffer
	# get highest level complete
	modeling_dir = f'{output_dir}modeling/'
	fnames = os.listdir(modeling_dir)
	model_path = None
	buffer_path = f'{modeling_dir}replay_buffer.zip'
	if not os.path.exists(buffer_path):
		buffer_path = None
	model_path = f'{modeling_dir}model.zip'
	highest_level = -1
	for fname in fnames:
		if 'model' in fname and 'level' in fname:
			parts = fname.split('.')[0].split('_')
			level = int(parts[2])
			if level > highest_level:
				highest_level = level
				model_path = f'{modeling_dir}{fname}.zip'
	if not os.path.exists(model_path):
		model_path = None
	model_component = configuration.get_component('Model')
	model_component.read_model_path = model_path
	model_component.read_replay_buffer_path = buffer_path
	gm.speak('continuing training...')
	return configuration

def learning_v1(buffer_size, exploration_fraction, data_path, airsim_map, motion, start_level, max_level):
	# Model to train
	from sb3models.dqn import DQN
	DQN(
		environment_component = 'Environment',
		policy = 'MultiInputPolicy', # input 2D space to CNN feature extractor and 1D space to flattened layers
		buffer_size = buffer_size, # number of observations to store in buffer for supervised training element
		device = 'cuda:0', # change pytorch device to cuda:x or cpu here
		exploration_fraction = exploration_fraction, # what percent of total_steps to randomly sample actions
		name = 'Model',
	)

	# SPAWNER
		# moves drone to desired starting location
		# sets the target goal since it is typically dependent on the start location
	from spawners.levels import Levels
	Levels(
		drone_component = 'Drone',
		levels_path = f'{data_path}astar_paths/v1/{airsim_map}_{motion}_train.p', # paths to randomly sample from during training
		start_level = start_level,
		max_level = max_level,
		random_path = True,
		name = 'Spawner',
	)

	## MODIFIERS
		# modifiers are like wrappers, and will add functionality before or after any component
	
	# CURRICULUM LEARNING
		# this modifier will be called at the end of every training episode to see the percent of succesfull paths
		# if enough paths were succesfull then this will level up to harder goal (based on the start_level, max_level for astar paths)
	from modifiers.curriculum import Curriculum
	Curriculum(
		base_component = 'Environment', # Modifiers will execute after/before this given component is used
		parent_method = 'end', # Modifiers will execute after/before this given component's method is used
		order = 'post', # Modifiers will run code either before (pre) or after (post) given component call
		spawner_component = 'Spawner', # which component to level up when enough episodes are succesfull
		start_level = start_level, # level to start at episode 0
		max_level = max_level, # can level up this many times after will terminate DRL learning loop
		eval_on_levelup = True,
		model_component='Model',
		name = 'Curriculum',
	)

def environment_v1(x_bounds, y_bounds, z_bounds, data_path, airsim_map):

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

	# read rooftops data struct to determine z-height of nearest collidable object below or above x,y coords
	from others.rooftops import Rooftops
	Rooftops(
		read_path = f'{data_path}rooftops/v1/{airsim_map}.p',
		name = 'Rooftops',
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

	# drone controller component - we will use AirSim
		# this can also be real world drone controller like Tello
	from drones.etherial import Etherial
	Etherial(
		map_component = 'Map',
		name = 'Drone',
		)

def observations_v1(observations_path, forward_sensor, x_bounds, y_bounds, z_bounds, airsim_map, motion, nPast=4):
	sensor_info = gm.read_json(f'{observations_path}{forward_sensor}/info.json')
	image_bands, image_height, image_width = sensor_info['array_size']
	# set parameters for data to fetch
	id_name = 'alpha' # when reading in observation data, which ID key words to use
	map_resolution_x = 2
	map_resolution_y = 2
	map_resolution_z = 4
	x_vals = [x for x in range(x_bounds[0], x_bounds[1]+1, map_resolution_x)]
	y_vals = [y for y in range(y_bounds[0], y_bounds[1]+1, map_resolution_y)]
	z_vals = [z for z in range(z_bounds[0], z_bounds[1]+1, map_resolution_z)]
	yaw_vals = [0, 1, 2, 3] # what yaws are accessible by drone (for mode='data' only)

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
		data_dir = f'{observations_path}{forward_sensor}/{airsim_map}/',
		id_name=id_name, x_vals=x_vals, y_vals=y_vals, z_vals=z_vals, yaw_vals=yaw_vals,
		name = 'DataDictForward',
	)
	from sensors.cache import Cache
	Cache(
		datadict_component = 'DataDictForward',
		drone_component = 'Drone',
		sensor_name = forward_sensor,
		map_name = airsim_map,
		observations_path = observations_path,
		queries_path = None,
		transformers_components=[],
		name = forward_sensor,
	)
	img_sensors.append(forward_sensor)
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


# discrete actions and move left or right by 90 deg, move forward at either 2, 4, 8, 16, 32 meters, or move up/down by 4 meters
def actions_v1(motion):
	## ACTION SPACE
	actions = []
	# we will just move forward and rotate for this example
	from actions.fixedrotate import FixedRotate 
	FixedRotate(
		drone_component = 'Drone',  
		yaw_diff = math.pi/2, # can rotate at 90 deg increments
		name = 'RotateRight90',
		)
	actions.append('RotateRight90')
	FixedRotate(
		drone_component = 'Drone',  
		yaw_diff = -1*math.pi/2, # can rotate at 90 deg increments
		name = 'RotateLeft90',
		)
	actions.append('RotateLeft90')
	from actions.fixedmove import FixedMove
	FixedMove(
		drone_component = 'Drone', 
		x_distance = 2,
		adjust_for_yaw = True, # this adjusts movement based on current yaw for relative moves
		name = 'MoveForward2',
		)
	actions.append('MoveForward2')
	FixedMove(
		drone_component = 'Drone', 
		x_distance = 4,
		adjust_for_yaw = True, # this adjusts movement based on current yaw for relative moves
		name = 'MoveForward4',
		)
	actions.append('MoveForward4')
	FixedMove(
		drone_component = 'Drone', 
		x_distance = 8,
		adjust_for_yaw = True, # this adjusts movement based on current yaw for relative moves
		name = 'MoveForward8',
		)
	actions.append('MoveForward8')
	FixedMove(
		drone_component = 'Drone', 
		x_distance = 16,
		adjust_for_yaw = True, # this adjusts movement based on current yaw for relative moves
		name = 'MoveForward16',
		)
	actions.append('MoveForward16')
	FixedMove(
		drone_component = 'Drone', 
		x_distance = 32,
		adjust_for_yaw = True, # this adjusts movement based on current yaw for relative moves
		name = 'MoveForward32',
		)
	actions.append('MoveForward32')
	# if vertical motion allowed
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
	# alternatively use default discrete or continuous Actor class
	TeleporterDiscrete(
		drone_component = 'Drone',
		actions_components = actions,
		discretize=True,
		name = 'Actor',
		)
	

def bounds_v1(airsim_map, motion, region='all'):
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
	return x_bounds, y_bounds, z_bounds

def rewards_v1(x_bounds, y_bounds, z_bounds, motion):
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
		astar_multiplier = 2, # max step size is this many times the astar length
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