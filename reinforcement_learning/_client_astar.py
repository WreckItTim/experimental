import sys
sys.dont_write_bytecode = True

import rl_utils as _utils
from configuration import Configuration
import os
import pickle
import numpy as np

import math

#### SETUP
instance_name, home_path, local_path, data_path, models_path = _utils.setup(levels_in=1)

# parameters
project_name = 'Navi'
experiment_name = 'DQN_v3'
version = 'v0'
run_name = sys.argv[1]
airsim_map = sys.argv[2]
motion = sys.argv[3]
astar_region = sys.argv[4]
device = sys.argv[5]
trial_name = f'astar_{airsim_map}_{motion}_{astar_region}_{version}'
nPast = 1 # past
nFuture = 0
num_evals_per_sublevel = 1
forward_sensor = 'DepthV2'
downward_sensor = 'DepthV3'

# have we finished this test yet?
working_directory = models_path + '/'.join([project_name, experiment_name, trial_name, run_name])+'/'
test_name = '_'.join([project_name, experiment_name, trial_name, run_name])
overwrite = True
if not overwrite and os.path.exists(working_directory + 'results.json'):
    print('job already complete')
    _utils.progress(test_name, f'complete')
    sys.exit(1)

# file IO
astar_dir = f'{data_path}paths_reduced/' # 'astar_paths/' f'{data_path}paths/'
astar_paths_path = f'{astar_dir}{airsim_map}_{motion}_{astar_region}.p'
rooftops_path = f'{data_path}rooftops/{airsim_map}.p' # match to map or use voxels if not available
observations_path = f'{data_path}observations/'
queries_path = None
# assuming forward and downward sensor dimensions are same
sensor_info = _utils.read_json(f'{observations_path}{forward_sensor}/info.json')
image_bands, image_height, image_width = sensor_info['array_size']

# astart paths to test on
astar_paths = _utils.pk_read(astar_paths_path)
start_level, max_level = 0, len(astar_paths['levels'])-1 # range of levels to test on from astar paths
num_sublevels = np.sum([len(astar_paths['levels'][level]) for level in range(start_level,max_level+1)])
num_episodes = int(num_evals_per_sublevel * num_sublevels)

# bounds drone can move in
bounds_region = 'all'
z_bounds = [-16, 0]
if airsim_map in ['AirSimNH']:
    if bounds_region in ['all']:
        x_bounds = [-240, 240]
        y_bounds = [-240, 240]
    elif bounds_region in ['train']:
        x_bounds = [-10, 240]
        y_bounds = [-240, 240]
    elif bounds_region in ['test']:
        x_bounds = [-240, 10]
        y_bounds = [-240, 240]
if airsim_map in ['Blocks']:
    if bounds_region in ['all']:
        x_bounds = [-120, 100]
        y_bounds = [-140, 140]
    elif bounds_region in ['train']:
        x_bounds = [-60, 100]
        y_bounds = [-140, 140]
    elif bounds_region in ['test']:
        x_bounds = [-120, 0]
        y_bounds = [-140, 140]
        
# setup for run, set system vars and prepare file system
overwrite_directory = True # True False
_utils.setup(
    working_directory, 
    project_name=project_name,
    experiment_name=experiment_name,
    trial_name=trial_name,
    run_name=run_name,
    overwrite_directory=overwrite_directory,
    )


#### CONFIGURATION

# CONTROLLER
from controllers.test import Test
controller = Test(
        environment_component = 'Environment', # environment to run test in
        model_component = 'Model', # used to make predictions
        results_directory = working_directory,
        num_episodes = num_episodes,
    )
# set meta data (anything you want here, just writes to config file as a dict)
meta = {
    'instance_name': instance_name,
    'project_name' : project_name,
    'experiment_name' : experiment_name,
    'trial_name' : trial_name,
    'run_name' : run_name,
    }
# make a new configuration file to add components to 
configuration = Configuration(
    meta, 
    controller,
    )

# ENVIRONMENT
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

# MAP
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

# DRONE
from drones._etherial import _Etherial
_Etherial(
    map_component = 'Map',
    name = 'Drone',
    )

## REWARD FUNCTION
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
    name = 'MaxStepsReward',
    update_steps = True, # update the maximum number of steps based on distance to goal
    max_steps = 4, # base number of steps, will scale with further goal
    max_max = 50, # absolute max number of steps, regardless of scaling from further goals
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
actions = []
from actions.fixedrotate import FixedRotate 
FixedRotate(
    drone_component = 'Drone',  
    yaw_diff = math.pi/2, # can rotate at 90 deg increments
    name = 'RotateRight',
    )
actions.append('RotateRight')
FixedRotate(
    drone_component = 'Drone',  
    yaw_diff = -1*math.pi/2, # can rotate at 90 deg increments
    name = 'RotateLeft',
    )
actions.append('RotateLeft')
from actions.fixedmove import FixedMove
FixedMove(
    drone_component = 'Drone', 
    x_distance = 2, # can move forward up to 10 meters
    adjust_for_yaw = True, # this adjusts movement based on current yaw
    name = 'MoveForward2',
    )
actions.append('MoveForward2')
FixedMove(
    drone_component = 'Drone', 
    x_distance = 4, # can move forward up to 10 meters
    adjust_for_yaw = True, # this adjusts movement based on current yaw
    name = 'MoveForward4',
    )
actions.append('MoveForward4')
FixedMove(
    drone_component = 'Drone', 
    x_distance = 8, # can move forward up to 10 meters
    adjust_for_yaw = True, # this adjusts movement based on current yaw
    name = 'MoveForward8',
    )
actions.append('MoveForward8')
FixedMove(
    drone_component = 'Drone', 
    x_distance = 16, # can move forward up to 10 meters
    adjust_for_yaw = True, # this adjusts movement based on current yaw
    name = 'MoveForward16',
    )
actions.append('MoveForward16')
FixedMove(
    drone_component = 'Drone', 
    x_distance = 32, # can move forward up to 10 meters
    adjust_for_yaw = True, # this adjusts movement based on current yaw
    name = 'MoveForward32',
    )
actions.append('MoveForward32')
if motion in ['3d']:
    FixedMove(
        drone_component = 'Drone', 
        z_distance = 4, # can move forward up to 10 meters
        adjust_for_yaw = True, # this adjusts movement based on current yaw
        name = 'MoveDownward4',
        )
    actions.append('MoveDownward4')
    FixedMove(
        drone_component = 'Drone', 
        z_distance = -4, # can move forward up to 10 meters
        adjust_for_yaw = True, # this adjusts movement based on current yaw
        name = 'MoveUpward4',
        )
    actions.append('MoveUpward4')
# ACTOR
from actors.teleporterdiscrete import TeleporterDiscrete
TeleporterDiscrete(
    drone_component = 'Drone',
    actions_components = actions,
    discretize=True,
    name = 'Actor',
    )

## OBSERVATION SPACE
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
    datapath = f'{observations_path}{forward_sensor}/{airsim_map}/data_dict.p',
    name = 'DataDictForward',
)
from sensors._client import _Client
_Client(
    datadict_component = 'DataDictForward',
    drone_component = 'Drone',
    sensor_name = forward_sensor,
    map_name = airsim_map,
    observations_path = observations_path,
    queries_path = queries_path,
    future = 0,
    transformers_components=[
    ],
    name = forward_sensor,
)
img_sensors.append(forward_sensor)
image_bands_remember += image_bands
# downward depth sensor
if motion in '3d':
    DataDict(
        parent_directory = f'{observations_path}{downward_sensor}/{airsim_map}/',
        name = 'DataDictDownward',
    )
    _Client(
        datadict_component = 'DataDictDownward',
        drone_component = 'Drone',
        sensor_name = downward_sensor,
        map_name = airsim_map,
        observations_path = observations_path,
        queries_path = queries_path,
        future = 0,
        transformers_components=[
        ],
        name = downward_sensor,
    )
    img_sensors.append(downward_sensor)
    image_bands_remember += image_bands
# future sensors
if nFuture > 0:
    # future forward depth sensor
    for action in actions:
        _Client(
            datadict_component = 'DataDictForward',
            drone_component='Drone',
            sensor_name = forward_sensor,
            map_name = airsim_map,
            observations_path = observations_path,
            queries_path = queries_path,
            future = nFuture,
            action_component = action,
            remember = False,
            transformers_components=[
            ],
            name = 'ForwardFuture' + action,
            )
        img_sensors.append('ForwardFuture' + action)
        image_bands_forget += image_bands
    # future downward depth sensor
    if motion in '3d':
        for action in actions:
            _Client(
                datadict_component = 'DataDictDownward',
                drone_component='Drone',
                sensor_name = downward_sensor,
                map_name = airsim_map,
                observations_path = observations_path,
                queries_path = queries_path,
                future = nFuture,
                action_component = action,
                remember = False,
                transformers_components=[
                ],
                name = 'DownwardFuture' + action,
                )
            img_sensors.append('DownwardFuture' + action)
            image_bands_forget += image_bands
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
# sense distance to boundary
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

# MODEL
from custommodels.astaraction import AstarAction
AstarAction(
    spawner_component = 'Spawner',
    name = 'Model',
)

# SPAWNER
from spawners.levels import Levels
Levels(
    drone_component = 'Drone',
    levels_path = astar_paths_path,
    start_level = start_level,
    max_level = max_level,
    paths_per_sublevel = num_evals_per_sublevel,
    name = 'Spawner',
)

# SAVER
from modifiers.saver import Saver
# save Train states and observations
Saver(
    base_component = 'Environment',
    parent_method = 'end',
    track_vars = [
                'observations', 
                'states',
                ],
    write_folder = working_directory + 'states/',
    order = 'post',
    save_config = False,
    save_benchmarks = False,
    frequency = num_episodes,
    name='Saver',
)
    
# CONNECT new and old COMPONENTS
configuration.connect_all()

# WRITE combined new+old CONFIGURATION
configuration.save()

# RUN CONTROLLER
configuration.controller.run()

# DISCONNECT COMPONENTS
configuration.disconnect_all()
