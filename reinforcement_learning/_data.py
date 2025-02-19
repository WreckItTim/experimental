import sys
sys.dont_write_bytecode = True

import rl_utils as _utils
from configuration import Configuration
import numpy as np
import time
import os

instance_name, home_path, local_path, data_path, models_path = _utils.db_setup(levels_in=1)

airsim_map = sys.argv[1]
sensor_name = sys.argv[2]
save_as = sys.argv[3]
prefix = sys.argv[4]
repeat = int(sys.argv[5])
xmin = int(sys.argv[6])
xmax = int(sys.argv[7])
xint = int(sys.argv[8])
ymin = int(sys.argv[9])
ymax = int(sys.argv[10])
yint = int(sys.argv[11])
zmin = int(sys.argv[12])
zmax = int(sys.argv[13])
zint = int(sys.argv[14])
part_name = f'{prefix}_{repeat}_{xmin}_{xmax}_{xint}_{ymin}_{ymax}_{yint}_{zmin}_{zmax}_{zint}'
job_name = f'data_{airsim_map}_{sensor_name}_{save_as}_{part_name}'

##  **** SETUP ****
observations_path = f'{data_path}observations/'

# set path to precompiled airsim release binary
if airsim_map == 'Blocks':
    release_path = local_path + 'airsim_maps/LinuxBlocks1.8.1/LinuxNoEditor/Blocks.sh'
if airsim_map == 'AirSimNH':
    release_path = local_path + 'airsim_maps/AirSimNH/LinuxNoEditor/AirSimNH.sh'

clock_speed = 1

# render screen? This should be false if SSH-ing from remote (edit here or in global_parameters.json)
render_screen = False

# default weather is none (sunny) -- different sensor versions will change this
weather_type = -1
weather_degree = 0

# if turned on, airsim will get all segmentation colors and save as map
segmentation = False

# set path to read rooftops from to determine collidable surface positions
rooftops_path = 'rooftops/' + airsim_map + '.p' # match to map or use voxels if not available

# write run files to this directory
working_directory = f'{observations_path}{sensor_name}/{airsim_map}/'

# setup for run, set system vars and prepare file system
_utils.setup(working_directory, overwrite_directory=False) # WARNING: overwrite_directory will clear all old data in this folder
_utils.set_local_parameter('part_name', part_name)
_utils.set_local_parameter('job_name', job_name)

# make data controller to teleport to each specified point and capture all data from given sensors
from controllers.data import Data
controller = Data(
    drone_component = 'Drone', # drone to issue commands to move around map
    sensor_component = sensor_name, # sensor to fetch data from
    map_component = 'Map', # map to move around in
    points = [], # list of points to visit
    part_name = part_name,
    job_name = job_name,
    save_as = save_as,
)

# make configuration object to add components to
configuration = Configuration({})
configuration.set_controller(controller)


##  **** CREATE COMPONENTS ****


# ** TRANSFORMERS **


from transformers.normalize import Normalize
Normalize(
    min_input = 1, # minimum range in front of sensor
    max_input = 255, # maximum range at horizon
    min_output = 1, # minimum range in front of sensor
    max_output = 255, # minimum range in front of sensor
    left = 1, # value returned for going under min  distance
    right = 255, # value returned for going over max distance
    name = 'NormalizeDistance',
    )
# get flattened depth map (obsfucated front facing distance sensors)
from transformers.resizeflat import ResizeFlat
# airsim camera's image pixel size default is  "Width": 256, "Height": 144,
max_cols4 = [4*(i+1) for i in range(64)] # splits depth map by columns
max_rows4 = [4*(i+1) for i in range(36)] # splits depth map by rows
ResizeFlat(
    max_cols = max_cols4,
    max_rows = max_rows4,
    name = 'ResizeFlat4',
    )
from transformers.datatype import DataType
DataType(
    to_type=np.uint8,
    name = 'DataType8',
    )
from transformers.reduce import Reduce
Reduce(
    reduce_method=np.min, # [np.max, np.min, np.mean, ...]
    reduce_shape=(1,4,4), # block size [ax1, ..., axN]
    name = 'Reduce4',
)


# ** TRANSFORMERS **




# ** SENSORS **


## forward facing depth camera 256x144
if sensor_name == 'DepthV1':
    array_size = [1,144,256]
    data_type = np.uint8
    from sensors.airsimcamera import AirSimCamera
    AirSimCamera(
        airsim_component = 'Map',
        camera_view = '0', 
        image_type = 2,
        transformers_components = ['NormalizeDistance', 'DataType8'],
        name = sensor_name,
    )
    sensor_info = {
        'obs_type':'array',
        'state_type':'4vec',
        'array_size':array_size,
        'notes':'forward facing depth',
    }

## forward facing depth camera with reduced image size
if sensor_name == 'DepthV2':
    array_size = [1,36,64]
    data_type = np.uint8
    from sensors.airsimcamera import AirSimCamera
    AirSimCamera(
        airsim_component = 'Map',
        camera_view = '0',
        image_type = 2,
        transformers_components = ['Reduce4', 'NormalizeDistance', 'DataType8'],
        name = sensor_name,
        )
    sensor_info = {
        'obs_type':'array',
        'state_type':'4vec',
        'array_size':array_size,
        'notes':'forward facing depth',
    }

## downward facing depth camera with reduced image size
if sensor_name == 'DepthV3':
    array_size = [1,36,64]
    data_type = np.uint8
    from sensors.airsimcamera import AirSimCamera
    AirSimCamera(
        airsim_component = 'Map',
        camera_view = '3',
        image_type = 2,
        transformers_components = ['Reduce4', 'NormalizeDistance', 'DataType8'],
        name = sensor_name,
        )
    sensor_info = {
        'obs_type':'array',
        'state_type':'4vec',
        'array_size':array_size,
        'notes':'downward facing depth',
    }

## downward facing depth camera 256x144
if sensor_name == 'DepthV4':
    array_size = [1,144,256]
    data_type = np.uint8
    from sensors.airsimcamera import AirSimCamera
    AirSimCamera(
        airsim_component = 'Map',
        camera_view = '3', 
        image_type = 2,
        transformers_components = ['NormalizeDistance', 'DataType8'],
        name = sensor_name,
    )
    sensor_info = {
        'obs_type':'array',
        'state_type':'4vec',
        'array_size':array_size,
        'notes':'forward facing depth',
    }

## forward facing scene camera (RGB)
if sensor_name == 'SceneV1':
    array_size = [3,144,256]
    data_type = np.uint8
    from sensors.airsimcamera import AirSimCamera
    AirSimCamera(
        airsim_component = 'Map',
        camera_view = '0',
        image_type = 0,
        transformers_components = ['DataType8'],
        name = sensor_name,
        )
    sensor_info = {
        'obs_type':'array',
        'state_type':'4vec',
        'array_size':array_size,
        'notes':'forward facing scene',
    }
    
## downward facing scene camera (RGB)
if sensor_name == 'SceneV2':
    array_size = [3,144,256]
    data_type = np.uint8
    from sensors.airsimcamera import AirSimCamera
    AirSimCamera(
        airsim_component = 'Map',
        camera_view = '3',
        image_type = 0,
        transformers_components = ['DataType8'],
        name = sensor_name,
        )
    sensor_info = {
        'obs_type':'array',
        'state_type':'4vec',
        'array_size':array_size,
        'notes':'forward facing scene',
    }

## forward facing scene camera (RGB) with 100% rain
if sensor_name == 'SceneV3':
    array_size = [3,144,256]
    data_type = np.uint8
    from sensors.airsimcamera import AirSimCamera
    AirSimCamera(
        airsim_component = 'Map',
        camera_view = '0',
        image_type = 0,
        transformers_components = ['DataType8'],
        name = sensor_name,
        )
    sensor_info = {
        'obs_type':'array',
        'state_type':'4vec',
        'array_size':array_size,
        'notes':'forward facing scene with 100% rain',
    }
    weather_type = 0
    weather_degree = 1

## forward facing scene camera (RGB) with 100% snow
if sensor_name == 'SceneV4':
    array_size = [3,144,256]
    data_type = np.uint8
    from sensors.airsimcamera import AirSimCamera
    AirSimCamera(
        airsim_component = 'Map',
        camera_view = '0',
        image_type = 0,
        transformers_components = ['DataType8'],
        name = sensor_name,
        )
    sensor_info = {
        'obs_type':'array',
        'state_type':'4vec',
        'array_size':array_size,
        'notes':'forward facing scene with 100% snow',
    }
    weather_type = 2
    weather_degree = 1

## forward facing scene camera (RGB) with 100% fog
if sensor_name == 'SceneV5':
    array_size = [3,144,256]
    data_type = np.uint8
    from sensors.airsimcamera import AirSimCamera
    AirSimCamera(
        airsim_component = 'Map',
        camera_view = '0',
        image_type = 0,
        transformers_components = ['DataType8'],
        name = sensor_name,
        )
    sensor_info = {
        'obs_type':'array',
        'state_type':'4vec',
        'array_size':array_size,
        'notes':'forward facing scene with 100% fog',
    }
    weather_type = 7
    weather_degree = 1

## forward facing depth camera with reduced image size
if sensor_name == 'SegmentationV1':
    array_size = [3,144,256]
    data_type = np.float64
    from sensors.airsimcamera import AirSimCamera
    AirSimCamera(
        airsim_component = 'Map',
        camera_view = '0',
        image_type = 5,
        transformers_components = [],
        name = sensor_name,
        )
    sensor_info = {
        'obs_type':'array',
        'state_type':'4vec',
        'array_size':array_size,
        'notes':'forward facing segmentation',
    }
    segmentation = True


# write sensor meta data
sensor_path = observations_path+sensor_name+'/'
os.makedirs(sensor_path, exist_ok=True)
_utils.write_json(sensor_info, sensor_path + 'info.json')

# **** SENSORS ****




# **** ENVIRONMENT TO COLLECT DATA FROM ****

## create airsim map to query commands to
from maps.airsimmap import AirSimMap
# add any console flags
console_flags = []
if render_screen:
    console_flags.append('-Windowed')
else:
    console_flags.append('-RenderOffscreen')
# create airsim map object
AirSimMap(
    release_path = release_path,
    settings = {
        'ClockSpeed': clock_speed, # speed up, >1, or slow down, <1. For aisrim, generally don't go higher than 10 - but this is dependent on your setup
        },
    setting_files = [
        'lightweight', # see maps/airsim_settings/... for different settings
        ],
    console_flags = console_flags.copy(),
    vehicle = 'multirotor',
    segmentation = segmentation,
    remove_animals = True,
    weather_type = weather_type,
    weather_degree = weather_degree,
    name = 'Map',
)

# read rooftops data struct to determine z-height of nearest collidable object below or above x,y coords
from others.rooftops import Rooftops
rooftops = Rooftops(
    read_path = rooftops_path,
    name = 'Rooftops',
)

# create drone actor to move around map
from drones.airsimdrone import AirSimDrone
AirSimDrone(
    airsim_component = 'Map',
    name='Drone',
)
# **** ENVIRONMENT TO COLLECT DATA FROM ****


# all components created
_utils.speak('configuration created!')

# connect all components in priority order
configuration.connect_all()
_utils.speak('all components connected.')


## **** SPECIFY POINTS TO COLLECT DATA FROM ****

## specify which points to grab data from on map
# here are some dummy values as an example...
# [x, y, z, yaw] (meters, meters, meters, rads:-pi,pi]) and in drone coords: 
    # x is positive forward facing from drone spawned at origin with 0 yaw
    # y is positive to the right from drone spawned at origin with 0 yaw
    # the upwards direction is negative z, not positive
    # yaw is radians rotated clockwise along the x,y plane

x_vals = range(xmin, xmax, xint)
y_vals = range(ymin, ymax, yint)
z_vals = range(zmin, zmax, zint)
# if airsim_map in ['Blocks']:
#     x_vals = range(-120, 101, 2)
#     y_vals = range(-140, 141, 2)
# if airsim_map in ['AirSimNH']:
#     x_vals = range(-240, 241, 2)
#     y_vals = range(-240, 241, 2)
yaw_vals = [i*np.pi/2 for i in range(-1, 3)]
for r in range(repeat):
    for x in x_vals:
        for y in y_vals:
            for z in z_vals:
                for yaw in yaw_vals:
                    point = [x, y, z, yaw]
                    in_object = rooftops.in_object(x, y, z)
                    if in_object: # check if point in object
                        continue
                    controller.points.append(point)
print('n_points=', len(controller.points))

# write configuration to file (to be viewed or loaded later)
configuration.save()

# run controller to collect all data
configuration.controller.run()

# clean up shop when finished
configuration.disconnect_all()

# wait for everything to properly disconnect
time.sleep(30)
_utils.progress(job_name, 'complete')