# ** insert this head at top of all main files with your proper paths **
local_dir = '/home/tim/local/' # any local files outside of repo
home_dir = '/home/tim/Dropbox/experimental/' # home directory of repo
temp_dir = f'{local_dir}temp/'
maps_dir = f'{local_dir}airsim_maps/'
import os
os.chdir(home_dir)
import sys
sys.path.append(home_dir)
import utils.global_methods as gm
import map_data.map_methods as mm
import reinforcement_learning.reinforcement_methods as rm

# local imports
from configuration import Configuration
from controllers.data import Data
import random
import numpy as np
import time
initial_locals = locals().copy()

# needed params 
map_name = 'null'
sensor_name = 'null'
job_name = f'null_{random.randint(0, 1_000_000)}'
xmin, xmax, xint, ymin, ymax, yint, zmin, zmax, zint = 0, 4, 2, 0, 4, 2, 4, 8, 4 

# default parametrs (can be overwritten by argv)
clock_speed = 1
render_screen = False # default to render off screen
weather_type, weather_degree = -1, 0 # -1, 0 is sunny
overwrite_directory = False # True will erase all files at otuput working directory
rooftops_version = 'v1'
remove_animals = True
vehicle = 'multirotor'
save_as = 'dict'
id_name = 'alpha'

# read params from command line
if len(sys.argv) > 1:
    arguments = gm.parse_arguments(sys.argv[1:])
    locals().update(arguments)
gm.set_global('job_name', job_name)
gm.set_global('home_dir', home_dir)
gm.set_global('local_dir', local_dir)
gm.set_global('temp_dir', temp_dir)

# set variable subpaths from root directories and params set above
if map_name == 'Blocks':
    release_path = f'{maps_dir}LinuxBlocks1.8.1/LinuxNoEditor/Blocks.sh'
if map_name == 'AirSimNH':
    release_path = f'{maps_dir}AirSimNH/LinuxNoEditor/AirSimNH.sh'
datamap = mm.DataMap(map_name, rooftops_version)
gm.set_global('datamap', datamap)
output_dir = f'{home_dir}map_data/observations/{sensor_name}/{map_name}/' # write run files to this directory
part_name = f'{id_name}_{xmin}_{xmax}_{xint}_{ymin}_{ymax}_{yint}_{zmin}_{zmax}_{zint}'
complete_path = f'{output_dir}completed__{part_name}.p'
gm.set_global('complete_path', complete_path)

# how to handle if completed path already exists (showing a previous job has finished this data collection already)
if os.path.exists(complete_path):
    if overwrite_directory:
        os.remove(complete_path)
    else:
        gm.progress(job_name, 'complete')
        sys.exit()


# setup for run, set system vars and prepare file system
gm.setup_output_dir(output_dir, overwrite_directory)

# make data controller to teleport to each specified point and capture all data from given sensors
gm.set_global('part_name', part_name)
controller = Data(
    drone_component = 'Drone', # drone to issue commands to move around map
    sensor_component = sensor_name, # sensor to fetch data from
    map_component = 'Map', # map to move around in
    points = [], # list of points to visit
    part_name = part_name,
    save_as = save_as,
)

# all variables here will be added to configuration parameters for reading later
all_local_vars = locals()
user_local_vars = {k:v for k, v in all_local_vars.items() if (not k.startswith('__') and k not in initial_locals and k not in ['initial_locals','all_local_vars'])}
config_params = user_local_vars.copy()

# make configuration object to add components to
configuration = Configuration({})
configuration.set_controller(controller)
for key in config_params:
    configuration.set_parameter(key, config_params[key])

##  **** CREATE COMPONENTS ****
# note that all of the transformers and sensors are coded and registered here


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
camera_height = 144
camera_width = 256
image_type = 0

segmentation = False

## forward facing depth camera 144x256
if sensor_name == 'DepthV1':
    image_type = 2
    array_size = [1,camera_height,camera_width]
    data_type = np.uint8
    from sensors.airsimcamera import AirSimCamera
    AirSimCamera(
        airsim_component = 'Map',
        camera_view = '0', 
        image_type = image_type,
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
    image_type = 2
    camera_height = 72
    camera_width = 128
    array_size = [1,camera_height,camera_width]
    data_type = np.uint8
    from sensors.airsimcamera import AirSimCamera
    AirSimCamera(
        airsim_component = 'Map',
        camera_view = '0',
        image_type = image_type,
        transformers_components = ['NormalizeDistance', 'DataType8'],
        name = sensor_name,
        )
    sensor_info = {
        'obs_type':'array',
        'state_type':'4vec',
        'array_size':array_size,
        'notes':'forward facing depth',
    }

## forward facing depth camera with further reduced image size
if sensor_name == 'DepthV3':
    image_type = 2
    camera_height = 36
    camera_width = 64
    array_size = [1,camera_height,camera_width]
    data_type = np.uint8
    from sensors.airsimcamera import AirSimCamera
    AirSimCamera(
        airsim_component = 'Map',
        camera_view = '0',
        image_type = image_type,
        transformers_components = ['NormalizeDistance', 'DataType8'],
        name = sensor_name,
        )
    sensor_info = {
        'obs_type':'array',
        'state_type':'4vec',
        'array_size':array_size,
        'notes':'downward facing depth 36x64',
    }

## downward facing depth camera 144x256
if sensor_name == 'DepthV4':
    image_type = 2
    array_size = [1,camera_height,camera_width]
    data_type = np.uint8
    from sensors.airsimcamera import AirSimCamera
    AirSimCamera(
        airsim_component = 'Map',
        camera_view = '3', 
        image_type = image_type,
        transformers_components = ['NormalizeDistance', 'DataType8'],
        name = sensor_name,
    )
    sensor_info = {
        'obs_type':'array',
        'state_type':'4vec',
        'array_size':array_size,
        'notes':'downward facing depth 144x256',
    }

## forward facing depth camera with increased image size
if sensor_name == 'DepthV5':
    image_type = 2
    camera_height = 288
    camera_width = 512
    array_size = [1,camera_height,camera_width]
    data_type = np.uint8
    from sensors.airsimcamera import AirSimCamera
    AirSimCamera(
        airsim_component = 'Map',
        camera_view = '0', 
        image_type = image_type,
        transformers_components = ['NormalizeDistance', 'DataType8'],
        name = sensor_name,
    )
    sensor_info = {
        'obs_type':'array',
        'state_type':'4vec',
        'array_size':array_size,
        'notes':'forward facing depth 288x512',
    }

## forward facing depth camera with greatly increased image size
if sensor_name == 'DepthV6':
    image_type = 2
    camera_height = 576
    camera_width = 1024
    array_size = [1,camera_height,camera_width]
    data_type = np.uint8
    from sensors.airsimcamera import AirSimCamera
    AirSimCamera(
        airsim_component = 'Map',
        camera_view = '0', 
        image_type = image_type,
        transformers_components = ['NormalizeDistance', 'DataType8'],
        name = sensor_name,
    )
    sensor_info = {
        'obs_type':'array',
        'state_type':'4vec',
        'array_size':array_size,
        'notes':'forward facing depth 576x1024',
    }

## forward facing scene camera (RGB)
if sensor_name == 'SceneV1':
    array_size = [3,camera_height,camera_width]
    data_type = np.uint8
    from sensors.airsimcamera import AirSimCamera
    AirSimCamera(
        airsim_component = 'Map',
        camera_view = '0',
        image_type = image_type,
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
    array_size = [3,camera_height,camera_width]
    data_type = np.uint8
    from sensors.airsimcamera import AirSimCamera
    AirSimCamera(
        airsim_component = 'Map',
        camera_view = '3',
        image_type = image_type,
        transformers_components = ['DataType8'],
        name = sensor_name,
        )
    sensor_info = {
        'obs_type':'array',
        'state_type':'4vec',
        'array_size':array_size,
        'notes':'downward facing scene',
    }

## forward facing scene camera (RGB) with 100% rain
if sensor_name == 'SceneV3':
    array_size = [3,camera_height,camera_width]
    data_type = np.uint8
    from sensors.airsimcamera import AirSimCamera
    AirSimCamera(
        airsim_component = 'Map',
        camera_view = '0',
        image_type = image_type,
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
    array_size = [3,camera_height,camera_width]
    data_type = np.uint8
    from sensors.airsimcamera import AirSimCamera
    AirSimCamera(
        airsim_component = 'Map',
        camera_view = '0',
        image_type = image_type,
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
    array_size = [3,camera_height,camera_width]
    data_type = np.uint8
    from sensors.airsimcamera import AirSimCamera
    AirSimCamera(
        airsim_component = 'Map',
        camera_view = '0',
        image_type = image_type,
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
    image_type = 5
    array_size = [3,camera_height,camera_width]
    data_type = np.float64
    from sensors.airsimcamera import AirSimCamera
    AirSimCamera(
        airsim_component = 'Map',
        camera_view = '0',
        image_type = image_type,
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
sensor_dir = f'{home_dir}map_data/observations/{sensor_name}/'
info_path = f'{sensor_dir}info.json'
if not os.path.exists(info_path):
    os.makedirs(sensor_dir, exist_ok=True)
    gm.write_json(sensor_info, info_path)

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
        "CameraDefaults": {
            "CaptureSettings": [
                {
                "ImageType": image_type,
                "Width": camera_width,
                "Height": camera_height,
                "FOV_Degrees": 90,
                "AutoExposureSpeed": 100,
                "MotionBlurAmount": 0
                }
            ]
        },
    },
    setting_file_paths = [
        f'{home_dir}map_data/airsim_settings/lightweight.json',
    ],
    console_flags = console_flags.copy(),
    vehicle = vehicle,
    segmentation = segmentation,
    remove_animals = remove_animals,
    weather_type = weather_type,
    weather_degree = weather_degree,
    name = 'Map',
)

# create drone actor to move around map
from drones.airsimdrone import AirSimDrone
AirSimDrone(
    airsim_component = 'Map',
    name='Drone',
)
# **** ENVIRONMENT TO COLLECT DATA FROM ****


# all components created
gm.speak('configuration created!')

# connect all components in priority order
configuration.connect_all()
gm.speak('all components connected.')


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
# if map_name in ['Blocks']:
#     x_vals = range(-120, 101, 2)
#     y_vals = range(-140, 141, 2)
# if map_name in ['AirSimNH']:
#     x_vals = range(-240, 241, 2)
#     y_vals = range(-240, 241, 2)
yaw_vals = [i*np.pi/2 for i in range(-1, 3)]
for x in x_vals:
    for y in y_vals:
        for z in z_vals:
            for yaw in yaw_vals:
                point = [x, y, z, yaw]
                in_object = datamap.in_object(x, y, z)
                if in_object: # check if point in object
                    continue
                controller.points.append(point)
gm.speak(f'n_points={len(controller.points)}')

# write configuration to file (to be viewed or loaded later)
#configuration.save()

# run controller to collect all data
configuration.controller.run()

# clean up shop when finished
configuration.disconnect_all()

# wait for everything to properly disconnect
time.sleep(20)
gm.progress(job_name, 'complete')
gm.speak('completed')