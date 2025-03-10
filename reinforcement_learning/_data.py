# ** insert this head at top of all main files with your proper paths **
home_dir = '/home/tim/Dropbox/' # home directory of repo
local_dir = '/home/tim/local/' # any local files outside of repo
progress_dir = f'{local_dir}progress/'
airsim_maps_dir = f'{local_dir}airsim_maps/'
data_dir = f'{home_dir}data/'
observations_dir = f'{data_dir}observations/'
rooftops_dir = f'{data_dir}rooftops/'
astar_dir = f'{data_dir}astar_paths/'
import sys
sys.path.append(home_dir)
from global_methods import *
set_global_parameters({'local_dir':local_dir, 'progress_dir':progress_dir, 'airsim_maps_dir':airsim_maps_dir, 'home_dir':home_dir, 
                       'data_dir':data_dir, 'observations_dir':observations_dir, 'rooftops_dir':rooftops_dir, 'astar_dir':astar_dir})

# local imports
from configuration import Configuration

# needed params 
airsim_map = 'null'
sensor_name = 'null'
job_name = f'null_{random.randint(0, 1_000_000)}'
xmin, xmax, xint, ymin, ymax, yint, zmin, zmax, zint = 0, 4, 2, 0, 4, 2, -4, -4, 4 

# default parametrs (can be overwritten by argv)
clock_speed = 1
render_screen = False # default to render off screen
weather_type, weather_degree = -1, 0 # -1, 0 is sunny
overwrite_directory = False # True will erase all files at otuput working directory
rooftops_version = 'v1'
id_name = 'alpha'
remove_animals = True
vehicle = 'multirotor'
context = 'airsim'
save_as = 'dict'

# read params from command line
if len(sys.argv) > 1:
    arguments = parse_arguments(sys.argv[1:])
    locals().update(arguments)
set_global_parameter('job_name', job_name)
set_global_parameter('context', context)

# set variable subpaths from root directories and params set above
if airsim_map == 'Blocks':
    release_path = f'{airsim_maps_dir}LinuxBlocks1.8.1/LinuxNoEditor/Blocks.sh'
if airsim_map == 'AirSimNH':
    release_path = f'{airsim_maps_dir}AirSimNH/LinuxNoEditor/AirSimNH.sh'
rooftops_path = f'{rooftops_dir}{rooftops_version}/' + airsim_map + '.p' # match to map or use voxels if not available
working_directory = f'{observations_dir}{sensor_name}/{airsim_map}/' # write run files to this directory

# setup for run, set system vars and prepare file system
setup(working_directory, overwrite_directory) # WARNING: overwrite_directory=True will clear all old data in this folder

# make data controller to teleport to each specified point and capture all data from given sensors
from controllers.data import Data
part_name = f'{id_name}_{xmin}_{xmax}_{xint}_{ymin}_{ymax}_{yint}_{zmin}_{zmax}_{zint}'
set_global_parameter('part_name', part_name)
controller = Data(
    drone_component = 'Drone', # drone to issue commands to move around map
    sensor_component = sensor_name, # sensor to fetch data from
    map_component = 'Map', # map to move around in
    points = [], # list of points to visit
    part_name = part_name,
    save_as = save_as,
)

# make configuration object to add components to
configuration = Configuration({})
configuration.set_controller(controller)


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

segmentation = False

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
        'notes':'downward facing scene',
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
sensor_path = f'{observations_dir}{sensor_name}/'
os.makedirs(sensor_path, exist_ok=True)
write_json(sensor_info, sensor_path + 'info.json')

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
    vehicle = vehicle,
    segmentation = segmentation,
    remove_animals = remove_animals,
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
speak('configuration created!')

# connect all components in priority order
configuration.connect_all()
speak('all components connected.')


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
#configuration.save()

# run controller to collect all data
configuration.controller.run()

# clean up shop when finished
configuration.disconnect_all()

# wait for everything to properly disconnect
time.sleep(60)
progress(job_name, 'complete')