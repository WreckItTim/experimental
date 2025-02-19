
import rl_utils as _utils
from configuration import Configuration
import os

instance_name, home_path, local_path, data_path, models_path = _utils.setup(levels_in=1)

# path to airsim release file - precompiled binary
airsim_map = 'Blocks'

# render_screen=True will render AirSim graphics to screen using Unreal engine, 
	# set to false if running from SSH or want to save resources
render_screen = False

# clock speed to run simulation at 
clock_speed = 1 # adjust this to your device (4-8 is stable on most modern devices)

# set path to airsim release
if airsim_map == 'Blocks':
	airsim_release_path = local_path + 'airsim_maps/LinuxBlocks1.8.1/LinuxNoEditor/Blocks.sh'
if airsim_map == 'AirSimNH':
	airsim_release_path = local_path + 'airsim_maps/AirSimNH/LinuxNoEditor/AirSimNH.sh'
	
# write run files to this directory
working_directory = f'{home_path}hosts/{airsim_map}/'

# path to get sensor observations from
observations_path = f'{data_path}observations/'

# path to make queries to
queries_path = f'{working_directory}queries/'


# setup for run, set system vars and prepare file system
_utils.setup(
	working_directory,
	)

# make controller to run configuration on (we will train a model)
from controllers._host import Host
controller = Host(
	drone_component = 'Drone',
	map_component = 'Map',
	map_name = airsim_map,
	read_directory = queries_path,
	write_directory = observations_path,
	)

# set meta data (anything you want here, just writes to config file as a dict)
meta = {
	'instance_name': instance_name,
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

## create map component to handle things such as simulation phsyics and graphics rendering
	# we will use Airsim here
from maps.airsimmap import AirSimMap
console_flags = [] # add any desired airsim commands here
if render_screen:
	console_flags.append('-Windowed') # render in windowed mode for more flexibility
else:
	console_flags.append('-RenderOffscreen') # do not render graphics, only handle logic in background
# create airsim map object
AirSimMap(
	release_path = airsim_release_path,
	settings = {
		'ClockSpeed': clock_speed, # reduce this value if experiencing lag
		},
	setting_files = [
		'lightweight', # see maps/airsim_settings
		],
	console_flags = console_flags.copy(),
	name = 'Map',
	)

# drone controller component - we will use AirSim
	# this can also be real world drone controller like Tello
from drones.airsimdrone import AirSimDrone
AirSimDrone(
	airsim_component = 'Map',
	name = 'Drone',
	)





# **** SENSORS ****

# TRANSFORMERS
from transformers.normalize import Normalize
Normalize(
	min_input = 1, # minimum range in front of sensor
	max_input = 255, # maximum range at horizon
	left = 0, # value returned for missing/errorneous observation
	name = 'NormalizeDistanceV1',
	)
# get flattened depth map (obsfucated front facing distance sensors)
from transformers.resizeflat import ResizeFlat
# airsim camera's image pixel size default is  "Width": 256, "Height": 144,
max_cols_v1 = [32*(i+1) for i in range(8)] # splits depth map by columns
max_rows_v1 = [24*(i+1) for i in range(6)] # splits depth map by rows
ResizeFlat(
	max_cols = max_cols_v1,
	max_rows = max_rows_v1,
	name = 'ResizeFlatV1',
	)
max_cols_v2 = [16*(i+1) for i in range(16)] # splits depth map by columns
max_rows_v2 = [12*(i+1) for i in range(12)] # splits depth map by rows
ResizeFlat(
	max_cols = max_cols_v2,
	max_rows = max_cols_v2,
	name = 'ResizeFlatV2',
	)


## flattened forward facing depth map 256x144 to flattened 8x6
from sensors.airsimcamera import AirSimCamera
sensor_name = 'FlattenedDepthV1'
AirSimCamera(
	airsim_component = 'Map',
	transformers_components = ['ResizeFlatV1', 'NormalizeDistanceV1'],
	name = sensor_name,
	)
sensor_info = {
	'obs_type':'vector',
	'state_type':'4vec',
	'vector_length':len(max_cols_v1)*len(max_rows_v1),
}
sensor_path = observations_path+sensor_name+'/'
os.makedirs(sensor_path, exist_ok=True)
_utils.pk_write(sensor_info, sensor_path + 'info.p')


## flattened forward facing depth map 256x144 to flattened 16x12
from sensors.airsimcamera import AirSimCamera
sensor_name = 'FlattenedDepthV2'
AirSimCamera(
	airsim_component = 'Map',
	transformers_components = ['ResizeFlatV2', 'NormalizeDistanceV1'],
	name = sensor_name,
	)
sensor_info = {
	'obs_type':'vector',
	'state_type':'4vec',
	'vector_length':len(max_cols_v2)*len(max_rows_v2),
}
sensor_path = observations_path+sensor_name+'/'
os.makedirs(sensor_path, exist_ok=True)
_utils.pk_write(sensor_info, sensor_path + 'info.p')


## forward facing depth camera 256x144
from sensors.airsimcamera import AirSimCamera
sensor_name = 'DepthMapV1'
AirSimCamera(
    airsim_component = 'Map',
    image_type = 2,
    name = sensor_name,
)
sensor_info = {
	'obs_type':'array',
	'state_type':'4vec',
	'array_size':[256,144],
}
sensor_path = observations_path+sensor_name+'/'
os.makedirs(sensor_path, exist_ok=True)
_utils.pk_write(sensor_info, sensor_path + 'info.p')


# **** SENSORS ****




# CONNECT COMPONENTS
configuration.connect_all()

# WRITE CONFIGURATION
configuration.save()

print('wd', working_directory)

# RUN CONTROLLER
configuration.controller.run()

# done
configuration.controller.stop()
