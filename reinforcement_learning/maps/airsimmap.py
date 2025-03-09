from maps.map import Map
import global_methods as md
import subprocess
import os
from component import _init_wrapper
import setup_path # need this in same directory as python code for airsim
import airsim
from others.voxels import Voxels
import psutil
import numpy as np
import time
from matplotlib import pyplot as plt

# handles airsim release executables
class AirSimMap(Map):

	# weather was Zixia Xia's contribution
	# weather_type values:
		# Normal (sunny): -1,
		# Rain: 0,
		# Roadwetness: 1,
		# Snow: 2,
		# RoadSnow: 3,
		# MapleLeaf: 4,
		# RoadLeaf: 5,
		# Dust: 6,
		# Fog: 7
	# constructor, pass in a dictionary for settings and/or file paths to merge multiple settings .json files
	@_init_wrapper
	def __init__(self,
					# tells the height of collidable objects
					rooftops_component=None,
					# voxels for 2d/3d numpy array represtation of objects
					voxels_component=None,
					# path to release (.sh/.exe) file to be launched
					# if this is not None, will launch airsim map automatically
					# otherwise it is up to the user to launch on their own
					release_path:str = None,
					# can define json-structured settings
					settings:dict = None,
					# or define setting files to read in from given directory
					# will aggregate passed in json settings and all files
					# update priority is given to the settings argument and last listed files
					# amoung the settings must be information for which sensors to use
					# below arg is a list of file names, see the maps/airsim_settings for examples
					settings_directory:str = 'maps/airsim_settings/',
					setting_files:list = ['vanilla'],
					# optional flags to put in command line when launching
					console_flags = None,
					remove_animals = True,
					segmentation = False,
					vehicle = 'multirotor',
					weather_type = -1,
					weather_degree = 1,
				 ):
		super().__init__()
		if voxels_component is None:
			self._voxels = None
		self._pid = None
		self._client = None
		# get path to release executable file to launch
		if release_path is not None:
			# create setting dictionary
			self.settings = {}
			if settings is not None:
				self.settings = settings.copy()
			# read in any other settings files
			other_settings = {}
			if setting_files is not None:
				other_settings = self.read_settings(settings_directory, setting_files)
			# merge all settings
			self.settings.update(other_settings)
			# write to temp file to be read in when launching realease executable
			self._settings_path = os.getcwd() + '/temp/overwrite_settings.json'
			self.write_settings(self.settings, self._settings_path)
			if 'LocalHostIp' in self.settings:
				md.set_global_parameter('LocalHostIp', self.settings['LocalHostIp'])
			else:
				md.set_global_parameter('LocalHostIp', '127.0.0.1')
			if 'ApiServerPort' in self.settings:
				md.set_global_parameter('ApiServerPort', self.settings['ApiServerPort'])
			else:
				md.set_global_parameter('ApiServerPort', 41451)
			# pipeline to open for console output

	def in_object(self, x, y, z):
		if self._rooftops is not None:
			return self._rooftops.in_object(x, y, z)
		return False

	# will get lowest z-point without being inside object at given x,y
	# this includes the floor (which will be lowest point found)
	def get_roof(self, x, y):
		if self._rooftops is not None:
			return self._rooftops.get_roof(x, y)
		return 0

	def make_voxels(self,
			  # ABSOLUTE path to write to, must be absolute
			  absolute_path:str,
			  # voxel params if make new voxels (else these are set from read)
			  # user be WARNED: use a cube centered around 0,0,0 because this sh** is wonky if not
			  center = [0,0,0], # in meters
			  resolution = 1, # in meters
			  x_length = 200, # total x-axis meters (split around center)
			  y_length = 200, # total y-axis  meters (split around center)
			  z_length = 200, # total z-axis  meters (split around center)
	):
		client = airsim.VehicleClient()
		center = airsim.Vector3r(center[0], center[1], center[2])
		# must create voxel using an absolute path
		client.simCreateVoxelGrid(center, 
									x_length, 
									y_length, 
									z_length, 
									resolution, 
									absolute_path,
									)

	# launch airsim map
	def connect(self, from_crash=False):
		if from_crash:
			md.speak('Recovering from AirSim crash...')
			self.disconnect()
		else:
			super().connect()
		
		# check OS to determine how to launch map
		OS = md.get_global_parameter('OS')
		# set flags
		flags = ''
		if self.console_flags is not None:
			flags = ' '.join(self.console_flags)
		# launch map from OS
		if OS == 'windows':
			_release_path = self.release_path
			# send command to terminal to launch the relase executable, if can
			if os.path.exists(_release_path):
				md.speak(f'Launching AirSim at {_release_path}')
				terminal_command = f'{_release_path} {flags} -settings=\"{self._settings_path}\"'
				md.speak(f'Issuing command to OS: {terminal_command}')
				process = subprocess.Popen(terminal_command, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
				self._pid = process.pid
			else:
				md.speak('AirSim release path DNE.')
		elif OS == 'linux':
			_release_path = self.release_path
			# send command to terminal to launch the relase executable, if can
			if os.path.exists(_release_path):
				md.speak(f'Launching AirSim at {_release_path}')
				terminal_command = f'sh {_release_path} {flags} -settings=\"{self._settings_path}\"'
				md.speak(f'Issuing command to OS: {terminal_command}')
				process = subprocess.Popen(terminal_command, shell=True, start_new_session=True)
				self._pid = process.pid
			else:
				md.speak('AirSim release path DNE.')
		else:
			md.speak('incompatible OS.')
				
		# wait for map to load
		time.sleep(60)

		# establish communication link with airsim client
		if self.vehicle in ['multirotor']:
			self._client = airsim.MultirotorClient(
				ip=md.get_global_parameter('LocalHostIp'),
				port=md.get_global_parameter('ApiServerPort'),
				timeout_value=10,
			)
		else:
			self._client = airsim.VehicleClient(
				timeout_value=10,
			)
		self._client.confirmConnection()
		if self.vehicle in ['multirotor', 'car']:
			self._client.enableApiControl(True)
			self._client.armDisarm(True)
			self._client.takeoffAsync().join()
		
		# remove animals - sad tear drop (there is a monument in the park to them, not jk)
		if self.remove_animals:
			self.remove_all_animals()
		
		# set class IDs for segmentation
		if self.segmentation:
			self.set_segmentation()
		
		# set weather
		if self.weather_type > -1:
			self.set_weather(self.weather_type, self.weather_degree)
			# add wet roads with rain
			if self.weather_type == 0:
				self.set_weather(1, self.weather_degree)
			# add snowy roads with snow
			if self.weather_type == 2:
				self.set_weather(3, self.weather_degree)
			# add leafty roads with leafs
			if self.weather_type == 4:
				self.set_weather(5, self.weather_degree)

		# wait to render
		time.sleep(10)

	def clear_weather(self):
		for i in range(8):
			self._client.simSetWeatherParameter(i, 0)

	def set_weather(self, weather_type, weather_degree):
		print('setting weather', weather_type, weather_degree)
		self._client.simEnableWeather(True)
		self._client.simSetWeatherParameter(weather_type, weather_degree)

	def remove_all_animals(self):
		objs = self._client.simListSceneObjects()
		animals = [name for name in objs if 'Deer' in name or 'Raccoon' in name or 'Animal' in name]
		_ = [self._client.simDestroyObject(name) for name in animals] # PETA has joined the chat
	
	# close airsim map
	def disconnect(self):
		# this should keep child in tact to kill same process created (can handle multi in parallel)
		if self._pid is not None:
			try:
				parent = psutil.Process(self._pid)
				for child in parent.children(recursive=True):
					child.kill()
				parent.kill()
			except:
				pass
	
	# read several json files with Airsim settings and merge
	def read_settings(self, settings_directory, setting_files):
		merged_settings = {}
		for setting_file in setting_files:
			setting_path = os.path.join(settings_directory, setting_file) + '.json'
			setting = md.read_json(setting_path)
			merged_settings.update(setting)
		return merged_settings

	# write a json settings dictionary to file
	def write_settings(self, merged_settings, settings_path):
		md.write_json(merged_settings, settings_path)

	def set_segmentation(self):
		# burn one
		request = airsim.ImageRequest('0', 5, False, False)
		self._client.simSetSegmentationObjectID(".*", 0, is_name_regex=True)
		burn = self._client.simGetImages([request])[0]
		
		# map objects to class
		# I did this by hand for AirSimNH please do not overwrite
		obj_dict = {
			# outside
			'Banister':'stairs',
			'Basketball_Hoop':'basketball_hoop',
			'Bench':'bench',
			'Birch':'tree',
			'Fir':'tree',
			'Oak':'tree',
			'Tree':'tree',
			'Car':'car',
			'Chimney':'house',
			'Flat_Roof':'house',
			'Cladding':'house',
			'Porch':'house',
			'Wall':'house',
			'House':'house',
			'Roof':'house',
			'Support':'house',
			'Veranda':'house',
			'Driveway':'driveway',
			'driveway':'driveway',
			'Road':'road',
			'Fence':'fence',
			'Floor':'floor',
			'Rock':'rock',
			'Garden_Tressel':'rack',
			'Hedge':'bush',
			'Leaves':'leaves',
			'Monument':'statue',
			'Steps':'stairs',
			'Stair':'stairs',
			'Power_Line':'cable',
			'Shutters':'window',
			'Sign':'sign',
			'Pool':'water',
			'bin':'bin',
			'Door':'door',
			'Fog':'fog',
			'Raccoon':'raccoon',
			'Deer':'deer',

			# inside
			'Tile':'tile',
			'Sink':'sink',
			'Lamp':'lamp',
			'Cupboard':'cupbard',
			'Shelf':'cupbard',
			'Headphones':'headphones',
			'Mouse':'mouse',
			'Keyboard':'keyboard',
			'Gamepad':'gamepad',
			'Fridge':'fridge',
			'fan':'fan',
			'Fan':'fan',
			'desktop':'computer',
			'Laptop':'computer',
			'Dufflebag':'bag',
			'Can':'can',
			'Drain_Pipe':'pipe',
			'Curtain':'curtain',
			'table':'table',
			'Table':'table',
			'Chair':'chair',
			'Bed':'bed',
			'Book':'book',
			'Mug':'cup',
			'Desk':'desk',
			'Oven':'oven',
			'Picture':'picture',
			'Light_Switch':'furnishing',
			'Plug_Socket':'furnishing',
			'Skateboard':'skateboard',
			'Sofa':'sofa',
			'sofa':'sofa',
			'Toilet':'toilet',
			
			# other
			'Landscape':'unknown',
			'LightSource':'unknown',
			'PointLight':'unknown',
			'landscape':'unknown',
			'SkyLight':'unknown',
			'Kitchen_Unit':'unknown',
			'lcd':'unknown',
			'Lcd':'unknown',
			'Tft':'unknown',
			'extractor':'unknown',
			'Flor':'unknown',
			'Player':'immaterial',
			'Actor':'immaterial',
			'Cube':'immaterial',
			'Cue':'immaterial',
			'Event':'immaterial',
			'Debugger':'immaterial',
			'State':'immaterial',
			'Session':'immaterial',
			'Network':'immaterial',
			'Manager':'immaterial',
			'Director':'immaterial',
			'Settings':'immaterial',
			'Camera':'immaterial',
			'HUD':'immaterial',
			'Capture':'immaterial',
			'Controller':'immaterial',
			'SimMode':'immaterial',
			'Demo':'immaterial',
			'Physics':'immaterial',
			'Volume':'immaterial',
			'SimpleFlight':'immaterial',
			'AirSimGameMode':'immaterial',
			'RecastNavMesh':'immaterial',
			'AbstractNavData':'immaterial',
		}

		# get unique ID numbers for object TYPES
		# id_dict = {}
		# id = 0
		# for key in obj_dict:
		# 	value = obj_dict[key]
		# 	if value not in id_dict:
		# 		if id > 255:
		# 			print('too many classes in airsimmap.set_segmentation()')
		# 		id_dict[value] = id
		# 		id += 1

		# get color mapping https://github.com/microsoft/AirSim/issues/3423
		colors = {0:[0,0,0]} # background (careful this can sometimes be wonky black or a brownish-green-grey color)
		for cls_id in range(1, 256):
			# map every asset to cls_id and extract the single RGB value produced
			self._client.simSetSegmentationObjectID(".*", cls_id, is_name_regex=True)
			img_array = []
			while len(img_array) <= 0: # loop for dead images (happens some times)
				response = self._client.simGetImages([request])[0]
				np_flat = np.fromstring(response.image_data_uint8, dtype=np.uint8)
				img_array = np.reshape(np_flat, (response.height, response.width, 3))
			#plt.imshow(img_array)
			#plt.show()
			color = list(np.unique(img_array.reshape([-1, 3]), axis=0)[-1]) # get last index to ignore background at first index
			colors[cls_id] = [int(band) for band in color]

		# map each indiviual object (default example is car) to segmentation id
		n_objs = 0
		objs = self._client.simListSceneObjects()
		obj_ids = {}
		for obj in objs:
			if 'Car' in obj:
				n_objs += 1
				self._client.simSetSegmentationObjectID(obj, n_objs, True)
				obj_ids[obj] = n_objs
				#print('car object name', obj, 'assigned', n_objs)
			else:
				self._client.simSetSegmentationObjectID(obj, 0, True)
				#print('other object name', obj, 'assigned', 0)

		working_directory = md.get_global_parameter('working_directory')
		part_name = md.get_global_parameter('part_name')
		md.write_json(colors, f'{working_directory}segmentation_colors__{part_name}.json')
		md.write_json(obj_ids, f'{working_directory}obj_ids__{part_name}.json')