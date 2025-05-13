from sensors.sensor import Sensor
from observations.vector import Vector
from observations.image import Image
from observations.array import Array
from component import _init_wrapper
import numpy as np
import map_data.map_methods as mm
import utils.global_methods as gm
import os

import time

# 
class DataMap(Sensor):
	
	@_init_wrapper
	def __init__(self,
				drone_component,
				sensor_name,
				sensor_dir,
				offline=False,
				transformers_components=None,
				out_size=(0),
				#future=0,
				#action_component=None,
				memory='all', # none all part
			  ):
		super().__init__(offline, memory)
		if not os.path.exists(self.sensor_dir):
			gm.speak(f'error: sensor named {self.sensor_name} DNE at observations path {self.observations_path}')
		info_path = self.sensor_dir + 'info.json'
		info = gm.read_json(info_path)
		self._obs_type = info['obs_type']
		self._state_type = info['state_type']
		self._vector_length = info['vector_length'] if 'vector_length' in info else 0
		self._names = info['names'] if 'names' in info else None
		self._image_bands = info['image_bands'] if 'image_bands' in info else 0
		self._image_height = info['image_height'] if 'image_height' in info else 0
		self._image_width = info['image_width'] if 'image_width' in info else 0
		self._is_gray = info['is_gray'] if 'is_gray' in info else False
		self._array_size = info['array_size'] if 'array_size' in info else 0
		self._datamap = gm.get_global('datamap')
		self.out_size = self._array_size

	def connect(self, state=None):
		super().connect(state)
		self.set_null()

	def get_vector_length(self):
		return self._vector_length

	def get_null(self):
		return self._null
	
	def set_null(self):
		if self._obs_type in ['vector']:
			null_data = np.zeros(self._vector_length).astype(float)
		elif self._obs_type in ['image']:
			null_data = np.zeros(self.get_image_shape()).astype(np.uint8)
		elif self._obs_type in ['array']:
			null_data = np.zeros(self._array_size).astype(float)
		observation = self.create_obj(null_data)
		transformed = self.transform(observation)
		self._null = transformed
	
	def get_image_shape(self):
		return [self._image_bands, self._image_height, self._image_width]

	def create_obj(self, data):
		if self._obs_type in ['vector']:
			observation = Vector(
				_data = data,
				names = self._names,
			)
		elif self._obs_type in ['image']:
			observation = Image(
				_data = data,
				is_gray = self._is_gray,
			)
		elif self._obs_type in ['array']:
			observation = Array(
				_data = data,
			)
		return observation

	def step(self, state=None):
		data = None
		x, y, z = state['drone_position']
		direction = state['direction']
		# if self._state_type in ['4vec']:
		# 	if self.future > 0:
		# 		stateb4 = [x, y, z, yaw]
		# 		collision_b4 = collision = self._drone.check_collision()
		# 		oob_b4 = oob = self._drone.out_of_bounds()
		# 		if not collision_b4 and not oob_b4:
		# 			transcribed_action = self._action.step(state, execute=False)
		# 			if 'x' in transcribed_action:
		# 				x += transcribed_action['x']
		# 			if 'y' in transcribed_action:
		# 				y += transcribed_action['y']
		# 			if 'z' in transcribed_action:
		# 				z += transcribed_action['z']
		# 			if 'yaw' in transcribed_action:
		# 				yaw += transcribed_action['yaw']
		# 			self._drone.teleport(x, y, z, yaw, ignore_collision=False, out_of_bounds=True)
		# 			collision = self._drone.check_collision()
		# 			oob = self._drone.out_of_bounds()
		# 		self._drone.teleport(*stateb4, ignore_collision=True, out_of_bounds=False)
		# 		self._drone._collision = collision_b4
		# 		self._drone._oob = oob_b4
		# 		if collision or oob:
		# 			data = self.get_null().to_numpy()
		# discretize
		data = self._datamap.get_data_point(x, y, z, direction, self.sensor_name)
		if data is None:
			print('no data at', x, y, z, direction)
			ini = input()
		observation = self.create_obj(data)
		transformed = self.transform(observation)
		return transformed