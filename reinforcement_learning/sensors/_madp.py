from sensors.sensor import Sensor
from observations.vector import Vector
from observations.image import Image
from observations.array import Array
from component import _init_wrapper
import numpy as np
import rl_utils as _utils
import map_data.map_methods as mm
import os

import time

# gets position from a miscellaneous component
# can send in a second component to get positoin and distance between two
class _Madp(Sensor):
	
	@_init_wrapper
	def __init__(self,
				datadictDepth_component,
				datadictMADP_component,
				drone_component,
				sensor_name,
				map_name,
				observations_path,
				queries_path,
				max_x, # when to refresh with new Depth frame
				use_future = False, # turning this on will add future predictions to obs - see below 2 params
				max_x2 = 0, # how far in future to forecase
				step_size = 2, # how large of steps into future (number of max future steps = max_x2/step_size)
				offline=False,
				transformers_components=None,
			  	memory='all', # none all part
				discretize=True,
			  ):
		super().__init__(offline, memory)
		self._sensor_path = observations_path + sensor_name + '/'
		if not os.path.exists(self._sensor_path):
			_utils.speak(f'error: sensor named {self.sensor_name} DNE at observations path {self.observations_path}')
		info_path = self._sensor_path + 'info.json'
		info = _utils.read_json(info_path)
		self._obs_type = info['obs_type']
		self._state_type = info['state_type']
		self._vector_length = info['vector_length'] if 'vector_length' in info else 0
		self._names = info['names'] if 'names' in info else None
		self._image_bands = info['image_bands'] if 'image_bands' in info else 0
		self._image_height = info['image_height'] if 'image_height' in info else 0
		self._image_width = info['image_width'] if 'image_width' in info else 0
		self._is_gray = info['is_gray'] if 'is_gray' in info else False
		self._array_size = info['array_size'] if 'array_size' in info else 0
		# update with future info
		if self.use_future:
			self.memory = 'part'
			self._future_bands = int(max_x2 / step_size)
			self._past_bands = self._array_size[0]
			self._remember_bands = self._past_bands # this is used by the Observer object when making obs to add to buffer
			self._array_size = [self._past_bands+self._future_bands, self._array_size[1], self._array_size[2]]

	def connect(self, state=None):
		super().connect(state)
		self.set_null()
		self.set_first_frame()

	def get_vector_length(self):
		return self._vector_length

	# returns null data as transformed observation object
	def get_null(self):
		return self._null

	# returns null data as numpy array before transformations
	def get_null_raw(self):
		return self._null_raw

	# returns a single band of null data as numpy array before transformations
	def get_single_null_raw(self):
		return self._null_raw[:1]
	
	def set_null(self):
		if self._obs_type in ['vector']:
			null_data = np.zeros(self._vector_length).astype(float)
		elif self._obs_type in ['image']:
			null_data = np.zeros(self.get_image_shape()).astype(np.uint8)
		elif self._obs_type in ['array']:
			null_data = np.zeros(self._array_size).astype(float)
		self._null_raw = null_data.copy()
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
		x, y, z = state['drone_position']
		yaw = state['yaw']
		# discretize
		if self.discretize:
			x, y, z = round(x), round(y), round(z)
			yaw = mm.yaw_to_direction(yaw)
		if np.isnan(self._x):
			use_madp = False
		else:
			if yaw == self._yaw:
				if yaw in [0, 2]:
					delta_x = abs(self._x - x)
				else:
					delta_x = abs(self._y - y)
				use_madp = delta_x <= self.max_x
			else:
				use_madp = False
		state['use_madp'] = use_madp
		# numpy array of shape [1, height, width]
		if use_madp:
			data = self._datadictMADP.get_data([self._x, self._y, self._z, self._yaw, delta_x])
		else:
			data = self._datadictDepth.get_data([x, y, z, yaw])
			self._x, self._y, self._z, self._yaw = x, y, z, yaw
		# if data is missing
		if data is None:
			#data = self.get_null_raw()
			job_name = _utils.get_local_parameter('job_name')
			_utils.progress(job_name, f'MISSING DATA ERROR')
			print('no data at', x, y, z, yaw, use_madp)
			if use_madp:
				print('MADP', self._x, self._y, self._z, self._yaw, delta_x)
			ini = input()
		# future predictions
		if self.use_future:
			# each future frame is a numpy array of shape [1, height, width]
			data2 = [None] * self._future_bands
			band_idx = 0
			x2, y2 = x, y
			# fetch full frames not exceeding delta_x2
			print('MADP step')
			while(True):
				if yaw in [0, 2]: # x-axis movement
					x2 += self.step_size*(1 if yaw==0 else -1) # account for direction
					delta_x2 = abs(self._x - x2)
				else: # yaw in [1, 3]: # y-axis movement
					y2 += self.step_size*(1 if yaw==1 else -1) # account for direction
					delta_x2 = abs(self._y - y2)
				if delta_x2 > self.max_x2:
					break
				print(self.step_size, delta_x2, self.max_x2, band_idx, self._future_bands)
				data2[band_idx] = self._datadictMADP.get_data([self._x, self._y, self._z, self._yaw, delta_x2])
				band_idx += 1
			# fill remaining frames with null
			for i in range(band_idx, self._future_bands):
				data2[i] = self.get_single_null_raw()
			data = np.vstack([data, np.vstack(data2)])
		print('MADP data shape', data.shape)
		print()
		observation = self.create_obj(data)
		transformed = self.transform(observation)
		return transformed

	def set_first_frame(self):
		self._x, self._y, self._z, self._yaw = np.nan, np.nan, np.nan, np.nan

	def reset(self, state=None):
		super().reset(state)
		self.set_first_frame()