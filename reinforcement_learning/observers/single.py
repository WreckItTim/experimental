# abstract class used to handle observations to input into rl algo
from observers.observer import Observer
from component import _init_wrapper
from gymnasium import spaces
import numpy as np
import global_methods as md
import copy

class Single(Observer):
	
	# and observer with same observation types (can be multiple sensors)
	# observation type is either vector or image
	# image shapes are channel first, rows, cols
	@_init_wrapper
	def __init__(
		self, 
		sensors_components,
		map_component='Map',
		vector_length_forget = None,
		vector_length_remember = None,
		is_image = False,
		image_height = None, 
		image_width = None,
		image_bands_forget = None,
		image_bands_remember = None,
		nPast = 1,
		null_if_in_obj = False, # inside an object
		null_if_oob = False, # out of bounds
	):
		super().__init__(
		)
		if is_image:
			self._output_shape = (image_bands_forget + image_bands_remember*nPast, image_height, image_width)
			self._history_shape = (image_bands_remember*nPast, image_height, image_width)
			self._history = np.full(self._history_shape, 0, dtype=np.uint8)
		else:
			self._output_shape = (vector_length_forget + vector_length_remember*nPast,)
			self._history_shape = (vector_length_remember * nPast,)
			self._history = np.full(self._history_shape, 0, dtype=float)
		self._old_names = []
		#print('self._output_shape', self._output_shape)

	def null_data(self):
		if self.is_image:
			np.zeros(self._output_shape, dtype=np.uint8)
		else:
			np.zeros(self._output_shape, dtype=float)

	# gets observations
	def step(self, state=None):
		#print('STEP')
		# temporarily toggle sensors offline if we are inside of an object
		offline_all = False
		x, y, z = state['drone_position']
		#print('xyz', x, y, z)
		if self.null_if_in_obj:
			offline_all = self._map.in_object(x, y, z)
			#print('null_if_in_obj', offline_all)
		if self.null_if_oob and not offline_all:
			offline_all = self._map.check_bounds(x, y, z)
			#print('null_if_oob', offline_all)
		if offline_all:
			for sensor in self._sensors:
				sensor._from_observer = sensor.offline
				sensor.offline = True
		# make observations and stack into local image/vector
		next_array_forget = []
		next_array_remember = []
		new_names_forget = []
		new_names_remember = []
		for sensor in self._sensors:
			# get obeservation
			if sensor.offline:
				if self.is_image:
					empty_observation = sensor.get_null()
					empty_name = 'I0'
				else:
					empty_observation = sensor.get_null()
					empty_name = 'V0'
				this_array = empty_observation.to_numpy()
				this_name = empty_name
			else:
				observation = sensor.step(state)
				this_array = observation.to_numpy()
				this_name = observation._name
			if sensor.memory == 'part' and sensor._remember_bands < this_array.shape[0]:
				print('partial memory', sensor._remember_bands, this_array.shape)
				# TODO -- change remember_bands to be a full shape idx array instead
				next_array_remember.append(this_array[:sensor._remember_bands])
				new_names_remember.append(this_name)
				next_array_forget.append(this_array[sensor._remember_bands:])
				new_names_forget.append(this_name)
			elif sensor.memory == 'none':
				next_array_forget.append(this_array)
				new_names_forget.append(this_name)
			else: # sensor.memory == 'all':
				next_array_remember.append(this_array)
				new_names_remember.append(this_name)
			#print('this_array.shape', this_array.shape)
		# toggle sensors offline status back to before checking if inside object
		if offline_all:
			for sensor in self._sensors:
				sensor.offline = sensor._from_observer
		# concatenate observations
		if self.nPast == 1:
			array = np.concatenate(next_array_forget+next_array_remember, 0)
			name = '_'.join(new_names_forget+new_names_remember)
			return array.copy(), name
		# rotate saved timesteps in history
		if self.is_image:
			for i in range(self.nPast-1, 0, -1):
				start_i = i * self.image_bands_remember
				save_to = slice(start_i, start_i + self.image_bands_remember)
				start_i = (i-1) * self.image_bands_remember
				load_from = slice(start_i, start_i + self.image_bands_remember)
				self._history[save_to,:,:] = self._history[load_from,:,:]
			save_to = slice(0, self.image_bands_remember)
			self._history[save_to,:,:] = np.concatenate(next_array_remember, 0)
		else:
			for i in range(self.nPast-1, 0, -1):
				start_i = i * self.vector_length_remember
				save_to = slice(start_i, start_i + self.vector_length_remember)
				start_i = (i-1) * self.vector_length_remember
				load_from = slice(start_i, start_i + self.vector_length_remember)
				self._history[save_to] = self._history[load_from]
			save_to = slice(0, self.vector_length_remember)
			self._history[save_to] = np.concatenate(next_array_remember, 0)
		self._old_names = [new_names_remember] + self._old_names
		if len(self._old_names) > self.nPast:
			self._old_names.pop(-1)
		these_names = copy.deepcopy(self._old_names)
		these_names[0] = new_names_forget + new_names_remember
		name = '_'.join(sum(these_names, []))
		if len(next_array_forget) > 0:
			these_observations = np.concatenate([np.concatenate(next_array_forget, 0), self._history.copy()], 0)
		else:
			these_observations = self._history.copy()
		return these_observations, name

	def reset(self, state=None):
		#print('RESET')
		for sensor in self._sensors:
			sensor.reset(state)
		if self.is_image:
			self._history = np.full(self._history_shape, 0, dtype=np.uint8)
		else:
			self._history = np.full(self._history_shape, 0, dtype=float)
		self._old_names = []

	# returns box space with proper dimensions
	def get_space(self):
		if self.is_image:
			return spaces.Box(low=0, high=255, shape=self._output_shape, dtype=np.uint8)
		return spaces.Box(0, 1, shape=self._output_shape, dtype=float)
