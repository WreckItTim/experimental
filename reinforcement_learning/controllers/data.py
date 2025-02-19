from controllers.controller import Controller
from component import _init_wrapper
import numpy as np
import math
import rl_utils as _utils

import os
import pickle
import time

# collects data by collecting observations
	# specify points to move to on map, along with which sensors to capture at each point
class Data(Controller):


	# constructor
	@_init_wrapper
	def __init__(self,
				 drone_component, # agent to move around map
				 sensor_component, # which sensor to capture at each point
				 map_component, # map to handle crash with
				 points, # list of points to capture data from
				 part_name='', # str value used purely for read/write purposes for unique file names
				 job_name='data', # unique-enough name for logging progress
				 states=[], # list of states at each point to pass into sensor.step()
				 crash_handler=True,
				 discretize=True,
				 checkpoint_frequency=100,
				 save_as='dict', # 'dict' or 'list'
				 sleep_time = 1, # seconds to wait between moving to next data point and collecting observation
				 ):
		super().__init__()
		self._added_points = 0
		if crash_handler:
			import msgpackrpc
			self._exception = msgpackrpc.error.TimeoutError
			self.run = self.run_with_crash
		else:
			self.run = self.run_without_crash

	def start(self):
		_utils.progress(self.job_name, f'collected 0%')
		working_directory = _utils.get_local_parameter('working_directory')
		self._last_obs = None

		self._log_path = f'{working_directory}log__{self.part_name}.json'
		self.read_log()

		self._point_list_path = f'{working_directory}point_list__{self.part_name}.p'
		self.read_point_list()

		if self.save_as in ['dict']:
			# load part data dictionary (data to be collected now)
			self._data_dict_path = f'{working_directory}data_dict__{self.part_name}.p'
			if not os.path.exists(self._data_dict_path):
				self._log = {}
				self._point_list = []
			self.read_data_dict()

		if self.save_as in ['list']:
			self._data_list_path = f'{working_directory}data_list__{self.part_name}.p'
			self.read_data_list()

	def finish(self):
		self.write_data()

	def get_data_point(self, point, p_idx, state):
		# get next point
		x, y, z, yaw, yaw_idx = self.next_point(point)

		# data point exists yet?
		if self.data_exists(x, y, z, yaw_idx, p_idx):
			return

		# make sensor observation at desired location
		data, x_drone, y_drone, z_drone, yaw_drone = self.observe(x, y, z, yaw, p_idx, state)

		# add new data point to list or dict
		self.add_data_point(x, y, z, yaw, yaw_idx, x_drone, y_drone, z_drone, yaw_drone, p_idx, data)

		# checkpoint
		if p_idx > 0 and p_idx % self.checkpoint_frequency == 0:
			self.checkpoint(p_idx)


	#### helper functions

	def check_tolerance(self,
		x, y, z, yaw, drone_x, drone_y, drone_z, drone_yaw, p_idx,
		tolerance_x=0.1, tolerance_y=0.1, tolerance_z=0.1, tolerance_yaw=0.1,
	):
		yaw_diff = abs(yaw - drone_yaw)
		yaw_diff = min(2*math.pi - yaw_diff, yaw_diff)
		failed = abs(drone_x-x) > tolerance_x or abs(drone_y-y) > tolerance_y or abs(drone_z-z) > tolerance_z or yaw_diff > tolerance_yaw
		if failed:
			self.add_log({
				'entry':'pos',
				'x':x,
				'y':y,
				'z':z,
				'yaw':yaw,
				'drone_x':drone_x,
				'drone_y':drone_y,
				'drone_z':drone_z,
				'drone_yaw':drone_yaw,
				'p_idx':p_idx,
			})
		return failed

	def random_tolerance(self, drone_x, drone_y, drone_z, drone_yaw):
		fup = np.random.randint(0, 100)
		if fup == 0:
			fup_dim = np.random.randint(0, 4)
			if fup_dim == 0:
				drone_x += 1
			if fup_dim == 1:
				drone_y += 1
			if fup_dim == 2:
				drone_z += 1
			if fup_dim == 3:
				drone_yaw += 0.4
		return drone_x, drone_y, drone_z, drone_yaw

	def random_crash(self):
		crash = np.random.randint(0, 100)
		if crash == 1:
			self._map.disconnect()

	def write_data(self):
		if self._added_points > 0:
			self.add_log({
					'entry':'write',
					'n_points':len(self._point_list),
			})
			if self.save_as in ['dict']:
				_utils.pk_write(self._data_dict, self._data_dict_path)
			if self.save_as in ['list']:
				_utils.pk_write(self._data_list, self._data_list_path)
			_utils.pk_write(self._point_list, self._point_list_path)
			_utils.write_json(self._log, self._log_path)
		self._added_points = 0

	def data_exists(self, x, y, z, yaw_idx, p_idx):
		if self.save_as in ['dict']:
			try:
				_ = self._data_dict[x][y][z][yaw_idx]
				return True
			except:
				return False
		if self.save_as in ['list']:
			return p_idx < len(self._data_list)

	def add_data_point(self, x, y, z, yaw, yaw_idx, x_drone, y_drone, z_drone, yaw_drone, p_idx, data):
		if self.save_as in ['dict']:
			if x not in self._data_dict:
				self._data_dict[x] = {}
			if y not in self._data_dict[x]:
				self._data_dict[x][y] = {}
			if z not in self._data_dict[x][y]:
				self._data_dict[x][y][z] = {}
			self._data_dict[x][y][z][yaw_idx] = data
		if self.save_as in ['list']:
			self._data_list.append(data)
		self._point_list.append({
			'x':x, 
			'y':y, 
			'z':z, 
			'yaw':yaw,
			'x_drone':x_drone, 
			'y_drone':y_drone, 
			'z_drone':z_drone, 
			'yaw_drone':yaw_drone,
		})
		self._added_points += 1

	def add_log(self, entry_dict):
		timestamp = _utils.get_timestamp()
		#entry_dict.update({'timestamp':timestamp})
		self._log[timestamp] = entry_dict

	def next_point(self, point):
		x, y, z, yaw = point
		if self.discretize:
			x, y, z = round(x), round(y), round(z)
			yaw_idx = _utils.yaw_to_idx(yaw)
		else:
			yaw_idx = yaw
		return x, y, z, yaw, yaw_idx

	def teleport(self, x, y, z, yaw):
		# call drone to teleport
		self._drone.teleport(x, y, z, yaw, ignore_collision=True)

	def check_clone(self, p_idx, n_checks, data):
		if data.dtype == np.uint8:
			data = data.astype(np.int16)
		if self._last_obs is not None:
			diff = np.abs(data-self._last_obs)
			if np.mean(diff) <= 2:
				self.add_log({
					'entry':'clone',
					'p_idx':p_idx,
					'n_checks':n_checks,
				})
				return True
		self._last_obs = data.copy()
		return False

	def observe(self, x, y, z, yaw, p_idx, state, n_checks=1):
		# teleport to position
		self.teleport(x, y, z, yaw)

		# time to sleep to allow drone to reach next point and render next frame
		time.sleep(self.sleep_time)

		# fetch observed position
		drone_pos, drone_yaw = self._drone.get_position(), self._drone.get_yaw()
		drone_x, drone_y, drone_z = drone_pos

		# check if drone's position is within tolerance of desired position
		failed = self.check_tolerance(x, y, z, yaw, drone_x, drone_y, drone_z, drone_yaw, p_idx)
		if failed and n_checks > 0:
			# make sensor observation
			n_checks -= 1
			data, drone_x, drone_y, drone_z, drone_yaw = self.observe(x, y, z, yaw, p_idx, state, n_checks)
		
		# make sensor observation
		data = self._sensor.step(state).to_numpy()

		# check if this is clone (this is a bug in AirSim that will return the previous observation)
		is_clone = self.check_clone(p_idx, n_checks, data)
		if is_clone and n_checks > 0:
			# make sensor observation
			n_checks -= 1
			data, drone_x, drone_y, drone_z, drone_yaw = self.observe(x, y, z, yaw, p_idx, state, n_checks)
		
		return data, drone_x, drone_y, drone_z, drone_yaw
		
	def checkpoint(self, p_idx):
		self.write_data()
		percent_complete = int(100 * p_idx / len(self.points))
		_utils.progress(self.job_name, f'collected {percent_complete}%')
		_utils.speak(f'{percent_complete}% data collected! On point index {p_idx}')

	def read_data_dict(self):
		try:
			self._data_dict = _utils.pk_read(self._data_dict_path)
		except:
			self._data_dict = {}

	def read_data_list(self):	
		try:
			self._data_list = _utils.pk_read(self._data_list_path)
		except:	
			self._data_list = []

	def read_point_list(self):	
		try:
			self._point_list = _utils.pk_read(self._point_list_path)
		except:	
			self._point_list = []

	def read_log(self):	
		try:
			self._log = _utils.read_json(self._log_path)
		except:	
			self._log = {}

	def run_without_crash(self):
		self.start()
		for p_idx, point in enumerate(self.points):
			if p_idx < len(self.states):
				state = self.states[p_idx]
			else:
				state = {}
			self.get_data_point(point, p_idx, state)
		self.finish()

	# runs control on components
	def run_with_crash(self):
		self.start()
		for p_idx, point in enumerate(self.points):
			if p_idx < len(self.states):
				state = self.states[p_idx]
			else:
				state = {}
			try_again = True # crash handler
			while try_again: # crash handler
				try: # crash handler
					self.get_data_point(point, p_idx, state)
					try_again = False # crash handler
				except self._exception as e: # crash handler
					_utils.speak(str(e) + ' crash caught at point index # ' + str(p_idx)) # crash handler
					self._map.connect(from_crash=True) # crash handler
					_utils.speak(str(e) + ' recovered from crash') # crash handler
					self.add_log({
						'entry':'crash',
						'point':point,
						'p_idx':p_idx,
						'state':state,
					})
		self.finish()
