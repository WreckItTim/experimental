from controllers.controller import Controller
from component import _init_wrapper
import numpy as np
import math
import rl_utils as _utils

import os
import pickle
import numpy as np
from PIL import Image

def save_png(img, path):
	# rgb to bgr
	img = img.copy()
	temp = img[0, :, :].copy()
	img[0, :, :] = img[2, :, :].copy()
	img[2, :, :] = temp
	# move channel first to channel last
	img = np.moveaxis(img, 0, 2)
	img = Image.fromarray(img)
	img.save(path)

# moves/teleports to desired points and does a tolerance check to see if successfully moved
class Tolerance(Controller):
	# constructor
	@_init_wrapper
	def __init__(self,
				 drone_component, # agent to move around map
				 map_component, # map to handle crash with
				 points, # list of points to capture data from
				 tolerance_x=0.1, # must get to the desired x-point within this absolute tolerance
				 tolerance_y=0.1, # must get to the desired y-point within this absolute tolerance
				 tolerance_z=0.1, # must get to the desired z-point within this absolute tolerance
				 tolerance_yaw=0.1, # must get to the desired yaw-point within this absolute tolerance
				 sensors_components=[], # optionally which sensors to capture at each point
				 job_name='tolerance', # unique name for logging progress (not needed)
				 crash_handler=True,
				 discretize=True,
				 ):
		super().__init__()
		if crash_handler:
			import msgpackrpc
			self._exception = msgpackrpc.error.TimeoutError
			self.run = self.run_with_crash
		else:
			self.run = self.run_without_crash
			
	def check_tolerance(self,
		x, y, z, yaw, 
		drone_x, drone_y, drone_z, drone_yaw,
	):
		if abs(drone_x-x) > self.tolerance_x:
			_utils.speak(f'not within x tolerance! {x} {y} {z} {yaw} {drone_x} {drone_y} {drone_z} {drone_yaw}')
		if abs(drone_y-y) > self.tolerance_y:
			_utils.speak(f'not within y tolerance! {x} {y} {z} {yaw} {drone_x} {drone_y} {drone_z} {drone_yaw}')
		if abs(drone_z-z) > self.tolerance_z:
			_utils.speak(f'not within z tolerance! {x} {y} {z} {yaw} {drone_x} {drone_y} {drone_z} {drone_yaw}')
		yaw_diff = abs(yaw - drone_yaw)
		yaw_diff = min(2*math.pi - yaw_diff, yaw_diff)
		if yaw_diff > self.tolerance_yaw:
			_utils.speak(f'not within yaw tolerance! {x} {y} {z} {yaw} {drone_x} {drone_y} {drone_z} {drone_yaw}')

	def goto_data_point(self, point, p_idx):
		x, y, z, yaw = point
		if self.discretize:
			x, y, z = round(x), round(y), round(z)
		self._drone.teleport(x, y, z, yaw, ignore_collision=True)
		for sensor in self._sensors:
			observation = sensor.step(state)
		# check tolerance
		drone_pos, drone_yaw = self._drone.get_position(), self._drone.get_yaw()
		drone_x, drone_y, drone_z = drone_pos
		self.check_tolerance(
			x, y, z, yaw,
			drone_x, drone_y, drone_z, drone_yaw,
		)
		if p_idx%100 == 0:
			_utils.progress(self.job_name, f'collected {int(100*p_idx/len(self.points))}%')

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
					_utils.speak(str(e) + ' caught at point index # ' + str(p_idx)) # crash handler
					self._map.connect(from_crash=True) # crash handler
					_utils.speak(str(e) + ' recovered') # crash handler
