from rewards.reward import Reward
from component import _init_wrapper
import numpy as np
import math
import utils.global_methods as gm

# calculates distance between drone and point relative to starting position/orientation
class Goal(Reward):
	# constructor, set the relative point and min-max distances to normalize by
	@_init_wrapper
	def __init__(self,
				drone_component, 
				goal_component, 
				include_z = True, # include z in distance calculations
				tolerance = 4, # min distance from goal for success 
				# if to_start=True will calculate rewards relative to start position
				# if to_start=False will calculate rewards relative to last position
				terminate = True, # =True will terminate episodes when Goal
		):
		super().__init__()
		#self.init_normalization()

	def get_distance(self):
		_drone_position = np.array(self._drone.get_position(), dtype=float)
		_goal_position = np.array(self._goal.get_position(), dtype=float)
		if not self.include_z:
			_drone_position = np.array([_drone_position[0], _drone_position[1]], dtype=float)
			_goal_position = np.array([_goal_position[0], _goal_position[1]], dtype=float)
		distance_vector = _goal_position - _drone_position
		distance = np.linalg.norm(distance_vector)
		return distance
	
	# get reward based on distance to point 
	def step(self, state):
		distance = self.get_distance()

		done = False
		value = 0
		if 'termination_reason' in state and state['termination_reason'] == 'action':
			value = -1
		if distance <= self.tolerance:
			value = 1
			done = True
			state['reached_goal'] = True
			if self.terminate and 'termination_reason' not in state:
				state['termination_reason'] = 'goal'

		return value, done and self.terminate