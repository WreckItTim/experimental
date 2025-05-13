from sensors.sensor import Sensor
from observations.vector import Vector
from component import _init_wrapper
import numpy as np

# keeps track of previous action
# self normalizes to [0,1]
class Actions(Sensor):
	
	@_init_wrapper
	def __init__(self,
			actor_component,
			prefix = '',
			transformers_components = None,
			offline = False,
			memory='all', # none all part
			out_size=(1),
		):
		super().__init__(offline, memory)

	def connect(self, state=None):
		super().connect(state)
		if self._actor._type == 'discrete':
			self.out_size = (1)
		else:
			self.out_size = (len(self._actor._actions))

	def create_obj(self, data):
		observation = Vector(
			_data = data,
		)
		return observation
	
	def get_null(self):
		if self._actor._type == 'discrete':
			data = [0]
		else:
			data = [0] * len(self._actor._actions)
		return self.create_obj(data)
		
	def step(self, state):
		nSteps = state['nSteps']
		if nSteps == 0:
			if self._actor._type == 'discrete':
				data = [1 / (len(self._actor._actions)+2)]
			else:
				data = [1 / (action[i].max_space - action.min_space + 2) for action in self._actor._actions]
		else:	
			rl_output = state['rl_output']
			if self._actor._type == 'discrete':
				# add 2 to length so that 0 is reserved for no-data and 1 is reserved for start of episode
				data = [(rl_output+2) / (len(self._actor._actions)+2)]
			else:
				# add 2 so that 0 is reserved for no-data and 1 is reserved for start of episode
				data = [0] * len(rl_output)
				for i in range(len(rl_output)):
					data[i] = (rl_output[i] - self._actor._actions[i].min_space + 2) / (self._actor._actions[i].max_space - self._actor._actions[i].min_space + 2)

		observation = self.create_obj(data)
		transformed = self.transform(observation)
		return transformed