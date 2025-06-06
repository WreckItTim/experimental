from actors.actor import Actor
from component import _init_wrapper
from gymnasium import spaces
import numpy as np
import utils.global_methods as gm
import random
import math

# uses continuous actions to teleport (set yaw and position)
# this is much more quicker, precise, and stable than real-time movements
# suggested to add noise to simulate more realistic movements
class TeleporterContinuous(Actor):
	# constructor
	@_init_wrapper
	def __init__(self,
	drone_component,
	actions_components=[],
		discretize=False,
	):
		super().__init__()
		self._type = 'continuous'


	# interpret action from RL
	def step(self, state):
		if 'transcribed_action' not in state:
			state['transcribed_action'] = {}
		current_position = self._drone.get_position() # meters
		current_yaw = self._drone.get_yaw() # radians
		target = {
			'x':current_position[0],
			'y':current_position[1],
			'z':current_position[2],
			'yaw':current_yaw,
			}
		for idx, action in enumerate(self._actions):
			action._idx = idx # tell action which index it is
			this = action.step(state, execute=False) # transcribe action but do not take
			state['transcribed_action'].update(this)
			for key in target:
				if key in this:
					target[key] += this[key]
		# teleport drone
		state['transcribed_action'].update(target)
		if 'end' in state['transcribed_action']:
			if state['transcribed_action']['end']:
				state['termination_reason'] = state['transcribed_action']['end_reason']
				return True
		if self.discretize:
			target['x'] = round(target['x'])
			target['y'] = round(target['y'])
			target['z'] = round(target['z'])
			target['yaw'] = round(target['yaw']/(math.pi/2))*(math.pi/2)
		self._drone.teleport(target['x'], target['y'], target['z'], target['yaw'], ignore_collision=False)
		return False
		
	# randomly sample RL output from action space unless specified
	def debug(self, state=None):
		if state is None:
			x = gm.prompt('enter r to randomize or rl_output values')
			if x == 'r':
				sampled_output = np.zeros(len(self._actions), dtype=float)
				for idx, action in enumerate(self._actions):
					sampled_output[idx] = np.random.uniform(action.min_space, action.max_space, size=None)
			else:
				sampled_output = np.array([float(_) for _ in x.split(' ')])
			state = {
				'rl_output': sampled_output,
			}
		else:
			sampled_output = state['rl_output']
		gm.speak(f'taking actions from continuous action-space: {sampled_output}...')
		self.step(state)


	# returns continous action space of type Box
	# defined from action components' min and max space vars
	def get_space(self):
		nActions = len(self._actions)
		min_space = np.zeros(nActions)
		max_space = np.zeros(nActions)
		for idx in range(nActions):
			min_space[idx] = self._actions[idx].min_space
			max_space[idx] = self._actions[idx].max_space
		return spaces.Box(min_space, max_space)
