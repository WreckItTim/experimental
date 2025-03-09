from actors.actor import Actor
from random import randint
from component import _init_wrapper
import random
import global_methods as md
from gymnasium import spaces

# handles discrete actions - RL returns an index specifying which action to take
class Discrete(Actor):
	# constructor
	@_init_wrapper
	def __init__(self, 
			  actions_components=[],
			  ):
		super().__init__()
		self._type = 'discrete'

	# interpret action from RL
	def step(self, state):
		rl_output = state['rl_output']
		self._actions[rl_output].step(state)
		state['transcribed_action'] = self._actions[rl_output]._name

	# returns dioscrete action space of type Discrete
	def get_space(self):
		return spaces.Discrete(len(self._actions))