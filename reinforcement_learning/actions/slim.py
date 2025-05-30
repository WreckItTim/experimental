from actions.action import Action
from component import _init_wrapper
import math
import utils.global_methods as gm

# coutinuouts output will scale resolution of input sensors
# sensors should start at maximum scale
class Slim(Action):
	# constructor takes a list of components to scale down resolution
	@_init_wrapper
	def __init__(self, 
				model_component, # list of components to call scale_down() on
				# set these values for continuous actions
				# # they determine the possible ranges of output from rl algorithm
				min_slim = 0.125,
				min_space = -1, # will scale base values by this range from rl_output
				max_space = 1, # min_space to -1 will allow you to reverse positive motion
				active = True, # False will just return default behavior
			):
		pass
		
	# move at input rate
	def step(self, state, execute=True):
		if not self.active:
			rho = 1
		else:
			rl_output = state['rl_output'][self._idx]
			rho = round(max(self.min_slim, rl_output), 4)
		self._rho = rho # give access to other components to last rho
		self._model._sb3model.slim = rho
		for module in self._model._sb3model.actor.modules():
			if 'Slim' in str(type(module)):
				module.slim = rho
		return {'slim':rho}