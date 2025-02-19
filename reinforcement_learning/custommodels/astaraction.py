# abstract class used to handle RL model
from custommodels.custommodel import CustomModel
from component import _init_wrapper
import numpy as np

# moves on set path
# used to evaluate astar paths
class AstarAction(CustomModel):
	# constructor
	@_init_wrapper
	def __init__(self,
			  spawner_component,
		):
		super().__init__(actor_critic=False)

	def predict(self, observation_data):
		self._step += 1
		if (self._step >= len(self._spawner._path)):
			return np.array(0)
		x, y, z = self._spawner._path[self._step]['position']
		yaw = self._spawner._path[self._step]['direction'] * np.pi / 2
		action = self._spawner._path[self._step]['action']
		return np.array(action)

	def reset(self, state=None):
		self._step = 0