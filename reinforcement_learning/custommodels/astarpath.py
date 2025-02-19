# abstract class used to handle RL model
from custommodels.custommodel import CustomModel
from component import _init_wrapper
import numpy as np

# moves on set path
# used to evaluate astar paths
class AstarPath(CustomModel):
	# constructor
	@_init_wrapper
	def __init__(self,
			  spawner_component,
		):
		super().__init__(actor_critic=False)
		self._last = [[0,0,0,0]]

	def predict(self, observation_data):
		self._step += 1
		if (self._step >= len(self._spawner._path)):
			return self._last
		x, y, z = self._spawner._path[self._step]['position']
		yaw = self._spawner._path[self._step]['direction'] * np.pi / 2
		rl_output = []
		rl_output.append([x, y, z, yaw])
		self._last = np.array(rl_output)
		return self._last

	def reset(self, state=None):
		self._step = 0