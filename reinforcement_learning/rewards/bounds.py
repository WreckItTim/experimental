from rewards.reward import Reward
from component import _init_wrapper
import utils.global_methods as gm

# rewards/terminatess based on x,y,z boundaries
class Bounds(Reward):
	# constructor
	@_init_wrapper
	def __init__(self, 
                 drone_component, 
                 x_bounds,
                 y_bounds,
                 z_bounds,
				 terminate=True, # =True will terminate episodes when oob
	):
		super().__init__()
		self._datamap = gm.get_global('datamap')

	# calculates rewards from agent's current state (call to when taking a step)
	def step(self, state):
		x, y, z = self._drone.get_position()
		oob = self._datamap.out_of_bounds(x, y, z, self.x_bounds, self.y_bounds, self.z_bounds)
		done = oob
		value = -1 * oob
		if done and self.terminate:
			state['termination_reason'] = 'bounds'
			state['termination_result'] = 'failure'
		return value, done and self.terminate
