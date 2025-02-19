from actions.action import Action
from component import _init_wrapper
import math

# specifies x,y,z,yaw coordinates to move to
class MoveTo(Action):
	@_init_wrapper
	def __init__(self, 
				drone_component,
				teleport_on_execute=True,
				min_space = -1, # will scale base values by this range from rl_output
				max_space = 1, # min_space to -1 will allow you to reverse positive motion
				active = True, # False will just return default behavior
			):
		pass
		
	# move at input rate
	def step(self, state, execute=True):
		if not self.active:
			return {}
		rl_output = state['rl_output'][self._idx]
		x, y, z, yaw = rl_output
		# move calculated rate
		if execute:
			if self.teleport_on_execute:
				self._drone.teleport(x, y, z, yaw )
		return {'x':x, 'y':y, 'z':z, 'yaw':yaw}