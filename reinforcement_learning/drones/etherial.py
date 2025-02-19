from drones.drone import Drone
from component import _init_wrapper
import numpy as np
import rl_utils as _utils


# drone launched is etherial in the sense that this only keeps track of:
	# (1) position 
	# (2) collisions
# no actual drone, simualted or real, is flown though
class _Etherial(Drone):
	@_init_wrapper
	def __init__(self,
			  map_component, # used to check for collisions
			  discretized = True,
			  ):
		super().__init__()
		self._collided = False
		self._oob = False
		self._pos = np.array([0,0,0])
		self._yaw = 0
	
	# check if has collided
	def check_collision(self):
		collided = self._collided
		self._collided = False
		return collided 
	
	# check if has collided
	def check_bounds(self):
		oob = self._oob
		self._oob = False
		return oob 
	
	# resets on episode
	def reset(self, state=None):
		self.check_collision()
		self.check_bounds()

	# teleports to position, yaw in radians
	def teleport(self, x, y, z, yaw, ignore_collision=False, check_bounds=False, stabelize=True):
		step_length=1
		if not ignore_collision or check_bounds:
			current_pos = np.array(self.get_position()).astype(float)
			displacement = np.array([x, y, z]) - current_pos
			distance = np.linalg.norm(displacement)
			if distance % 1 != 0:
				_utils.speak('NEGATIVE MOVEMENT')
				_utils.progress('NEGATIVE MOVEMENT', 'uh oh')
				ini = input()
			if distance > step_length:
				n_steps = int(distance / step_length)
				step_vector = step_length * displacement / distance
				for step in range(n_steps):
					current_pos += step_vector
					if check_bounds:
						self._oob = self._map.check_bounds(*current_pos)
						if self._oob:
							break
					if not ignore_collision:
						self._collided = self._map.in_object(*current_pos)
						if self._collided:
							break
		self.set_pos([x, y, z], ignore_collision, check_bounds)
		self.set_yaw(yaw)
		
	def set_pos(self, pos, ignore_collision=False, check_bounds=False):
		if self.discretized:
			pos = np.round(pos)
		if check_bounds:
			self._oob = self._oob or self._map.check_bounds(*pos)
		if not ignore_collision:
			self._collided = self._collided or self._map.in_object(*pos)
		self._pos = np.array(pos)
		
	def set_yaw(self, yaw):
		# discretize
		if self.discretized:
			self._yaw = round(yaw/(np.pi/2))*np.pi/2
		else:
			self._yaw = yaw
		while self._yaw < -1*np.pi:
			self._yaw  += 2*np.pi
		while self._yaw >= np.pi:
			self._yaw -= 2*np.pi
		
	# rotates along z-axis, yaw in radians offset from current yaw
	def rotate(self, yaw):
		self.set_yaw(self.yaw + yaw)

	# get (x, y, z) positon, z is negative for up, x is positive for forward, y is positive for right (from origin)
	def get_position(self):
		if self.discretized:
			return [int(coord) for coord in self._pos]
		return [float(coord) for coord in self._pos]

	# get rotation about the z-axis (yaw), returns in radians between -pi to +pi
	def get_yaw(self):
		return self._yaw
	
	def disconnect(self, state=None):
		pass
