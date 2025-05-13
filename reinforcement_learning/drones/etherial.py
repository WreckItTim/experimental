from drones.drone import Drone
from component import _init_wrapper
import numpy as np
import utils.global_methods as gm
import map_data.map_methods as mm

# drone launched is etherial in the sense that this only keeps track of:
	# (1) position 
	# (2) collisions
# no actual drone, simualted or real, is flown though
class Etherial(Drone):
	@_init_wrapper
	def __init__(self,
			map_component, # used to check for collisions
			check_for_collision = True, 
			check_for_bounds = True,
			stop_at_invalid = False,
			step_length = 1,
		):
		super().__init__()
		self._collided = False
		self._out_of_bounds = False
		self.reset_kinematics()

	def reset_kinematics(self):
		self._x = 0
		self._y = 0
		self._z = 0
		self._direction = 0
		
	def rotate_right(self):
		direction = self._direction + 1
		if direction == 4:
			direction = 0
		#print('rot right', 'from', self._direction , 'to', direction)
		self.set_direction(direction)

	def rotate_left(self):
		direction = self._direction - 1
		if direction == -1:
			direction = 3
		#print('rot left', 'from', self._direction , 'to', direction)
		self.set_direction(direction)
		
	def rotate_180(self):
		if self._direction == 0:
			direction = 2
		elif self._direction == 1:
			direction = 3
		elif self._direction == 2:
			direction = 0
		else: #self._direction == 3:
			direction = 1
		#print('rot 180', 'from', self._direction , 'to', direction)
		self.set_direction(direction)

	def move_forward(self, magnitude):
		x, y, z = self._x, self._y, self._z
		if self._direction == 0:
			y += magnitude
		elif self._direction == 1:
			x += magnitude
		elif self._direction == 2:
			y -= magnitude
		else:# self._direction == 3:
			x -= magnitude
		#print('move forw', magnitude, 'from', [self._x, self._y, self._z], 'to', [x, y, z])
		self.move(x, y, z)

	def move_backward(self, magnitude):
		x, y, z = self._x, self._y, self._z
		if self._direction == 0:
			y -= magnitude
		elif self._direction == 1:
			x -= magnitude
		elif self._direction == 2:
			y += magnitude
		else:# self._direction == 3:
			x += magnitude
		#print('move back', magnitude, 'from', [self._x, self._y, self._z], 'to', [x, y, z])
		self.move(x, y, z)

	def move_left(self, magnitude):
		x, y, z = self._x, self._y, self._z
		if self._direction == 0:
			x -= magnitude
		elif self._direction == 1:
			y += magnitude
		elif self._direction == 2:
			x += magnitude
		else:# self._direction == 3:
			y -= magnitude
		#print('move left', magnitude, 'from', [self._x, self._y, self._z], 'to', [x, y, z])
		self.move(x, y, z)

	def move_right(self, magnitude):
		x, y, z = self._x, self._y, self._z
		if self._direction == 0:
			x += magnitude
		elif self._direction == 1:
			y -= magnitude
		elif self._direction == 2:
			x -= magnitude
		else:# self._direction == 3:
			y += magnitude
		#print('move right', magnitude, 'from', [self._x, self._y, self._z], 'to', [x, y, z])
		self.move(x, y, z)

	def move_up(self, magnitude):
		x, y, z = self._x, self._y, self._z
		z += magnitude
		#print('move up', magnitude, 'from', [self._x, self._y, self._z], 'to', [x, y, z])
		self.move(x, y, z)

	def move_down(self, magnitude):
		x, y, z = self._x, self._y, self._z
		z -= magnitude
		#print('move down', magnitude, 'from', [self._x, self._y, self._z], 'to', [x, y, z])
		self.move(x, y, z)
	
	# check if has collided
	def check_collision(self):
		collided = self._collided
		self._collided = False
		return collided 
	
	# check if has collided
	def check_out_of_bounds(self):
		out_of_bounds = self._out_of_bounds
		self._out_of_bounds = False
		return out_of_bounds 
	
	# resets on episode
	def reset(self, state=None):
		self.check_collision()
		self.check_out_of_bounds()
		self.reset_kinematics()

	def move(self, x, y, z):
		#print('move', x, y, z)
		self.teleport(x, y, z, self._direction, not self.check_for_collision, self.check_for_bounds)

	# moves to position
	# must check for colliision on the way
	# stop_at_invalid will take the movements until right before collision or out_of_bounds
	# , otherwise will not take any movements if collision or out_of_bounds
	def teleport(self, x, y, z, direction, ignore_collision, check_bounds, stabelize=True):
		#print('teleport', x, y, z)
		valid = True
		if not ignore_collision or check_bounds:
			current_position = np.array(self.get_position()).astype(float)
			last_valid_position = current_position
			displacement = np.array([x, y, z]) - current_position
			distance = np.linalg.norm(displacement)
			if distance >= self.step_length:
				n_steps = int(distance / self.step_length)
				step_vector = self.step_length * displacement / distance
				for step in range(n_steps):
					proposed_position = current_position + step_vector
					if check_bounds:
						self._out_of_bounds = self._map.out_of_bounds(*[int(x) for x in proposed_position])
						if self._out_of_bounds:
							valid = False
							break
					if not ignore_collision:
						self._collided = self._map.in_object(*[int(x) for x in proposed_position])
						if self._collided:
							valid = False
							break
					current_position = proposed_position
					if current_position[0] % 2 == 0 and current_position[1] % 2 == 0 and current_position[2] % 4 == 0:
						last_valid_position = current_position
			if valid:
				proposed_position = np.array([x, y, z])
				if check_bounds:
					self._out_of_bounds = self._map.out_of_bounds(*[int(x) for x in proposed_position])
					if self._out_of_bounds:
						valid = False
				if valid and not ignore_collision:
					self._collided = self._map.in_object(*[int(x) for x in proposed_position])
					if self._collided:
						valid = False
		self.set_direction(direction)
		if valid:
			self.set_position(int(x), int(y), int(z))
			#print('valid', int(x), int(y), int(z))
		else:
			if self.stop_at_invalid:
				self.set_position(*[int(x) for x in last_valid_position])
				#print('last_valid_position', [int(x) for x in last_valid_position])
				self._out_of_bounds = False
				self._collided = False
			else:
				self.set_position(*[int(x) for x in proposed_position])
				#print('proposed_position', [int(x) for x in proposed_position])
		
	def get_position(self):
		return [self._x, self._y, self._z]

	def get_direction(self):
		return self._direction

	def get_yaw(self):
		return mm.direction_to_yaw(self._direction)
	
	def set_direction(self, direction):
		self._direction = direction

	def set_position(self, x, y, z):
		self._x = x
		self._y = y
		self._z = z