from sensors.sensor import Sensor
from observations.vector import Vector
from component import _init_wrapper
import numpy as np
import math

# gets position from a miscellaneous component
# can send in a second component to get positoin and distance between two
class DistanceBounds(Sensor):
	
	@_init_wrapper
	def __init__(self,
			drone_component, 
			x_bounds,
			y_bounds,
			z_bounds,
			transformers_components = None,
			include_z = False,
			offline = False,
			memory='all', # none all part
			out_size=(1),
		):
		super().__init__(offline, memory)

	def connect(self, state=None):
		super().connect(state)
		if self.include_z:
			self.out_size = (3)

	def create_obj(self, data):
		observation = Vector(
			_data = data,
		)
		return observation
	
	def get_null(self):
		if self.include_z:
			return self.create_obj([0, 0, 0])
		return self.create_obj([0])
	
	# get information reltaive between current and objective point
	def step(self, state=None):
		data = []
		x, y, z = np.array(self._drone.get_position())
		direction = np.array(self._drone.get_direction())
		if direction == 0:
			xy_distance = abs(y - self.y_bounds[1])
		elif direction == 1:
			xy_distance = abs(x - self.x_bounds[1])
		elif direction == 2:
			xy_distance = abs(y - self.y_bounds[0])
		else:# direction == 3:
			xy_distance = abs(x - self.x_bounds[0])
		data.append(xy_distance)
		if self.include_z:
			z_distance_up = abs(z - self.z_bounds[1])
			z_distance_down = abs(z - self.z_bounds[0])
			data.append(z_distance_up)
			data.append(z_distance_down)
			
		observation = self.create_obj(data)
		transformed = self.transform(observation)
		return transformed