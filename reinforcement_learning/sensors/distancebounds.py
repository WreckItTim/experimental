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
			  ):
		super().__init__(offline, memory)

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
		position = np.array(self._drone.get_position())
		yaw = np.array(self._drone.get_yaw())
		while yaw < 0:
			yaw += 2*np.pi
		while yaw >= 2*np.pi:
			yaw -= 2*np.pi
		yaw = round(yaw/(np.pi/2))
		if yaw == 0: # facing forward
			xy_distance = abs(position[0] - self.x_bounds[1])
		if yaw == 1: # facing right
			xy_distance = abs(position[1] - self.y_bounds[1])
		if yaw == 2: # facing backward
			xy_distance = abs(position[0] - self.x_bounds[0])
		if yaw == 3: # facing left
			xy_distance = abs(position[1] - self.y_bounds[0])
		data.append(xy_distance)
		if self.include_z:
			z_distance_up = abs(position[2] - self.z_bounds[0])
			z_distance_down = abs(position[2] - self.z_bounds[1])
			data.append(z_distance_up)
			data.append(z_distance_down)
			
		observation = self.create_obj(data)
		transformed = self.transform(observation)
		return transformed