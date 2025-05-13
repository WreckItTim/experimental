from sensors.sensor import Sensor
from observations.vector import Vector
from component import _init_wrapper
import numpy as np
import utils.global_methods as gm
import map_data.map_methods as mm
import math

# gets orientation from a miscellaneous component
# yaw is expressed betwen [0, 2pi)  
# pass in a second component to get the orienation difference between the two
class Orientation(Sensor):
	
	@_init_wrapper
	def __init__(self,
			misc_component,
			misc2_component=None,
			prefix = '',
			transformers_components = None,
			offline = False,
			calc_diff = False,
			memory='all', # none all part
			out_size=(1),
		):
		super().__init__(offline, memory)

	def create_obj(self, data):
		observation = Vector(
			_data = data,
		)
		return observation
	
	def get_null(self):
		return self.create_obj([0])

	# get state information from drone
	def step(self, state=None):
		data = []
		if self._misc2 is None:
			yaw = self._misc.get_yaw()
			data.append(yaw)
		else:
			position1 = np.array(self._misc.get_position())
			position2 = np.array(self._misc2.get_position())
			distance_vector = position2 - position1
			yaw_1_2 = math.atan2(distance_vector[1], distance_vector[0])
			#print('orientation', position1, position2, distance_vector, yaw_1_2)
			if self.calc_diff:
				yaw1 = self._misc.get_yaw()
				yaw_diff = yaw_1_2 - yaw1
				data.append(mm.fix_yaw(yaw_diff))
			else:
				data.append(mm.fix_yaw(yaw_1_2))
		observation = self.create_obj(data)
		transformed = self.transform(observation)
		return transformed
