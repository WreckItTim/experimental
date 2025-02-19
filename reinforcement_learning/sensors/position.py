from sensors.sensor import Sensor
from observations.vector import Vector
from component import _init_wrapper
import numpy as np

# gets position from a miscellaneous component
# can send in a second component to get position and distance between two
class Position(Sensor):
	
	@_init_wrapper
	def __init__(self,
                 misc_component, 
                 misc2_component=None, 
				 prefix = '',
				 transformers_components = None,
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
		return self.create_obj([0,0,0])
		
	# get information reltaive between current and objective point
	def step(self, state=None):
		data = []
		if self._misc2 is None:
			position = self._misc.get_position()
			data.append(position[0])
			data.append(position[1])
			data.append(position[2])
		else:
			position1 = np.array(self._misc.get_position())
			position2 = np.array(self._misc2.get_position())
			distance_vector = position2 - position1
			data.append(distance_vector[0])
			data.append(distance_vector[1])
			data.append(distance_vector[2])

		observation = self.create_obj(data)
		transformed = self.transform(observation)
		return transformed