from sensors.sensor import Sensor
from observations.vector import Vector
from component import _init_wrapper

# gets facing direction of drone
class Direction(Sensor):
	
	@_init_wrapper
	def __init__(self,
			drone_component,
			prefix = '',
			transformers_components = None,
			offline = False,
			out_size=(1),
			memory='all', # none all part
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
		data = [self._drone.get_direction()]
		observation = self.create_obj(data)
		transformed = self.transform(observation)
		return transformed
