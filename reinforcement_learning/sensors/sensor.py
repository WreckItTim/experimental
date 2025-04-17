
from component import Component
import utils.global_methods as gm

# abstract class used to handle sensors
class Sensor(Component):
	# constructor, offline is a boolean that handles if sensor goes offline
	def __init__(self,
			  offline=False,
			  memory='all', # none all part
			  ):
		self.connect_priority = -1 # order: sensors, observers, environment, model

	# creates an observation object given a data array
	def create_obj(self, data):
		raise NotImplementedError
	
	# steps in environment to collect observation
	def step(self, state=None):
		raise NotImplementedError

	# applies transformations on observation object
	def transform(self, observation):
		if self._transformers is not None:
			for transformer in self._transformers:
				observation_conversion = transformer.transform(observation)
				if observation_conversion is not None:
					observation = observation_conversion
		return observation

	def debug(self, state=None):
		observation = self.step(state)
		observation.display()

	def connect(self, state=None):
		super().connect()
		
	def reset(self, state=None):
		if self._transformers is not None:
			for transformer in self._transformers:
				transformer.reset(state)