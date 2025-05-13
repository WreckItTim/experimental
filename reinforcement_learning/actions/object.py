from actions.action import Action
from component import _init_wrapper

class Object(Action):
	@_init_wrapper
	def __init__(self, 
			  object_component,
			  action_name,
			  action_params={},
			  ):
		pass

	# rotate yaw at fixed rate
	def step(self, state=None):
		getattr(self._object, self.action_name)(**self.action_params)