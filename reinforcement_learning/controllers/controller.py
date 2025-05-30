# abstract class used to handle all components
from component import Component
from configuration import Configuration
import utils.global_methods as gm

class Controller(Component):

	# constructor
	def __init__(self):
		# tell the init_wrapper to not add to configuration file as component or create a unique name
		self._add_timers = False
		self._set_name = False
		self._add_to_configuration = False

	# runs control on components
	def run(self):
		raise NotImplementedError

	def connect(self):
		super().connect()

	def save(self, path):
		json_dict = self._to_json()
		gm.write_json(json_dict, path)