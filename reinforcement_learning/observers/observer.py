# abstract class used to handle observations to input into rl algo
from component import Component
from os.path import exists
from os import makedirs

class Observer(Component):
	# constructor
	def __init__(self):
		self.connect_priority = -2 # order: sensors, observers, environment, model

	# returns observation transcribed for input into RL model
	def step(self, state=None):
		raise NotImplementedError

	# returns observation space for this observer
	def get_space(self):
		raise NotImplementedError