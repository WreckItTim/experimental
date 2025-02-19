from observers.observer import Observer
from component import _init_wrapper
from gymnasium import spaces
import numpy as np

class Empty(Observer):
	
	# empty observer returns empty observation
	@_init_wrapper
	def __init__(
		self, 
	):
		super().__init__()
		self._output_shape = [1]
		
	def null_data(self):
		np.zeros(self._output_shape, dtype=float)

	# gets observations
	def step(self, state=None):
		return self.null_data(), 'empty'

	# returns box space with proper dimensions
	def get_space(self):
		return spaces.Box(0, 1, shape=self._output_shape, dtype=float)
