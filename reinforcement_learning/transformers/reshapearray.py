from transformers.transformer import Transformer
import numpy as np
from component import _init_wrapper
from observations.vector import Vector
from observations.image import Image
from observations.array import Array

# reshapes numpy array of an observation and casts to new data type
class ReshapeArray(Transformer):
	# constructor
	@_init_wrapper
	def __init__(self, 
			  new_shape, # [dim1, dim2, ..., dimN]
			  new_type, # 'vector' 'image' 'array'
			  ):
		super().__init__()

	def transform(self, observation):
		#observation.check(Image)
		np_array = observation.to_numpy()
		np_array = np_array.reshape(self.new_shape)
		if self.new_type in ['vector']:
			ObservationType = Vector
		if self.new_type in ['image']:
			ObservationType = Image
		if self.new_type in ['array']:
			ObservationType = Array
		return ObservationType(np_array)