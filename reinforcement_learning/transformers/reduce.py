from transformers.transformer import Transformer
import numpy as np
from component import _init_wrapper
from observations.vector import Vector
from observations.image import Image
from observations.array import Array
import numpy as np
import skimage

# reshapes numpy array of an observation and casts to new data type
class Reduce(Transformer):
	# constructor
	@_init_wrapper
	def __init__(self, 
			  reduce_method, # [np.max, np.min, np.mean, ...]
			  reduce_shape, # block size [ax1, ..., axN]
			  ):
		super().__init__()

	def transform(self, observation):
		data = observation.to_numpy()
		data = skimage.measure.block_reduce(data, self.reduce_shape, self.reduce_method)
		observation.set_data(data)
