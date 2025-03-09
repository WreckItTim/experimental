# used to handle array observations saved as np arrays
from observations.observation import Observation
import matplotlib.pyplot as plt
import global_methods as md

class Array(Observation):

	# constructor
	def __init__(self, _data, data_path=None):
		super().__init__(_data=_data, data_path=data_path)

	# displays observation
	def display(self):
		print(self._data)
		
	def write(self, path_without_ftype, ftype='.p'):
		md.pk_write(self._data, path_without_ftype + '.p')
