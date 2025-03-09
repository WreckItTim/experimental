from others.other import Other
from component import _init_wrapper
import numpy as np
import random
import global_methods as md
import os

# reads saved observations from dictionary
# each dictionary is saved by a subset
	# so will load at each step if changed subset
class DataDict(Other):
	@_init_wrapper
	def __init__(self, 
					data_dir,
					id_name,
					x_vals,
					y_vals,
					z_vals,
					yaw_vals,
				):
		pass

	def connect(self, state=None):
		super().connect(state)
		self.read_data_dict()

	def set_data(self, keys, data):
		dlevel = self._data_dict
		for key in keys:
			if key not in dlevel:
				dlevel[key] = {}
		dlevel[key] = data

	def get_data(self, keys):
		dlevel = self._data_dict
		for key in keys:
			if key not in dlevel:
				return None
			dlevel = dlevel[key]
		return dlevel
		
	def read_data_dict(self):
		self._data_dict = {}
		file_names = os.listdir(self.data_dir)
		for file_name in file_names:
			if 'data_dict__' not in file_name:
				continue
			part_name = file_name.split('__')[1]
			parts = part_name.split('_')
			this_id_name = parts[0]
			if this_id_name not in self.id_name:
				continue
			data_dict = md.pk_read(f'{self.data_dir}{file_name}')
			for x in data_dict:
				if x not in self.x_vals:
					continue
				for y in data_dict[x]:
					if y not in self.y_vals:
						continue
					for z in data_dict[x][y]:
						if z not in self.z_vals:
							continue
						for yaw in data_dict[x][y][z]:
							if yaw not in self.yaw_vals:
								continue
							if x not in self._data_dict:
								self._data_dict[x] = {}
							if y not in self._data_dict[x]:
								self._data_dict[x][y] = {}
							if z not in self._data_dict[x][y]:
								self._data_dict[x][y][z] = {}
							observation = data_dict[x][y][z][yaw]
							self._data_dict[x][y][z][yaw] = observation