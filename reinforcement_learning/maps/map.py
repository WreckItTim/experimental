# abstract class used to handle maps - where the drone is flying in
# this is the part that connects the program to the real application
from component import Component
import map_data.map_methods as mm
import utils.global_methods as gm


class Map(Component):
	# constructor
	def __init__(self):
		self._x_bounds = gm.get_global('x_bounds')
		self._y_bounds = gm.get_global('y_bounds')
		self._z_bounds = gm.get_global('z_bounds')
		self._datamap = gm.get_global('datamap')

	def in_object(self, x, y, z):
		return self._datamap.in_object(x, y, z)
	
	def out_of_bounds(self, x, y, z):
		return self._datamap.out_of_bounds(x, y, z, self._x_bounds, self._y_bounds, self._z_bounds)

	# will get lowest z-point without being inside object at given x,y
	# this includes the floor (which will be lowest point found)
	def get_roof(self, x, y):
		return self._datamap.get_roof(x, y)

	def connect(self):
		super().connect()