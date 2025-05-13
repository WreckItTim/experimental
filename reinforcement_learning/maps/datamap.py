from maps.map import Map
from component import _init_wrapper
import map_data.map_methods as mm
import utils.global_methods as gm

# no map is rendered 
	# looks up object positions and roofs for collisions from dictionary object
class DataMap(Map):
	@_init_wrapper
	def __init__(self, x_bounds, y_bounds, z_bounds):
		self._datamap = gm.get_global('datamap')

	def in_object(self, x, y, z):
		return self._datamap.in_object(x, y, z)
	
	def out_of_bounds(self, x, y, z):
		return self._datamap.out_of_bounds(x, y, z, self.x_bounds, self.y_bounds, self.z_bounds)

	# will get lowest z-point without being inside object at given x,y
	# this includes the floor (which will be lowest point found)
	def get_roof(self, x, y):
		return self._datamap.get_roof(x, y)