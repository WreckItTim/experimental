from maps.map import Map
from component import _init_wrapper

# no map is rendered 
	# looks up object positions and roofs for collisions from dictionary object
class _Etherial(Map):
	@_init_wrapper
	def __init__(self,
				 # tells the height of collidable objects
				 rooftops_component=None,
				 x_bounds=None, 
				 y_bounds=None,
				 z_bounds=None,
				 ):
		super().__init__()

	def in_object(self, x, y, z):
		if self._rooftops is not None:
			return self._rooftops.in_object(x, y, z)
		return False
	
	def check_bounds(self, x, y, z):
		return (x < self.x_bounds[0] or x > self.x_bounds[1] or 
				  y < self.y_bounds[0] or y > self.y_bounds[1] or
				  z < self.z_bounds[0] or z > self.z_bounds[1])

	# will get lowest z-point without being inside object at given x,y
	# this includes the floor (which will be lowest point found)
	def get_roof(self, x, y):
		if self._rooftops is not None:
			return self._rooftops.get_roof(x, y)
		return 0