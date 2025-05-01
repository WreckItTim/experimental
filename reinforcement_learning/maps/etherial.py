from maps.map import Map
from component import _init_wrapper

# no map is rendered 
	# looks up object positions and roofs for collisions from dictionary object
class Etherial(Map):
	@_init_wrapper
	def __init__(self):
		super().__init__()