from maps.map import Map
from component import _init_wrapper

class Empty(Map):
    @_init_wrapper
    def __init__(self,
                 ):
        super().__init__()