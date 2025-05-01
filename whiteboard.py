import psutil
import numpy as np
import gc

# check RAM 
def check_mem():
    mem_dic = psutil.virtual_memory()._asdict()
    for key in mem_dic:
        mem_dic[key] = round(mem_dic[key]*1e-9)
    return mem_dic
print(f'start {check_mem()}')

toggle = True # False does not free memory but True does free up memory

# for i in range(2):
#     print()
#     print('iter', i)

#     data = {x:np.full((10000, 10000), 1, dtype=float) for x in range(2**3)}
#     print(f'make 2d float {check_mem()}')
#     data.clear()
#     print(f'free 2d float {check_mem()}')
    
#     data = {x:np.full((3, 144, 256), 1, dtype=float) for x in range(2**12)}
#     print(f'make 3d float {check_mem()}')
#     data.clear()
#     print(f'free 3d float {check_mem()}')
    
#     data = {x:np.full((10000, 10000), 1, dtype=np.uint8) for x in range(32)}
#     print(f'make 2d uint {check_mem()}')
#     data.clear()
#     print(f'free 2d uint {check_mem()}')
    
#     data = {x:np.full((3, 144, 256), 1, dtype=np.uint8) for x in range(2**15)}
#     print(f'make 3d uint {check_mem()}')
#     data.clear() # needed with toggle on otherwise does not clear mem
#     if toggle: # both lines below are needed with toggle on otherwise does not clear mem
#         data = {x:np.full((10000, 10000), 1, dtype=float) for x in range(1)}
#         data.clear()
#     gc.collect() # this doesn't affect anything, put it here to show I tried it
#     print(f'free 3d uint {check_mem()}')

# create data to save to file
data = {}
n_layers = 4
exp = 16 # 2**exp number of elements in each layer
layers = [[i for i in range(2**(exp-n_layers))]]
for j in range(n_layers-1):
    layers.append([0, 1])
def recursive_allocate(_data, layer, layers):
    for item in layer:
        if len(layers) == 0:
            _data[item] = np.full((3, 144, 256), 1, dtype=np.uint8)
        else:
            _data[item] = {}
            recursive_allocate(_data[item], layers[0], layers[1:])
recursive_allocate(data, layers[0], layers[1:])
print(f'make  {check_mem()}')

# free data 
def recursive_free(_data):
    keys = list(_data.keys())
    for key in keys:
        value = _data[key]
        if isinstance(value, dict):
            recursive_free(value)
        else:
            _data[key] = np.full((3, 144, 256), 1, dtype=float)
            #del _data[key]
    #_data.clear()
    #gc.collect()
recursive_free(data)
data.clear()
if toggle: # both lines below are needed with toggle on otherwise does not clear mem
    data = {x:np.full((10000, 10000), 1, dtype=float) for x in range(1)}
    data.clear()
gc.collect()
print(f'free  {check_mem()}')
