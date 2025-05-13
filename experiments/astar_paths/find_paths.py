import os
root_dir = '/home/tim/Dropbox/experimental/' # your path here where to parent directory where repos are
local_dir = '/home/tim/local/'
os.chdir(root_dir)
import sys
sys.path.append(root_dir)
import map_data.map_methods as mm
import shortest_path.shortest_methods as sm
import utils.global_methods as gm
import numpy as np
import random
import matplotlib.pyplot as plt
initial_locals = locals().copy() # will exclude these parameters from config parameters written to file

map_name = 'AirSimNH'
motion = '2d'
region = 'houses_all'
job_name = 'null'
random_seed = 777
checkpoint_freq = 1
job_name = 'null'
min_distance = 6
max_distance = np.inf
delta_x = 777 # allowed deviation from start x value
delta_y = 777 # allowed deviation from start y value
go_straight = False # will only find paths with no objects in way
n_paths = 1_000 # number of random paths to find (including old paths at file out)
z_level = 4 # if using 2d motion, the fixed z-value to move horizontally at
astar_version = 'v1' # determines set of actions to take and logging
rooftops_version = 'v1'
overwrite = False
max_iterations = 100_000
ckpt_freq = 20
x_res = 2 # all randomized x-values will be an integer multiple of this
y_res = 2 # all randomized y-values will be an integer multiple of this
z_res = 4 # all randomized z-values will be an integer multiple of this

if len(sys.argv) > 1:
    arguments = gm.parse_arguments(sys.argv[1:])
    locals().update(arguments)
    
out_dir = f'map_data/astar_paths/{astar_version}/{map_name}/{motion}/{region}/{random_seed}/'
out_path = f'{out_dir}paths.p'
failed_out_path = f'{out_dir}failed.p'
os.makedirs(out_dir, exist_ok=True)

# save all params to file
all_locals = locals()
new_locals = {k:v for k, v in all_locals.items() if (not k.startswith('__') and k not in initial_locals and k not in ['initial_locals','all_locals'])}
params = new_locals.copy()
gm.pk_write(params, f'{out_dir}params.p')

print('running job with params', params)
gm.set_global('local_dir', local_dir)
gm.progress(job_name, 'started')

path_names = {}
all_paths_path = f'map_data/astar_paths/{astar_version}/{map_name}/{motion}/{region}/paths.p'
if os.path.exists(all_paths_path):
    all_paths = gm.pk_read(all_paths_path)
    for path_idx in range(len(all_paths['paths'])):
        start = all_paths['starts'][path_idx]
        target = all_paths['targets'][path_idx]
        start_pos = list(start)
        end_pos = list(target)
        name1 = str(start_pos)+str(end_pos)
        name2 = str(end_pos)+str(start_pos)
        if name1 not in path_names:
            path_names[name1] = True
            path_names[name2] = True
failed_paths = []
failed_paths_path = f'map_data/astar_paths/{astar_version}/{map_name}/{motion}/{region}/failed.p'
if os.path.exists(failed_paths_path):
    failed_paths = gm.pk_read(failed_paths_path)
    for path_idx in range(len(failed_paths)):
        start, target, outer_iterations, result = failed_paths[path_idx]
        start_pos = list(start)
        end_pos = list(target)
        name1 = str(start_pos)+str(end_pos)
        name2 = str(end_pos)+str(start_pos)
        if name1 not in path_names:
            path_names[name1] = True
            path_names[name2] = True

# make datamap to find paths in
datamap = mm.DataMap(map_name, rooftops_version)

# set random seed for fidning random paths
gm.set_random_seed(random_seed)

# set output path
try:
    paths = gm.pk_read(out_path)
except:
    paths = []

# get bounds of map
x_bounds, y_bounds, z_bounds = datamap.get_bounds(region, motion=motion)

# get needed astar version params
if astar_version in ['v1']:
    roof_collision = 2 # the allowed vertical distance from the roof before triggering collision, in z

while len(paths) < n_paths:
    
    # randomize x,y,z start and target points until find valid one
    while True:
        # randomize start and taret points
        x1 = random.randint(*x_bounds)
        if delta_x == 0:
            x2 = x1
        else:
            x2 = random.randint(*x_bounds)
        y1 = random.randint(*y_bounds)
        if delta_y == 0:
            y2 = y1
        else:
            y2 = random.randint(*y_bounds)
        if motion in ['2d']:
            z1 = z2 = z_level
        if motion in ['3d']:
            z1 = random.randint(*z_bounds)
            z2 = random.randint(*z_bounds)
        target = np.array([x1, y1, z1])
        start = np.array([x2, y2, z2])
        
        if go_straight and y2 < y1:
            continue

        if go_straight:
            temp_y = y1
            while(True):
                if datamap.in_object(x1, temp_y + y_res, z_level, collision_threshold=roof_collision):
                    break
                if temp_y + y_res > y2:
                    break
                temp_y += y_res
            y2 = temp_y
    
        # check if same point
        if x1 == x2 and y1 == y2 and z1 == z2:
            continue

        # check delta-x and delta-y thresholds
        if np.abs(x2-x1) > delta_x or np.abs(y2-y1) > delta_y:
            continue
    
        # check if valid resolution
        if x1%x_res != 0 or x2%x_res != 0 or y1%y_res != 0 or y2%y_res != 0 or z1%z_res != 0 or z2%z_res != 0:
            continue
    
        # check if colliding with object
        if (datamap.in_object(x1, y1, z1, collision_threshold=roof_collision) or
            datamap.in_object(x2, y2, z2, collision_threshold=roof_collision)):
            continue
    
        # check if distance meets our given constraint
        euclidean = np.linalg.norm(target-start)
        if euclidean < min_distance or euclidean > max_distance:
            continue

        # check if unique path
        name1 = str(start)+str(target)
        name2 = str(target)+str(start)
        if name1 in path_names or name2 in path_names:
            continue
    
        # if we reached here then we have a valid proposed start and target position
        break
    
    # attempt to find path
    print('searching for path between', start, 'and', target)
    path, outer_iterations, result = sm.get_astar_path(start, target, motion, datamap, 
        x_bounds, y_bounds, z_bounds, astar_version=astar_version, max_iterations=max_iterations
    )
    
    # check if path was successfull
    if result in ['success']:
        paths.append([start, target, path])
        print('path succesfully found! length =', len(paths))
    else:
        failed_paths.append([start, target, outer_iterations, result])
        print('path not found =(')

    if len(paths) % ckpt_freq == 0:
        gm.pk_write(paths, out_path)
        gm.pk_write(failed_paths, failed_out_path)
        gm.progress(job_name, f'{100*len(paths)/n_paths:.2f}%')
        
    path_names[name1] = True
    path_names[name2] = True

gm.pk_write(True, f'{out_dir}complete.p')
gm.progress(job_name, 'complete')