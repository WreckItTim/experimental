import sys
sys.dont_write_bytecode = True
import os
dropbox_path = '/home/tim/Dropbox/'
sys.path.append(dropbox_path)
from commands.setup import *
instance_name, dropbox_path, local_path, data_path, models_path = setup(dropbox_path=dropbox_path)
from astar import *

working_sub = 'paths'
checkpoint_freq = 1

airsim_map = sys.argv[1]
motion = sys.argv[2]
region = sys.argv[3]
run_num = int(sys.argv[4])
min_distance = int(sys.argv[5])
max_distance = int(sys.argv[6])
region_version = int(sys.argv[7])
n_paths = int(sys.argv[8])
job_name = f'paths_{airsim_map}_{motion}_{region}_{run_num}_{min_distance}_{max_distance}_{region_version}_{n_paths}'
instance_seeds = {
    'hephaestus':1_000,
    'apollo':2_000,
    'fox':3_000,
    'flareon':4_000,
    'ace':5_000,
    'magma':6_000,
    'pyro':7_000,
    'torch':8_000,
    'phoenix':9_000,
    'tron':10_000,
}
random_seed = instance_seeds[instance_name] + run_num
random.seed(random_seed)
np.random.seed(random_seed)
resolution = 1

phase = 'collect'
#phase = 'analyze'

read_map = True
write_map = False
view_maps = False
make_edits = True
write_levels = True

dataset = airsim_map + '_' + motion + '_' + region
print(instance_name, dataset)


# there are 3 change of axis to display the roof top heat map below.
# 1. AirSim uses yx(-z) coordinates
# 2. Voxels outputs data in real world Unity xyz coordinates, inputting airsim coordinats
# 3. plt.imshow(2d-array) plots the second dimension along the horizontal and first along the vertical

# In[ ]:

gridpath_airsim = data_path + 'grids/grid_airsim_' + airsim_map + '.p'
gridpath_unity = data_path + 'grids/grid_unity_' + airsim_map + '.p'
roofpath_airsim = data_path + 'rooftops/rooftops_airsim_' + airsim_map + '.p'
roofpath_unity = data_path + 'rooftops/rooftops_unity_' + airsim_map + '.p'

if airsim_map in ['AirSimNH']:
    xrange, yrange, zrange = [-240, 240], [-240, 240], [-10, 40]
if airsim_map in ['Blocks']:
    xrange, yrange, zrange = [-140, 140], [-120, 100], [-10, 30]

if read_map:
    grid_airsim = pickle.load(open(gridpath_airsim, 'rb'))
    grid_unity = pickle.load(open(gridpath_unity, 'rb'))
    rooftops_airsim = pickle.load(open(roofpath_airsim, 'rb'))
    rooftops_unity = pickle.load(open(roofpath_unity, 'rb'))
    
else:

    if airsim_map in ['AirSimNH']:
        airsimnh_topleft = Voxels('voxels/airsimnh_topleft.binvox', floor=104, xmin=-250, ymin=0, zmin=-125) # unity world coords
        airsimnh_topright = Voxels('voxels/airsimnh_topright.binvox', floor=104, xmin=0, ymin=0, zmin=-125) # unity world coords
        airsimnh_botleft = Voxels('voxels/airsimnh_botleft.binvox', floor=104, xmin=-250, ymin=-250, zmin=-125) # unity world coords
        airsimnh_botright = Voxels('voxels/airsimnh_botright.binvox', floor=104, xmin=0, ymin=-250, zmin=-125) # unity world coords
        
        grid_unity = combine_voxels([
            airsimnh_topleft, airsimnh_topright, airsimnh_botleft, airsimnh_botright
        ], xrange, yrange, zrange, 1) # unity world coords

    if airsim_map in ['Blocks']:
        blocks_left = Voxels('voxels/blocks_left.binvox', floor=104, xmin=-250, ymin=-125, zmin=-125) # unity world coords
        blocks_right = Voxels('voxels/blocks_right.binvox', floor=104, xmin=0, ymin=-125, zmin=-125) # unity world coords
        grid_unity = combine_voxels([blocks_left, blocks_right], xrange, yrange, zrange, 1)
    
    grid_airsim = unity_to_airsim(grid_unity) # unity to airsim world coords
    rooftops_airsim = grid_to_rooftops(grid_airsim, roof=-1*zrange[1], floor=-1*zrange[0], delta_z=-1, buffer=1, minmax='min')
    rooftops_unity = grid_to_rooftops(grid_unity, roof=zrange[1], floor=zrange[0], delta_z=1, buffer=1, minmax='max')
    
rooftops = rooftops_unity

if write_map:
    pickle.dump(grid_airsim, open(gridpath_airsim, 'wb'))
    pickle.dump(grid_unity, open(gridpath_unity, 'wb'))
    pickle.dump(rooftops_airsim, open(roofpath_airsim, 'wb'))
    pickle.dump(rooftops_unity, open(roofpath_unity, 'wb'))
    
if view_maps:
    if airsim_map in ['AirSimNH']:
        interval = 60
    if airsim_map in ['Blocks']:
        interval = 20

    plt.imshow(rooftops.T, cmap='hot', interpolation='nearest', origin='lower')
    cbar = plt.colorbar(pad=0.1)
    cbar.ax.get_yaxis().labelpad = 25
    cbar.ax.set_ylabel('z [meters]', rotation=270)

    plt.title(f'{airsim_map} - Rooftops')
    plt.xticks([i for i in range(0, len(rooftops), interval)], 
               [xrange[0] + i for i in range(0, len(rooftops), interval)], rotation=90)
    plt.xlabel('x [meters]')
    plt.yticks([i for i in range(0, len(rooftops[0]), interval)], 
               [yrange[0] + i for i in range(0, len(rooftops[0]), interval)])
    plt.ylabel('y [meters]')
    plt.grid()
    plt.show()


# In[ ]:

if phase in ['collect']:
    paths_path = f'{data_path}{working_sub}/collect/{dataset}_{instance_name}_{run_num}.p'
    try:
        paths = pickle.load(open(paths_path, 'rb'))
    except:
        paths = []

    # magnitudes of movement forward to make
    magnitudes = [2**i for i in range(1, 6)]
    # magnitudes of movement up or down
    magnitudes2 = [4]

    # accesible region of map
    z_area1 = [4, 16]
    z_area2 = [1, 4]
    if airsim_map in ['AirSimNH']:
        if region in ['train']:
            if region_version in [1]: # full range
                x_area1 = [0, 480]
                y_area1 = [230, 480]
                x_area2 = [0, 240]
                y_area2 = [115, 240]
            if region_version in [2]: # inside residentials
                x_area1 = [80, 400]
                y_area1 = [230, 400]
                x_area2 = [0, 200]
                y_area2 = [115, 200]
        if region in ['test']:
            if region_version in [1]: # full range
                x_area1 = [0, 480]
                y_area1 = [0, 250]
                x_area2 = [0, 240]
                y_area2 = [0, 125]
            if region_version in [2]: # inside residentials
                x_area1 = [80, 400]
                y_area1 = [80, 250]
                x_area2 = [40, 200]
                y_area2 = [40, 125]
    if airsim_map in ['Blocks']:
        if region in ['train']:
            x_area1 = [0, 280]
            y_area1 = [60, 220]
            x_area2 = [0, 140]
            y_area2 = [30, 110]
        if region in ['test']:
            x_area1 = [0, 280]
            y_area1 = [0, 120]
            x_area2 = [0, 140]
            y_area2 = [0, 60]
            
    #x_area = x_area1
    #y_area = y_area1
    #z_area = z_area1
    #x_multiplier = 1
    #y_multiplier = 1
    #z_multiplier = 1
    z_level = 4
    x_area = x_area2
    y_area = y_area2
    z_area = z_area2
    x_multiplier = 2
    y_multiplier = 2
    z_multiplier = 4
    roof_collision = 2
    
    # make astar object
    astar = Astar(rooftops, magnitudes, magnitudes2, x_area, y_area, z_area)

    # generate start and target position
    old_progress = f'{local_path}progress/{job_name} {int(100*len(paths)/n_paths)}%'
    pk_write(' ', old_progress)
    while True:
        finished = False
        n_collected = len(paths)
        if n_collected >= n_paths:
            new_progress = f'{local_path}progress/{job_name} complete'
            finished = True
        if finished or n_collected % checkpoint_freq == 0:
            pk_write(paths, paths_path)
            if not finished:
                new_progress = f'{local_path}progress/{job_name} found {int(100*n_collected/n_paths)}%'
            if os.path.exists(old_progress):
                os.remove(old_progress)
            pk_write(' ', new_progress)
            old_progress = new_progress
            if finished:
                break
                
        while True:
            x1 = random.randint(*x_area)*x_multiplier
            y1 = random.randint(*y_area)*y_multiplier
            x2 = random.randint(*x_area)*x_multiplier
            y2 = random.randint(*y_area)*y_multiplier
            if x1 == x2 and y1 == y2:
                continue

            if motion in ['2d']:
                roof1 = rooftops[x1, y1]
                if z_level - roof_collision <= roof1:
                    continue
                roof2 = rooftops[x2, y2]
                if z_level - roof_collision <= roof2:
                    continue
                z1 = z2 = z_level
            if motion in ['3d']:
                z1 = int((rooftops[x1, y1]+roof_collision+z_multiplier)/z_multiplier)*z_multiplier
                z2 = int((rooftops[x2, y2]+roof_collision+z_multiplier)/z_multiplier)*z_multiplier
            start = np.array([x1, y1, z1])
            target = np.array([x2, y2, z2])
            euclidean = np.linalg.norm(target-start)
            if euclidean < min_distance or euclidean > max_distance:
                continue
            break
        
        # plot objective
        if view_maps:
            print(start, target)
            print('finding path...')
            #clear_output()
            fig, ax1 = plt.subplots()
            ax1.set_title(f'Drone Path (to scale)')#' Epoch #{set_num}')
            ax1.set_xlabel('x [meters]')
            ax1.set_ylabel('y [meters]')
            plt.imshow(rooftops.T, cmap='hot', interpolation='nearest', origin='lower')
            plt.colorbar(pad=0.1)
            ax1.scatter(start[0], start[1], marker='x', color='cyan', s=32)
            ax1.scatter(target[0], target[1], marker='*', color='cyan', s=32)

        # run astar to find shortest path
        path, outer_iterations, result = astar.search(start, target, motion, max_iterations = np.random.randint(40_000, 320_000))

        # plot path
        if view_maps:
            #clear_output()
            #fig, ax1 = plt.subplots()
            #ax1.set_title(f'Drone Path (to scale)')#' Epoch #{set_num}')
            #ax1.set_xlabel('x [meters]')
            #ax1.set_ylabel('y [meters]')
            plt.imshow(rooftops.T, cmap='hot', interpolation='nearest', origin='lower')
            #plt.colorbar(pad=0.1)
            for point_idx, point in enumerate(path[:-1]):
                #if (path[point_idx]['position'] == path[point_idx+1]['position']).all():
                 #   continue
                marker = 10
                if point['direction'] == 1:
                    marker = 9
                elif point['direction'] == 2:
                    marker = 11
                elif point['direction'] == 3:
                    marker = 8
                ax1.scatter(point['position'][0], point['position'][1], color = 'cyan', s=8, marker=marker)
            plt.show()

        if result in ['success']:
            paths.append(path)
            print('passed', len(paths))
        else:
            print('failed')
