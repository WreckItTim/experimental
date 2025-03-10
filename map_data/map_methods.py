import os
import numpy as np
import math
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from IPython.display import clear_output
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from time import localtime, time   
from utils.global_methods import * # sys.path.append('path/to/parent/repo/')

# x_shift, y_shift are pixel values used to shift from map coordinates to numpy array coordinates
map_params = {
    'Blocks':{ # AirSim map with static objects that are arbitrary 3D shapes
        'x_shift':120,
        'y_shift':140,
    },
    'AirSimNH':{ # AirSim map with realistic, static objects that resemble a neighborhood
        'x_shift':240,
        'y_shift':240,
    },
}

# **** SET GLOBAL VARIABLES ****
class Globals:
    var_dict = {}
    def __init__(self):
        pass
    def get(key):
        if key in Globals.var_dict:
            return Globals.var_dict[key]
        return None
    def set(key, value):
        Globals.var_dict[key] = value
Globals.set('data_dicts', {map_name:{} for map_name in map_params})
Globals.set('loaded_parts', {map_name:{} for map_name in map_params})
Globals.set('rooftops', {})

# converts observation read to make pyplot plottable
def convert_observation(observation, sensor_name):
    if observation is None:
        return observation
    observation = observation.copy()
    # list of dictionary bounding boxes
    if 'Boxes' in sensor_name:
        observation = observation
    # future steps
    elif 'b2f' in sensor_name:
        observation = observation.astype(float)
        observation[observation == 0] = np.nan
    # dictionary of boolean arrays (print largest mask to frame)
    elif 'Masks' in sensor_name:
        if len(observation) == 0:
            observation = None#[[0]]
        else:
            largest_val, largest_name = -1, None
            for obj_name in observation:
                mask = observation[obj_name]
                val = np.sum(mask)
                if val > largest_val:
                    largest_val = val
                    largest_name = obj_name
            observation = observation[largest_name]
    # color
    elif 'Scene' in sensor_name or 'Segmentation' in sensor_name:
        # bgr to rgb
        temp = observation[0, :, :].copy()
        observation[0, :, :] = observation[2, :, :].copy()
        observation[2, :, :] = temp
        # move channel first to channel last
        observation = np.moveaxis(observation, 0, 2)
    # grey scale
    else:
        observation = observation[0]
    return observation

def load_rooftops(map_name, version='v1'):
    rooftops_path = f'map_data/rooftops/{version}/{map_name}.p'
    rooftops_dict = pickle.load(open(rooftops_path, 'rb'))
    rooftops_arr = []
    for key in rooftops_dict:
        row = []
        for key2 in rooftops_dict[key]:
            row.append(rooftops_dict[key][key2])
        rooftops_arr.append(row)
    rooftops_arr = np.array(rooftops_arr)
    return rooftops_dict, rooftops_arr

def get_rooftops(map_name, version='v1'):
    rooftops = Globals.get('rooftops')
    if version not in rooftops:
        rooftops[version] = {}
    if map_name not in rooftops[version]:
        rooftops_dict, rooftops_arr = load_rooftops(map_name, version)
        rooftops[version][map_name] = {}
        rooftops[version][map_name]['dict'] = rooftops_dict
        rooftops[version][map_name]['arr'] = rooftops_arr
    return rooftops[version][map_name]['dict'], rooftops[version][map_name]['arr']
    
def plot_map(fig, ax, map_name, show_z=False, 
             resolution=None, x_anchor=None, y_anchor=None, ):
    rooftops_dict, rooftops_arr = get_rooftops(map_name)
    interval = 40
    x_min = list(rooftops_dict.keys())[0]
    y_min = list(rooftops_dict[x_min].keys())[0]
    im = ax.imshow(rooftops_arr, cmap='hot', interpolation='nearest', origin='lower')
    if show_z:
        cbar = fig.colorbar(im, shrink=0.8)
        cbar.ax.get_yaxis().labelpad = 25
        cbar.ax.set_ylabel('z [meters]', rotation=270)
    ax.set_xticks([i for i in range(0, len(rooftops_arr[0]), interval)], 
              [y_min + i for i in range(0, len(rooftops_arr[0]), interval)], rotation=90)
    ax.set_yticks([i for i in range(0, len(rooftops_arr), interval)], 
              [x_min + i for i in range(0, len(rooftops_arr), interval)])
    ax.set_aspect('auto')
    if resolution is None:
        ax.set_xlim(0, len(rooftops_arr[0]))
        ax.set_ylim(0, len(rooftops_arr))
    else:
        ax.set_ylim(x_anchor-resolution[0], x_anchor+resolution[0])
        ax.set_xlim(y_anchor-resolution[1], y_anchor+resolution[1])
    
def view_map(fig, ax, x, y, z, yaw, map_name, 
             start=None, target=None, path=None, show_z=False, resolution=None ):
    x_shift = map_params[map_name]['x_shift']
    y_shift = map_params[map_name]['y_shift']
    # yaw markers for each direction
    yaw_markers = {
        0:10,
        1:9,
        2:11,
        3:8,
    }
    plot_map(fig, ax, map_name, show_z, resolution, x+x_shift, y+y_shift,)
    ax.scatter(y+y_shift, x+x_shift, color='cyan', marker=yaw_markers[yaw])
    horizion = 255
    if yaw == 0: a, b, c, d = -1, 1, 1, 1
    if yaw == 1: a, b, c, d = 1, 1, 1, -1
    if yaw == 2: a, b, c, d = -1, -1, 1, -1
    if yaw == 3: a, b, c, d = -1, 1, -1, -1
    fov1_x_points = [y+y_shift+a*i for i in range(horizion)]
    fov1_y_points = [x+x_shift+b*i for i in range(horizion)]
    fov2_x_points = [y+y_shift+c*i for i in range(horizion)]
    fov2_y_points = [x+x_shift+d*i for i in range(horizion)]
    ax.plot(fov1_x_points, fov1_y_points, color='cyan', linestyle='--')
    ax.plot(fov2_x_points, fov2_y_points, color='cyan', linestyle='--')
    if start is not None:
        ax.scatter(start[1]+y_shift, start[0]+x_shift, color='blue', marker='x', s=16)
    if target is not None:
        ax.scatter(target[1]+y_shift, target[0]+x_shift, color='green', marker='*', s=32)
    if path is not None:
        for point in path:
            ax.scatter(point[1]+y_shift, point[0]+x_shift, color='blue', marker='x', s=16)
    ax.set_aspect('auto')


def animate(map_name, sensor_names, ncols, nrows, frames, states, 
            start=None, target=None, show_path=False, text_blocks=None, resolution=None, sensor_psuedonames={}):
    # Create a figure and axes
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows)#, figsize=(4*ncols,4*nrows))
    fig.tight_layout()#h_pad=0, w_pad=0)
    
    # Animation function
    path = []
    def update(t):
        # next frame
        observations = frames[t]
        state = states[t]
        x, y, z, yaw = state
        # view map
        ax = axs[0, 0]
        ax.clear()
        view_map(fig, ax, x, y, z, yaw, map_name, start, target, path, False, resolution)
        #ax.set_title(f'x:{int(x)} y:{int(y)} z:{int(z)} dir:{yaw}') # dir for d-pad direction -- 0,1,2,3
        ax.set_title(f'Birds-eye View') # dir for d-pad direction -- 0,1,2,3
        if show_path:
            path.append([x, y])
        n_steps = 5
        # view observations
        offset = 0 # track of invisible observation panes
        for sensor_idx, observation in enumerate(observations):
            sensor_name = sensor_names[sensor_idx]
            if observation is None:
                sensor_idx2 = sensor_idx+1-offset
                col = (sensor_idx2) % ncols
                row = int( (sensor_idx2) / ncols)
                ax = axs[row, col]
                ax.clear()
                ax.set_aspect('auto')
                ax.set_visible(False)
            elif 'Boxes' in sensor_name:
                offset += 1
                sensor_idx2 = sensor_idx+1-offset
                col = (sensor_idx2) % ncols
                row = int( (sensor_idx2) / ncols)
                ax = axs[row, col]
                ax.set_visible(True)
                for obj_name in observation:
                    x_min, y_min, width, height = observation[obj_name]
                    rect = patches.Rectangle((x_min, y_min), width, height, 
                                             linewidth=1, edgecolor='white', facecolor='none')
                    ax.add_patch(rect)
                    ax.text(x_min, y_min, obj_name, color='white', fontsize=8)
            elif 'b2f' in sensor_name:
                if observation is None:
                    offset -= n_steps
                    continue
                n_steps = observation.shape[0]
                for step in range(n_steps):
                    sensor_idx2 = sensor_idx+1-offset
                    col = (sensor_idx2) % ncols
                    row = int( (sensor_idx2) / ncols)
                    ax = axs[row, col]
                    ax.clear()
                    ax.set_visible(True)
                    if observation is not None:
                        ax.imshow(observation[step], vmin=0, vmax=255)
                    else:
                        ax.set_visible(False)
                    ax.set_aspect('auto')
                    sensor_psuedoname = sensor_name
                    if sensor_name in sensor_psuedonames:
                        sensor_psuedoname = sensor_psuedonames[sensor_name]
                    ax.set_title(f'{sensor_psuedoname} step {step+1}')
                    offset -= 1
            else:
                sensor_idx2 = sensor_idx+1-offset
                col = (sensor_idx2) % ncols
                row = int( (sensor_idx2) / ncols)
                ax = axs[row, col]
                ax.clear()
                ax.set_visible(True)
                if observation.ndim == 3:
                    ax.imshow(observation)
                else:
                    if observation.dtype == bool:
                        ax.imshow(observation, cmap='grey', vmin=0, vmax=1)
                    else:
                        ax.imshow(observation, cmap='grey', vmin=0, vmax=255)
                ax.set_aspect('auto')
                sensor_psuedoname = sensor_name
                if sensor_name in sensor_psuedonames:
                    sensor_psuedoname = sensor_psuedonames[sensor_name]
                ax.set_title(sensor_psuedoname)
                
        # add text blocks to bottom grid
        if text_blocks is not None:
            sensor_idx2 = 1+sensor_idx+1-offset
            col = (sensor_idx2) % ncols
            row = int( (sensor_idx2) / ncols)
            ax = axs[row, col]
            text_block = text_blocks[t]
            ax.clear()
            ax.imshow(np.full([36, 64], 255).astype(np.uint8), cmap='grey', vmin=0, vmax=255)
            line_height = 5
            for line_idx, text_line in enumerate(text_block):
                ax.text(0, line_height+line_height*line_idx, text_line)
            ax.axis("off")
            ax.set_aspect('auto')
            ax.set_title('State')
            
        # turn off dead grids
        for col2 in range(col+1, ncols):
            ax = axs[row, col2]
            ax.clear()
            ax.set_visible(False)

        # turn off ticks
        for r in range(nrows):
            for c in range(ncols):
                ax = axs[r, c]
                ax.set_xticks([])
                ax.set_yticks([])
                
        return tuple(axs.flatten())
    
    # Create the animation
    T = len(frames)
    plt.close() # otherwise will show last frame as static plot
    ani = FuncAnimation(fig, update, frames=range(T), interval=200)
    return ani




#### FETCH/VIEW DATA

def clear_cache(map_name=None, sensor_name=None):
    if map_name is None and sensor_name is None:
        Globals.set('data_dicts', {map_name:{} for map_name in map_params})
        Globals.set('loaded_parts', {map_name:{} for map_name in map_params})
    else:
        data_dicts = Globals.get('data_dicts')
        if sensor_name in data_dicts[map_name]:
            del data_dicts[map_name][sensor_name]
            loaded_parts = Globals.get('loaded_parts')
            del loaded_parts[map_name][sensor_name]
        

def _data_start(map_name, sensor_names, return_data, ncols, additional_subplots=0, ):
    # main loop through all points
    frames = [] # fills this list to later animate
    states = [] # fills this list to later animate
    data_sets = None
    if return_data is not None:
        obs_dict = {sensor_name:[] for sensor_name in sensor_names}
        data_sets = {'observations':obs_dict, 'coordinates':[]}
    visible_sensors = [sensor_name for sensor_name in sensor_names if 'Boxes' not in sensor_name]
    nrows = math.ceil((1 + len(visible_sensors)+additional_subplots)/ncols)
    return frames, states, data_sets, nrows

def get_data_point(_x, _y, _z, _yaw, map_name, sensor_name, id_names, ):
    data_dicts = Globals.get('data_dicts')
    loaded_parts = Globals.get('loaded_parts')
    if sensor_name not in data_dicts[map_name]:
        data_dicts[map_name][sensor_name] = {}
        loaded_parts[map_name][sensor_name] = []
    data_dict = data_dicts[map_name][sensor_name]
    try:
        observation = data_dict[_x][_y][_z][_yaw]     
    except:
        sensor_dir = f'map_data/observations/{sensor_name}/{map_name}/'
        file_names = os.listdir(sensor_dir)
        for file_name in file_names:
            if 'data_dict__' not in file_name:
                continue
            part_name = file_name.split('__')[1].split('.')[0]
            if part_name in loaded_parts[map_name][sensor_name]:
                continue
            parts = part_name.split('_')
            this_id_name = parts[0]
            if this_id_name not in id_names:
                continue
            xmin, xmax, xint, ymin, ymax, yint, zmin, zmax, zint = [int(part) for part in parts[1:]] 
            if not (_x >= xmin and _x < xmax and _y >= ymin and _y < ymax and _z >= zmin and _z < zmax):
                continue
            loaded_parts[map_name][sensor_name].append(part_name)
            try:
                data_dict_part = pk_read(f'{sensor_dir}{file_name}')
                for x in data_dict_part:
                    for y in data_dict_part[x]:
                        for z in data_dict_part[x][y]:
                            for yaw in data_dict_part[x][y][z]:
                                if x not in data_dict:
                                    data_dict[x] = {}
                                if y not in data_dict[x]:
                                    data_dict[x][y] = {}
                                if z not in data_dict[x][y]:
                                    data_dict[x][y][z] = {}
                                observation = data_dict_part[x][y][z][yaw]
                                data_dict[x][y][z][yaw] = observation
            except:
                continue
        try:
            observation = data_dict[_x][_y][_z][_yaw]     
        except:
            observation = None     
    return observation

def _data_point(x, y, z, yaw, map_name, sensor_names, id_names,
                make_animation, frames, states, ncols, nrows, 
                return_data, data_sets, include_nulls=False):
    state = [x, y, z, yaw]
    if make_animation:
        this_frame = []
    # fetch observations
    contains_nulls = False
    observations = {}
    for sensor_idx, sensor_name in enumerate(sensor_names):
        observation = get_data_point(x, y, z, yaw, map_name, sensor_name, id_names, )
        if make_animation:
            observation_viewable = convert_observation(observation, sensor_name)
            this_frame.append(observation_viewable)
        if return_data:
            if observation is None: # null
                contains_nulls = True
            observations[sensor_name] = observation
    if make_animation:
        frames.append(this_frame)
        states.append(state)
    if return_data:
        if contains_nulls and not include_nulls:
            return
        data_sets['coordinates'].append(state)
        for sensor_name in observations:
            data_sets['observations'][sensor_name].append(observations[sensor_name])
            
def _data_end(map_name, sensor_names, make_animation, ncols, nrows, frames, states, start, target, show_path, 
              return_data, data_sets, text_blocks=None, resolution=None, sensor_psuedonames={}):
    animation = None
    if make_animation:
        animation = animate(map_name, sensor_names, ncols, nrows, frames, states, start, target, show_path, text_blocks, resolution, sensor_psuedonames)
    if return_data:
        for sensor_name in sensor_names:
            if 'Masks' not in sensor_name and 'Boxes' not in sensor_name:
                data_sets['observations'][sensor_name] = np.array(data_sets['observations'][sensor_name])
        data_sets['coordinates'] = np.array(data_sets['coordinates'])
    return animation

# coordinates is list of [x, y, z, yaw] 
def data_at_coordinates(map_name, sensor_names, id_names, coordinates, 
                        make_animation=True, return_data=True, ncols=2, additional_subplots=0, resolution=None,
                        sensor_psuedonames={}, include_nulls=False):
    frames, states, data_sets, nrows = _data_start(map_name, sensor_names, return_data, ncols, additional_subplots)
    for coordinate in coordinates:
        _data_point(*coordinate, map_name, sensor_names, id_names,
                    make_animation, frames, states, ncols, nrows,
                    return_data, data_sets, include_nulls)
    animation = _data_end(map_name, sensor_names, make_animation, ncols, nrows, frames, states, 
                          None, None, False, return_data, data_sets, None, resolution, sensor_psuedonames)
    return data_sets, animation
        

# fetches and/or visualizes data following a given path
def data_at_path(map_name, sensor_names, id_names, path, 
                 make_animation=True, return_data=True, ncols=2, additional_subplots=0, resolution=None, 
                 sensor_psuedonames={}, include_nulls=False):
    frames, states, data_sets, nrows = _data_start(map_name, sensor_names, return_data, ncols, additional_subplots)
    start = path[0]['position']
    target = path[-1]['position']
    for point in path:
        x, y, z = point['position']
        yaw = point['direction']
        _data_point(x, y, z, yaw, map_name, sensor_names, id_names,
                make_animation, frames, states, ncols, nrows, 
                return_data, data_sets, include_nulls)
    animation = _data_end(map_name, sensor_names, make_animation, ncols, nrows, frames, states, 
                          start, target, True, return_data, data_sets, None, resolution, sensor_psuedonames)
    return data_sets, animation

# fetches and/or visualizes data following a path returned from a DRL episode
def yaw_to_idx(yaw):
    while yaw < 0:
        yaw += 2*np.pi
    while yaw >= 2*np.pi:
        yaw -= 2*np.pi
    yaw_idx = round(yaw/(np.pi/2))
    return yaw_idx
def data_at_episode(map_name, sensor_names, id_names, episode, actions, 
                    make_animation=True, return_data=True, ncols = 2, additional_subplots=0, 
                    resolution=None, sensor_psuedonames={}, include_nulls=False):
    frames, states, data_sets, nrows = _data_start(map_name, sensor_names, return_data, ncols, additional_subplots+1)
    initial_state = episode[0]
    start_pos = initial_state['drone_position']
    start_yaw = initial_state['yaw']
    target_pos = initial_state['goal_position']
    astar_length = initial_state['astar_length']
    path_linearity = initial_state['path_linearity']
    path_nonlinearity = initial_state['path_nonlinearity']
    path_idx = initial_state['path_idx']
    text_blocks =  []
    for state_idx, state in enumerate(episode):
        x, y, z = state['drone_position']
        yaw = yaw_to_idx(state['yaw'])
        if state_idx == len(episode)-1:
            reached_goal = state['reached_goal']
            has_collided = state['has_collided']
            termination_reason = state['termination_reason']
            text_block = [
                f'step: {state_idx+1}',
                f'goal: {reached_goal}',
                f'collided: {has_collided}',
                f'terminated: {termination_reason}',
            ]
        else:
            next_state = episode[state_idx+1]
            action = actions[next_state['rl_output']]
            reward = next_state['total_reward']
            text_block = [
                f'step: {state_idx+1}',
                f'action: {action}',
                f'reward: {reward:.4f}',
            ]
        text_blocks.append(text_block)
        _data_point(x, y, z, yaw, map_name, sensor_names, id_names,
                make_animation, frames, states, ncols, nrows, 
                return_data, data_sets, include_nulls )
    animation = _data_end(map_name, sensor_names, make_animation, ncols, nrows, frames, states, 
                          start_pos, target_pos, True, return_data, data_sets, text_blocks, resolution, sensor_psuedonames)
    return data_sets, animation


# helper function to view data results after running a view/fetch cell below
global dummy_warning
def display_data_results(data, animation, sensor_psuedonames={}):

    # visualize animation
    if animation is not None:
        display(HTML(animation.to_jshtml()))
        dummy_warning = animation
    
    # print dimensions of returned data
    if data is not None:
        N = len(data['coordinates'])
        print('collected', N, 'many data points!')
        for sensor_name in data['observations']:
            data_set = data['observations'][sensor_name]
            sensor_psuedoname = sensor_name
            if sensor_name in sensor_psuedonames:
                sensor_psuedoname = sensor_psuedonames[sensor_name]
            if 'Boxes' in sensor_name or 'Masks' in sensor_name:
                obj_counts = {}
                for n in range(N):
                    obj_data = data_set[n]
                    for obj_name in obj_data:
                        if obj_name not in obj_counts:
                            obj_counts[obj_name] = 0
                        obj_counts[obj_name] += 1
                print(sensor_psuedoname, 'collected from', len(obj_counts), 'many objects')
            else:
                print(sensor_psuedoname, 'has shape of', data_set.shape)

# region = 'train' 'test' 'all'
# sensor_names = ['DepthV2', 'SceneV1', etc]
# resample=True to not use static set of random indicies (suggested resample=False for repeatibility)
def get_data(map_name, sensor_names, region, sample_size=None, resample=False, id_names=['alpha'], ):
    assert region in ['train', 'test', 'all', 'houses_train', 'houses_test', 'houses_all'], 'invalid region'
    # set coordinate ranges
    x_bounds, y_bounds, z_bounds = get_bounds(map_name, region)
    x_vals = [x for x in range(x_bounds[0], x_bounds[1], 2)]
    y_vals = [y for y in range(y_bounds[0], y_bounds[1], 2)]
    z_vals = [-4]
    yaw_vals = [0, 1, 2, 3]

    # get valid coordinates
    coordinates = []
    for x in x_vals:
        for y in y_vals:
            for z in z_vals:
                if in_object(x, y, z, map_name):
                    continue
                for yaw in yaw_vals:
                    coordinates.append([x, y, z, yaw])
    coordinates = np.array(coordinates)

    # read coordinate_idxs for static sampling
    coordinate_idxs_path = f'map_data/coordinate_idxs/{map_name}__{region}.p'
    if sample_size is not None:
        if resample or not os.path.exists(coordinate_idxs_path):
            idxs = [i for i in range(len(coordinates))]
            random.shuffle(idxs)
            if not os.path.exists(coordinate_idxs_path):
                pk_write(idxs, coordinate_idxs_path)
        if not resample:
            idxs = pk_read(coordinate_idxs_path)
        coordinates = coordinates[idxs[:sample_size]]
    
    # fetch data from coordinates
    location_data, location_animation = data_at_coordinates(map_name, sensor_names, id_names, coordinates, make_animation=False, return_data=True)
    
    # return resulting data
    return location_data

def yaw_to_idx(yaw):
    while yaw < 0:
        yaw += 2*np.pi
    while yaw >= 2*np.pi:
        yaw -= 2*np.pi
    yaw_idx = round(yaw/(np.pi/2))
    return yaw_idx

def get_bounds(map_name, region):
    # bounds drone can move in
    z_bounds = [-40, 0]
    if region in ['astar']:
        z_bounds = [4, 16] # set the valid bounds to explore for z-values in astar coords
    if map_name in ['Blocks']:
        if region in ['all']:
            x_bounds = [-120, 100]
            y_bounds = [-140, 140]
        elif region in ['train']:
            x_bounds = [-60, 100]
            y_bounds = [-140, 140]
        elif region in ['test']:
            x_bounds = [-120, 0]
            y_bounds = [-140, 140]
        elif region in ['astar']:
            x_bounds = [0, 280] # set the valid bounds to explore for x-values in astar coords
            y_bounds = [0, 220] # set the valid bounds to explore for y-values in astar coords
    if map_name in ['AirSimNH']:
        if region in ['all']:
            x_bounds = [-240, 240]
            y_bounds = [-240, 240]
        elif region in ['train']: 
            x_bounds = [-10, 240]
            y_bounds = [-240, 240]
        elif region in ['test']:
            x_bounds = [-240, 10]
            y_bounds = [-240, 240]
        elif region in ['astar']:
            x_bounds = [0, 480] # set the valid bounds to explore for x-values in astar coords
            y_bounds = [0, 480] # set the valid bounds to explore for y-values in astar coords
        elif region in ['houses_astar']:
            x_bounds = [80, 400] # set the valid bounds to explore for x-values in astar coords
            y_bounds = [80, 400] # set the valid bounds to explore for y-values in astar coords
        elif region in ['houses_all']:
            x_bounds = [-160, 160]
            y_bounds = [-160, 160]
        elif region in ['houses_train']:
            x_bounds = [-10, 160]
            y_bounds = [-160, 160]
        elif region in ['houses_test']:
            x_bounds = [-160, 10]
            y_bounds = [-160, 160]
    return x_bounds, y_bounds, z_bounds
            
# checks if coordinates are in object based on collision_threshold (meters) above rooftop (drone coords use -z for above floor)
def in_object(x, y, z, map_name, collision_threshold=-2):
    rooftops_dict, rooftops_arr = get_rooftops(map_name)
    roof = rooftops_dict[int(x)][int(y)]
    return bool(roof + collision_threshold < z)