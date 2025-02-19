import os
import numpy as np
import math
import pickle
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from IPython.display import clear_output
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import json
from time import localtime, time


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
# https://gist.github.com/thriveth/8560036
color_blinds = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
Globals.set('color_blinds', color_blinds)
map_params = {
    'Blocks':{
        'x_shift':120,
        'y_shift':140,
    },
    'AirSimNH':{
        'x_shift':240,
        'y_shift':240,
    },
}
Globals.set('map_params', map_params)
Globals.set('nulls', {})
Globals.set('data_dicts', {map_name:{} for map_name in map_params})
Globals.set('cache', {map_name:{} for map_name in map_params})


# **** common utility funcitons **** 
# STOPWATCH
# simple stopwatch to time whatevs, in (float) seconds
# keeps track of laps along with final time
class StopWatch:
    def __init__(self):
        self.start()
    def start(self):
        self.start_time = time()
        self.last_time = self.start_time
        self.laps = []
    def lap(self):
        this_time = time()
        delta_time = this_time - self.last_time
        self.laps.append(delta_time)
        self.last_time = this_time
        return delta_time
    def stop(self):
        self.stop_time = time()
        self.delta_time = self.stop_time - self.start_time
        return self.delta_time
def pk_read(path):
    return pickle.load(open(path, 'rb'))
def pk_write(obj, path):
    pickle.dump(obj, open(path, 'wb'))
def read_json(path):
	return json.load(open(path, 'r'))
def write_json(dictionary, path):
	json.dump(dictionary, open(path, 'w'), indent=2)
def fix_directory(directory):
	if '\\' in directory:
		directory = directory.replace('\\', '/')
	if directory[-1] != '/':
		directory += '/'
	return directory
def yaw_to_idx(yaw):
	while yaw < 0:
		yaw += 2*np.pi
	while yaw >= 2*np.pi:
		yaw -= 2*np.pi
	yaw_idx = round(yaw/(np.pi/2))
	return yaw_idx
            

# converts observation read in by AirSim to make pyplot plottable
def convert_observation(observation, sensor_name):
    nulls = Globals.get('nulls')
    if isinstance(observation, str): # null
        return nulls[sensor_name]
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
    
# **** beta methods ****
def update_experimental_data(airsim_map, sensor_name, old_id='beta', new_id='alpha'):
    parent_path = f'observations/{sensor_name}/{airsim_map}/'
    old_file_names = os.listdir(parent_path)
    for old_file_name in old_file_names:
        old_path = parent_path+old_file_name
        if old_id in old_file_name:
            new_file_name = old_file_name.replace(old_id, new_id)
            new_path = parent_path+new_file_name
            if os.path.exists(new_path):
                os.remove(new_path)
            os.rename(old_path, new_path)
def remove_file(sensor_name, airsim_map, file_name):
    os.remove(f'data/observations/{sensor_name}/{airsim_map}/{file_name}')

def load_rooftops(rt_path):
    rooftops_dict = pickle.load(open(rt_path, 'rb'))
    rooftops_arr = []
    for key in rooftops_dict:
        row = []
        for key2 in rooftops_dict[key]:
            row.append(rooftops_dict[key][key2])
        rooftops_arr.append(row)
    rooftops_arr = np.array(rooftops_arr)
    return rooftops_dict, rooftops_arr
    
def plot_map(fig, ax, data_dir, airsim_map, show_z=False, 
             resolution=None, x_anchor=None, y_anchor=None, ):
    rooftops_path = f'{data_dir}rooftops/{airsim_map}.p'
    rooftops_dict, rooftops_arr = load_rooftops(rooftops_path)
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
    #ax.set_ylabel('x [meters]')
    #ax.set_xlabel('y [meters]')
    ax.set_aspect('auto')
    if resolution is None:
        ax.set_xlim(0, len(rooftops_arr[0]))
        ax.set_ylim(0, len(rooftops_arr))
    else:
        ax.set_ylim(x_anchor-resolution[0], x_anchor+resolution[0])
        ax.set_xlim(y_anchor-resolution[1], y_anchor+resolution[1])
    
def view_map(fig, ax, x, y, z, yaw, data_dir, airsim_map, 
             start=None, target=None, path=None, show_z=False, resolution=None ):
    map_params = Globals.get('map_params')[airsim_map]
    x_shift = map_params['x_shift']
    y_shift = map_params['y_shift']
    # yaw markers for each direction
    yaw_markers = {
        0:10,
        1:9,
        2:11,
        3:8,
    }
    plot_map(fig, ax, data_dir, airsim_map, show_z, 
             resolution, x+x_shift, y+y_shift,)
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
    ax.set_title('AirSim Map and Perspective')


def animate(data_dir, airsim_map, sensor_names, ncols, nrows, frames, states, 
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
        view_map(fig, ax, x, y, z, yaw, data_dir, airsim_map, start, target, path, False, resolution)
        #ax.set_title(f'x:{int(x)} y:{int(y)} z:{int(z)} dir:{yaw}') # dir for d-pad direction -- 0,1,2,3
        ax.set_title(f'Birds-eye View') # dir for d-pad direction -- 0,1,2,3
        if show_path:
            path.append([x, y])
        n_steps = 5
        # view observations
        offset = 0 # track of invisible observation panes
        for sensor_idx, observation in enumerate(observations):
            sensor_name = sensor_names[sensor_idx]
            if 'Boxes' in sensor_name:
                offset += 1
                sensor_idx2 = sensor_idx+1-offset
                col = (sensor_idx2) % ncols
                row = int( (sensor_idx2) / ncols)
                ax = axs[row, col]
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
                if observation is not None:
                    if observation.ndim == 3:
                        ax.imshow(observation)
                    else:
                        if observation.dtype == bool:
                            ax.imshow(observation, cmap='grey', vmin=0, vmax=1)
                        else:
                            ax.imshow(observation, cmap='grey', vmin=0, vmax=255)
                else:
                    ax.set_visible(False)
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

def airsim_to_astar(x, y, z, x_shift=None, y_shift=None):
    if x_shift is None:
        x_shift = Globals.get('x_shift')
    if y_shift is None:
        y_shift = Globals.get('y_shift')
    return y+y_shift, x+x_shift, -1*z
def astar_to_airsim(x, y, z, y_shift=None, x_shift=None):
    if x_shift is None:
        x_shift = Globals.get('x_shift')
    if y_shift is None:
        y_shift = Globals.get('y_shift')
    return y-x_shift, x-y_shift, -1*z

def get_astar_path(motion, airsim_map, start, target, max_iterations=40_000):
    if airsim_map in ['Blocks']:
        x_area = [0, 280] # set the valid bounds to explore for x-values in astar coords
        y_area = [0, 220] # set the valid bounds to explore for y-values in astar coords
    if airsim_map in ['AirSimNH']:
        x_area = [0, 480] # set the valid bounds to explore for x-values in astar coords
        y_area = [0, 480] # set the valid bounds to explore for y-values in astar coords
    z_area = [4, 16] # set the valid bounds to explore for z-values in astar coords
    
    # convert AirSim coordinates to Astar/Unity coordinates (positive integers starting at 0,0 in top left bottom corner)
    start = airsim_to_astar(*start)
    target = airsim_to_astar(*target)
    
    # make astar object to call 
    roofpath_astar = pk_read(f'rooftops/rooftops_unity_{airsim_map}.p')
    astar = Astar(roofpath_astar, x_area, y_area, z_area)
    path, outer_iterations, result = astar.search(np.array(start), np.array(target), motion=motion, max_iterations=max_iterations)
    print('astar path result:', result, '. Astar took', outer_iterations,'number of iterations to finish. Increase max_iterations if failed and you can see that a path exists.')
    
    # convert Astar coordinates back to stupid AirSim coordinates
    start = astar_to_airsim(*start)
    target = astar_to_airsim(*target)
    for point in path:
        point['position'] = astar_to_airsim(*point['position'])
    return path

def load_paths(paths_path):
    paths_info = pk_read(paths_path)
    n_total_paths = len(paths_info['paths'])
    return paths_info, n_total_paths




#### FETCH/VIEW DATA

def clear_cache(airsim_map=None, sensor_name=None):
    if airsim_map is None and sensor_name is None:
        Globals.set('data_dicts', {map_name:{} for map_name in map_params})
        Globals.set('cache', {map_name:{} for map_name in map_params})
    else:
        data_dicts = Globals.get('data_dicts')
        if sensor_name in data_dicts[airsim_map]:
            del data_dicts[airsim_map][sensor_name]
            cache = Globals.get('cache')
            del cache[airsim_map][sensor_name]
        

def _data_start(airsim_map, sensor_names, return_data, ncols, additional_subplots=0, ):
    # main loop through all points
    frames = [] # fills this list to later animate
    states = [] # fills this list to later animate
    data_sets = None
    if return_data is not None:
        obs_dict = {sensor_name:[] for sensor_name in sensor_names}
        data_sets = {'observations':obs_dict, 'coordinates':[]}
    visible_sensors = [sensor_name for sensor_name in sensor_names if 'Boxes' not in sensor_name]
    nrows = math.ceil(0.5+(len(visible_sensors)+additional_subplots)/ncols)
    return frames, states, data_sets, nrows

def get_data_point(_x, _y, _z, _yaw, data_dir, airsim_map, sensor_name, id_names, ):
    data_dicts = Globals.get('data_dicts')
    cache = Globals.get('cache')
    nulls = Globals.get('nulls')
    if sensor_name not in data_dicts[airsim_map]:
        data_dicts[airsim_map][sensor_name] = {}
        cache[airsim_map][sensor_name] = []
    data_dict = data_dicts[airsim_map][sensor_name]
    try:
        observation = data_dict[_x][_y][_z][_yaw]     
    except:
        sensor_dir = f'{data_dir}observations/{sensor_name}/{airsim_map}/'
        file_names = os.listdir(sensor_dir)
        for file_name in file_names:
            if 'data_dict__' not in file_name:
                continue
            part_name = file_name.split('__')[1].split('.')[0]
            if part_name in cache[airsim_map][sensor_name]:
                continue
            parts = part_name.split('_')
            this_id_name = parts[0]
            if this_id_name not in id_names:
                continue
            xmin, xmax, xint, ymin, ymax, yint, zmin, zmax, zint = [int(part) for part in parts[2:]] 
            if not (_x >= xmin and _x < xmax and _y >= ymin and _y < ymax and _z >= zmin and _z < zmax):
                continue
            cache[airsim_map][sensor_name].append(part_name)
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
                                if sensor_name not in nulls:
                                    if 'Boxes' in sensor_name or 'Masks' in sensor_name:
                                        nulls[sensor_name] = {}
                                    else:
                                        nulls[sensor_name] = np.full(observation.shape, 0).astype(observation.dtype)
            except:
                continue
        try:
            observation = data_dict[_x][_y][_z][_yaw]     
        except:
            observation = 'null'        
    return observation

def _data_point(x, y, z, yaw, data_dir, airsim_map, sensor_names, id_names,
                make_animation, frames, states, ncols, nrows, 
                return_data, data_sets, include_nulls=False):
    state = [x, y, z, yaw]
    if make_animation:
        this_frame = []
    # fetch observations
    contains_nulls = False
    observations = {}
    for sensor_idx, sensor_name in enumerate(sensor_names):
        observation = get_data_point(x, y, z, yaw, data_dir, airsim_map, sensor_name, id_names, )
        if make_animation:
            observation_viewable = convert_observation(observation, sensor_name)
            this_frame.append(observation_viewable)
        if return_data:
            if isinstance(observation, str): # null
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
            
def _data_end(data_dir, airsim_map, sensor_names, make_animation, ncols, nrows, frames, states, start, target, show_path, 
              return_data, data_sets, text_blocks=None, resolution=None, sensor_psuedonames={}):
    animation = None
    if make_animation:
        animation = animate(data_dir, airsim_map, sensor_names, ncols, nrows, frames, states, start, target, show_path, text_blocks, resolution, sensor_psuedonames)
    if return_data:
        for sensor_name in sensor_names:
            if 'Masks' not in sensor_name and 'Boxes' not in sensor_name:
                data_sets['observations'][sensor_name] = np.array(data_sets['observations'][sensor_name])
        data_sets['coordinates'] = np.array(data_sets['coordinates'])
    return animation

def data_at_coordinates(data_dir, airsim_map, sensor_names, id_names, x_vals, y_vals, z_vals, yaw_vals, 
                        make_animation=True, return_data=True, ncols=2, additional_subplots=0, resolution=None,
                        sensor_psuedonames={}, include_nulls=False):
    data_dir = fix_directory(data_dir)
    frames, states, data_sets, nrows = _data_start(airsim_map, sensor_names, return_data, ncols, additional_subplots)
    for xi, x in enumerate(x_vals):
        for yi, y in enumerate(y_vals):
            for zi, z in enumerate(z_vals):
                for yawi, yaw in enumerate(yaw_vals):
                    _data_point(x, y, z, yaw, data_dir, airsim_map, sensor_names, id_names,
                                make_animation, frames, states, ncols, nrows,
                                return_data, data_sets, include_nulls)
    animation = _data_end(data_dir, airsim_map, sensor_names, make_animation, ncols, nrows, frames, states, 
                          None, None, False, return_data, data_sets, None, resolution, sensor_psuedonames)
    return data_sets, animation
        

# fetches and/or visualizes data following a given path
def data_at_path(data_dir, airsim_map, sensor_names, id_names, path, 
                 make_animation=True, return_data=True, ncols = 2, resolution=None, 
                 sensor_psuedonames={}, include_nulls=False):
    data_dir = fix_directory(data_dir)
    sensor_names, frames, states, data_sets, nrows = _data_start(sensor_names, return_data, ncols)
    start = path[0]['position']
    target = path[-1]['position']
    for point in path:
        x, y, z = point['position']
        yaw = point['direction']
        _data_point(x, y, z, yaw, data_dir, airsim_map, sensor_names, id_names,
                make_animation, frames, states, ncols, nrows, 
                return_data, data_sets, include_nulls)
    animation = _data_end(data_dir, airsim_map, sensor_names, make_animation, ncols, nrows, frames, states, 
                          start, target, True, return_data, data_sets, None, resolution, sensor_psuedonames)
    return data_sets, animation

# fetches and/or visualizes data following a path returned from a DRL episode
def data_at_episode(data_dir, airsim_map, sensor_names, id_names, episode, actions, 
                    make_animation=True, return_data=True, ncols = 2, additional_subplots=0, 
                    resolution=None, sensor_psuedonames={}, include_nulls=False):
    data_dir = fix_directory(data_dir)
    frames, states, data_sets, nrows = _data_start(airsim_map, sensor_names, return_data, ncols, additional_subplots+1)
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
        _data_point(x, y, z, yaw, data_dir, airsim_map, sensor_names, id_names,
                make_animation, frames, states, ncols, nrows, 
                return_data, data_sets, include_nulls )
    animation = _data_end(data_dir, airsim_map, sensor_names, make_animation, ncols, nrows, frames, states, 
                          start_pos, target_pos, True, return_data, data_sets, text_blocks, resolution, sensor_psuedonames)
    return data_sets, animation




#### ASTAR SHORTEST PATH

class Node:
    def __init__(self, position, direction, action, parent=None):
        self.parent = parent
        self.position = position.copy()
        self.direction = direction
        self.action = action
        self.g = 0
        self.h = 0
        self.f = 0
    
class Astar:
    def __init__(self, rooftops, x_area, y_area, z_area, roof_collision=2):

        # these determine step sizes that can take moving forward
        self.magnitudes = [2**i for i in range(1, 6)]
        
        # these determine step sizes that can take moving up or down
        self.magnitudes2 = [4]
        
        self.rooftops = rooftops.copy()
        self.min_x = x_area[0]
        self.max_x = x_area[1]
        self.min_y = y_area[0]
        self.max_y = y_area[1]
        self.min_z = z_area[0]
        self.max_z = z_area[1]
        self.roof_collision = roof_collision
    
    # get path found
    def return_path(self, current_node):
        path = []
        current = current_node
        while current is not None:
            path.append({
                'position':current.position,
                'direction':current.direction,
                'action':current.action,
            })
            current = current.parent
        # reverse order to return nodes ordered from start to end
        path = path[::-1]
        return path
    
    
    # search for optimal path
    def search(self, start, end, motion='2d', cost=1, goal_tolerance=4, max_iterations = 10_000):
        start_node = Node(start, 0, -1, None)

        # Initialize both yet_to_visit and visited list
        # in this list we will put all node that are yet_to_visit for exploration. 
        # From here we will find the lowest cost node to expand next
        yet_to_visit_list = []  
        # in this list we will put all node those already explored so that we don't explore it again
        visited_list = [] 

        # Add the start node
        yet_to_visit_list.append(start_node)

        # Adding a stop condition. This is to avoid any infinite loop and stop 
        # execution after some reasonable number of steps
        outer_iterations = 0
            
        # Loop until you find the end
        while len(yet_to_visit_list) > 0:

            # Every time any node is referred from yet_to_visit list, counter of limit operation incremented
            outer_iterations += 1    

            # Get the current node
            current_node = yet_to_visit_list[0]
            current_index = 0
            for index, item in enumerate(yet_to_visit_list):
                if item.f < current_node.f:
                    current_node = item
                    current_index = index

            # if we hit this point return the path such as it may be no solution or 
            # computation cost is too high
            if outer_iterations > max_iterations:
                return self.return_path(current_node), outer_iterations, 'failure'

            # Pop current node out off yet_to_visit list, add to visited list
            yet_to_visit_list.pop(current_index)
            visited_list.append(current_node)

            # test if goal is reached or not, if yes then return the path
            if np.linalg.norm(current_node.position - end) < goal_tolerance:
                return self.return_path(current_node), outer_iterations, 'success'

            # Generate children from all adjacent squares
            children = []

            # get current state
            current_position = current_node.position
            current_direction = current_node.direction

            # child node for rotating right
            proposed_direction = current_direction + 1
            if proposed_direction == 4:
                proposed_direction = 0
            children.append(Node(current_position, proposed_direction, 0, current_node))

            # child node for rotating left
            proposed_direction = current_direction - 1
            if proposed_direction == -1:
                proposed_direction = 3
            children.append(Node(current_position, proposed_direction, 1, current_node))

            # children nodes for moving forward (facing current direction
            proposed_position = current_position.copy()
            for magnitude in range(1, max(self.magnitudes)+1):
                if current_direction == 0: # move forward
                    proposed_position[1] += 1
                if current_direction == 1: # move right
                    proposed_position[0] += 1
                if current_direction == 2: # move backward
                    proposed_position[1] -= 1
                if current_direction == 3:# move left
                    proposed_position[0] -= 1

                # check bounds
                x, y, z = proposed_position
                if (x > self.max_x or x < self.min_x or 
                    y > self.max_y or y < self.min_y or 
                    z > self.max_z or z < self.min_z
                   ):
                    break

                # check collision
                if z - self.roof_collision <= self.rooftops[x, y]:
                    break

                # do we add this child?
                if magnitude in self.magnitudes:
                    action_idx = 2+self.magnitudes.index(magnitude)
                    children.append(Node(proposed_position, current_direction, action_idx, current_node))

            # child nodes for moving up or down
            if motion in ['3d']:
                for updown in [-1, 1]:
                    proposed_position = current_position.copy()
                    for magnitude in range(1, max(self.magnitudes2)+1):
                        proposed_position[2] += updown # move up or down
                        
                        # check bounds
                        x, y, z = proposed_position
                        if (x > self.max_x or x < self.min_x or 
                            y > self.max_y or y < self.min_y or 
                            z > self.max_z or z < self.min_z
                           ):
                            break
        
                        # check collision
                        if z - self.roof_collision <= self.rooftops[x, y]:
                            break
        
                        # do we add this child?
                        if magnitude in self.magnitudes2:
                            if updown == -1:
                                action_idx = 2+len(self.magnitudes)+self.magnitudes2.index(magnitude)
                            if updown == 1:
                                action_idx = 2+len(self.magnitudes)+len(self.magnitudes2)+self.magnitudes2.index(magnitude)
                            children.append(Node(proposed_position, current_direction, action_idx, current_node))
                
            # Loop through children
            for child in children:

                # Child is on the visited list (search entire visited list)
                if len([visited_child for visited_child in visited_list if visited_child == child]) > 0:
                    continue

                # Create the f, g, and h values
                child.g = current_node.g + cost # 1 cost per move
                # estimate finish cost as number of moves it would take if moving straight to goal
                #child.h = np.linalg.norm(child.position - end) # eucl distance
                child.h = 0

                # heuristic takes into account rotation and movements
                x_tol_dis = child.position[0] - end[0]
                x_tol_dis_abs = np.abs(x_tol_dis) - goal_tolerance
                if x_tol_dis_abs > 0:
                    if child.direction != 1 and x_tol_dis < 0:
                        child.h += 1
                    elif child.direction != 3 and x_tol_dis > 0:
                        child.h += 1
                    if x_tol_dis_abs >= 2:
                        child.h += int(np.log2(x_tol_dis_abs))
                y_tol_dis = child.position[1] - end[1] 
                y_tol_dis_abs = np.abs(y_tol_dis) - goal_tolerance
                if y_tol_dis_abs > 0:
                    if child.direction != 0 and y_tol_dis < 0:
                        child.h += 1
                    elif child.direction != 2 and y_tol_dis > 0:
                        child.h += 1
                    if y_tol_dis_abs >= 2:
                        child.h += int(np.log2(y_tol_dis_abs))
                z_tol_dis = child.position[2] - end[2] 
                z_tol_dis_abs = np.abs(z_tol_dis) - goal_tolerance
                if z_tol_dis_abs > 0:
                    child.h += int(z_tol_dis_abs/4)
                    
                child.f = child.g + child.h

                # Child is already in the yet_to_visit list and g cost is already lower
                if len([i for i in yet_to_visit_list if child == i and child.g > i.g]) > 0:
                    continue

                # Add the child to the yet_to_visit list
                yet_to_visit_list.append(child)
                
        return self.return_path(current_node), outer_iterations, 'no children'


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



# json files output with all string key names
# process so that the evaluation dictionary structure is such:
    # episode # - int
        # step # - int
            # state - dictionary of misc key, value pairs for that state
def process_episodes(json_evaluation):
    nEpisodes = len(json_evaluation)
    episodes = [None] * nEpisodes
    episode_idx = 0
    for episode_str in json_evaluation:
        if 'episode_' not in episode_str:
            continue
        json_episode = json_evaluation[episode_str]
        nSteps = len(json_episode)
        states = [None] * nSteps
        for step_str in json_episode:
            step_num = int(step_str.split('_')[1])
            state = json_episode[step_str]
            states[step_num] = state
        episodes[episode_idx] = states
        episode_idx += 1
    return episodes
def read_evaluations(evaluation_folder):
    evaluation_files = [file for file in os.listdir(evaluation_folder) if 'states' in file]
    nEvaluations = len(evaluation_files)
    evaluations = [None] * nEvaluations
    for evaluation_file in evaluation_files:
        if '.json' not in evaluation_file:
            continue
        epoch = int(evaluation_file.split('.')[0].split('_')[-1])
        #print(evaluation_file, epoch)
        json_evaluation = json.load(open(evaluation_folder + evaluation_file, 'r'))
        episodes = process_episodes(json_evaluation)
        evaluations[epoch] = episodes
    return evaluations
# architecture for evaluations:
# evaluations - list of episodes (indexed of evaluation number) - 0 idx is first evaluation
    # episodes - list of states (indexed by step number)
        # states - dict of (key, value) pairs for state at all_evaluations[instance][evaluation][episode][step]