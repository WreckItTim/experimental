import json
from time import localtime, time
import math
import os
import shutil
import torch
import numpy as np
import random
import binvox as bv
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import math
import pickle

# set plot params
plt.rcParams.update({'font.size': 22})
plt.show()

# uci primary colors
uci_blue = (0/255, 62/255, 120/255)
uci_gold = (253/255, 185/255, 19/255)

# uci secondary color palette
uci_light_blue = (106/255, 162/255, 184/255)
uci_light_gray = (197/255, 190/255, 181/255)
uci_dark_blue = (27/255, 61/255, 109/255)
uci_orange = (247/255, 141/255, 45/255)
uci_light_yellow = (247/255, 235/255, 95/255)
uci_dark_gray = (85/255, 87/255, 89/255)
uci_lime_green = (122/255, 184/255, 0/255)

# color blind friendly colors
# https://gist.github.com/thriveth/8560036
color_blinds = {
    'blue':   [55/255,  126/255, 184/255],  #377eb8 
    'orange': [255/255, 127/255, 0/255],    #ff7f00
    'green':  [77/255,  175/255, 74/255],   #4daf4a
    'pink':   [247/255, 129/255, 191/255],  #f781bf
    'brown':  [166/255, 86/255,  40/255],   #a65628
    'purple': [152/255, 78/255,  163/255],  #984ea3
    'gray':   [153/255, 153/255, 153/255],  #999999
    'red':    [228/255, 26/255,  28/255],   #e41a1c
    'yellow': [222/255, 222/255, 0/255]     #dede00
} 
color_blinds_list = [color_blinds[color] for color in color_blinds]

object_color = uci_blue


def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

def read_json(path):
    return json.load(open(path, 'r'))

def write_json(dictionary, path):
    json.dump(dictionary, open(path, 'w'), indent=2)

def get_timestamp():
    secondsSinceEpoch = time()
    time_obj = localtime(secondsSinceEpoch)
    timestamp = '%d_%d_%d_%d_%d_%d' % (
        time_obj.tm_year, time_obj.tm_mon, time_obj.tm_mday,  
        time_obj.tm_hour, time_obj.tm_min, time_obj.tm_sec
    )
    return timestamp

def get_operating_system():
    import platform
    return platform.system()

# STOPWATCH
# simple stopwatch to time whatevs, in (float) seconds
# keeps track of laps along with final time
class StopWatch:
    def __init__(self):
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

# this class reads a voxels object from a binvox file
    # fetches a surface map of all objects (top to bottom)
class Voxels:
    def __init__(self, binvox_path):
        self.binvox_path = binvox_path
        self.voxels = self.read_binvox(binvox_path)
        self.shift = self.voxels.translate[0]
        self.n = int(abs(self.shift*2)) # voxels is a cube (only thing that works with AirSim)
        
    def read_binvox(self, path):
        self._global_plt_patches = None
        return bv.Binvox.read(path, 'dense')

    def plot_map(self, axis, object_color=object_color, horz_pad_size=0, vert_pad_size=4):
        if self._global_plt_patches is None:
            # to fill with 1s where an object is
            self.map_2d = np.zeros((self.n, self.n), dtype=int)
            self.map_3d = np.zeros((self.n, self.n, self.n), dtype=int)
            # get floor z-index
            self.floor = max([i for i, x in enumerate(self.voxels.data[int(self.n/2), int(self.n/2), :]) if x])
            # to fill with highest z-index at x,y where object is
            self.roof_tops = np.full((self.n, self.n), -1, dtype=int)
            self._global_plt_patches = []
            for x in range(self.n):
                for y in range(self.n):
                    for z in range(self.n-1, -1, -1):
                        if self.voxels.data[x, y, z]:
                            self.roof_tops[x, y] = min(self.n-1, z+vert_pad_size)
                            # fill whole column starting at roof (with vertical padding)
                            for k in range(z+vert_pad_size, -1, -1):
                                if k >= self.n:
                                    continue
                                self.map_3d[x,y,k] = 1
                            # get x,y,z coords from indecies
                            _x = x + self.shift
                            _y = y + self.shift
                            # add to grid if roof is not floor
                            if z > self.floor:
                                patch = patches.Rectangle((_x, _y), 1, 1, color = object_color)
                                self._global_plt_patches.append(patch)
                                self.map_2d[x, y] = 1
                            break
            # add horizontal padding to objects
            if horz_pad_size > 0:
                map_2d_padded = self.map_2d.copy()
                map_3d_padded = self.map_3d.copy()
                for x in range(self.n):
                    for y in range(self.n):
                        if self.map_2d[x, y] == 1:
                            for i in range(-1*pad_size, 1+pad_size):
                                for j in range(-1*pad_size, 1+pad_size):
                                    xp = x + i
                                    yp = y + j
                                    if xp >= self.n or xp < 0:
                                        continue
                                    if yp >= self.n or yp < 0:
                                        continue
                                    # add padding
                                    if map_2d_padded[xp, yp] == 0:
                                        map_2d_padded[xp, yp] = 1
                                        # get x,y,z coords from indecies
                                        _x = xp + self.shift
                                        _y = yp + self.shift
                                        patch = patches.Rectangle((_x, _y), 1, 1, color = 'red')
                                        self._global_plt_patches.append(patch)
                                        for zp in range(self.roof_tops[xp, yp], -1, -1):
                                            map_3d_padded[xp, yp, zp] = 1
                self.map_2d = map_2d_padded.copy()
                self.map_3d = map_3d_padded.copy()
            self.floor += self.shift
        # add list of patches (much quicker than iteratively drawing)
        map_stuff = PatchCollection(self._global_plt_patches, match_original=True)
        # axis.gca().add_collection(map_stuff)
        axis.add_collection(map_stuff)
        
class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0
        self.h = 0
        self.f = 0
    def __eq__(self, other):
        return self.position == other.position
    
class Astar:
    # if is_2d==False then will be 3d
    def __init__(self, maze, moves, get_roof):
        self.maze = maze.copy()
        self.moves = moves.copy()
        self.get_roof = get_roof # used to calculate top z-coord to define collision in maze
    
    # get path found
    def return_path(self, current_node):
        path = []
        current = current_node
        while current is not None:
            path.append(current.position)
            current = current.parent
        path = path[::-1]
        return path
    
    
    # search for optimal path
    def search(self, start, end, cost=1):

        # Create start and end node with initized values for g, h and f
        start_node = Node(None, tuple(start))
        start_node.g = start_node.h = start_node.f = 0
        end_node = Node(None, tuple(end))
        end_node.g = end_node.h = end_node.f = 0

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
        max_iterations = (len(self.maze) // 2) ** 10

        #find maze has got how many rows and columns 
        no_rows, no_columns, no_zs = np.shape(self.maze)
            
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
                return self.return_path(current_node), outer_iterations, False

            # Pop current node out off yet_to_visit list, add to visited list
            yet_to_visit_list.pop(current_index)
            visited_list.append(current_node)

            # test if goal is reached or not, if yes then return the path
            if current_node == end_node:
                return self.return_path(current_node), outer_iterations, True

            # Generate children from all adjacent squares
            children = []

            for move in self.moves: 

                # Get node position
                x = int(current_node.position[0] + move[3]*move[0])
                y = int(current_node.position[1] + move[3]*move[1])
                z = int(current_node.position[2] + move[3]*move[2])
                node_position = (x, y, z)

                # Make sure within range (check if within maze boundary)
                if (node_position[0] > (no_rows - 1) or 
                    node_position[0] < 0 or 
                    node_position[1] > (no_columns -1) or 
                    node_position[1] < 0 or 
                    node_position[2] > (no_zs -1) or 
                    node_position[2] < 0):
                    continue

                # Make sure walkable terrain
                collision = False
                for s in range(1, move[3]+1):
                    x = int(current_node.position[0] + s*move[0])
                    y = int(current_node.position[1] + s*move[1])
                    z = int(current_node.position[2] + s*move[2])
                    roof = self.get_roof(x, y)
                    if z < roof:
                        collision = True
                        break
                if collision:
                    continue
                
                # Create new node
                new_node = Node(current_node, node_position)

                # Append
                children.append(new_node)

            # Loop through children
            for child in children:

                # Child is on the visited list (search entire visited list)
                if len([visited_child for visited_child in visited_list if visited_child == child]) > 0:
                    continue

                # Create the f, g, and h values
                child.g = current_node.g + cost
                ## Heuristic costs calculated here, this is using eucledian distance
                child.h = (((child.position[0] - end_node.position[0]) ** 2) + 
                           ((child.position[1] - end_node.position[1]) ** 2) + 
                           ((child.position[2] - end_node.position[2]) ** 2)
                          ) 

                child.f = child.g + child.h

                # Child is already in the yet_to_visit list and g cost is already lower
                if len([i for i in yet_to_visit_list if child == i and child.g > i.g]) > 0:
                    continue

                # Add the child to the yet_to_visit list
                yet_to_visit_list.append(child)