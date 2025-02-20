import os
import platform
import json
import numpy as np
import random
import binvox as bv
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import math
import pickle
import random
def pk_read(path):
    return pickle.load(open(path, 'rb'))
def pk_write(obj, path):
    pickle.dump(obj, open(path, 'wb'))
#from IPython.display import clear_output

# this class reads a voxels object from a binvox file
    # fetches a surface map of all objects (top to bottom)
    # only handles cubes because of microsoft airsim stuff
class Voxels:
    def __init__(self, binvox_path, floor, xmin, ymin, zmin, vert_pad_size=4):
        self.floor = floor
        self.xmin = xmin
        self.ymin = ymin
        self.zmin = zmin
        self.binvox_path = binvox_path
        self.voxels = self.read_binvox(binvox_path)
        self.shift = self.voxels.translate[0]
        self.n = int(abs(self.shift*2)) # voxels is a cube (only thing that works with AirSim)
        self.roof_tops = np.full((self.n, self.n), -1, dtype=int)
        for x in range(self.n):
            for y in range(self.n):
                for z in range(self.n-1, -1, -1):
                    if self.voxels.data[x, y, z]:
                        self.roof_tops[x, y] = min(self.n-1, z+vert_pad_size)
                        break
                        
    # assumes a resolution of 1 cubic meter           
    def get_grid(self, resolution = 1):
        grid = {}
        x = self.xmin
        for i in range(self.n):
            grid[x] = {}
            y = self.ymin
            for j in range(self.n):
                grid[x][y] = {}
                z = self.zmin
                for k in range(self.n):
                    grid[x][y][z] = self.voxels.data[i, j, k]
                    z += resolution
                y += resolution
            x += resolution
        return grid
        
    def read_binvox(self, path):
        self._global_plt_patches = None
        return bv.Binvox.read(path, 'dense')

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
    def __init__(self, rooftops, magnitudes, magnitudes2, x_area, y_area, z_area, roof_collision=2):
        self.rooftops = rooftops.copy()
        self.magnitudes = magnitudes.copy()
        self.magnitudes2 = magnitudes2.copy()
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
                return self.return_path(current_node), outer_iterations, 'max iters'

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

# assumes a resolution of 1 cubic meter
def combine_voxels(voxelss, x_range, y_range, z_range, resolution):
    nx = int((x_range[1] - x_range[0]) / resolution) + 1
    ny = int((y_range[1] - y_range[0]) / resolution) + 1
    nz = int((z_range[1] - z_range[0]) / resolution) + 1
    
    grid = {}
    x = x_range[0]
    for _ in range(nx):
        grid[x] = {}
        y = y_range[0]
        for _ in range(ny):
            grid[x][y] = {}
            z = z_range[0]
            for _ in range(nz):
                grid[x][y][z] = False
                z += resolution
            y += resolution
        x += resolution
        
    for voxels in voxelss:
        subgrid = voxels.get_grid()
        for x in subgrid:
            if x not in grid:
                continue
            for y in subgrid[x]:
                if y not in grid[x]:
                    continue
                for z in subgrid[x][y]:
                    if z not in grid[x][y]:
                        continue
                    grid[x][y][z] = subgrid[x][y][z]
        
    return grid
               
def unity_to_airsim(grid_unity):
    grid_airsim = {}
    for x in grid_unity:
        for y in grid_unity[x]:
            for z in grid_unity[x][y]:
                if y not in grid_airsim:
                    grid_airsim[y] = {}
                if x not in grid_airsim[y]:
                    grid_airsim[y][x] = {}
                grid_airsim[y][x][-1*z] = grid_unity[x][y][z]
    return grid_airsim

def grid_to_rooftops(grid, roof, floor, delta_z, buffer=1, minmax='max'):
    rooftops = np.full((len(grid), len(grid[0])), floor, dtype=int)
    i = 0
    for y in grid:
        j = 0
        for x in grid[y]:
            z = roof
            for k in range(len(grid[y][x])):
                if grid[y][x][z]:
                    rooftops[i, j] = z + buffer # 3d padding
                    break
                z -= delta_z
            j += 1
        i += 1
        
    # add 2d padding
    pad = np.max
    if minmax in ['min']:
        pad = np.min
    for _ in range(buffer):
        rooftops_padded = rooftops.copy()
        for i in range(rooftops.shape[0]):
            for j in range(rooftops.shape[1]):
                roofs = []
                for i2 in [i+b for b in [-1, 0, 1]]:
                    if i2 > 0 and i2 < rooftops.shape[0]:
                        for j2 in [j+b for b in [-1, 0, 1]]:
                            if j2 > 0 and j2 < rooftops.shape[1]: 
                                roofs.append(rooftops[i2, j2])
                rooftops_padded[i, j] = pad(roofs)
        rooftops = rooftops_padded.copy()
    return rooftops
