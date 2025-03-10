import numpy as np
import sys
home_dir = '/home/tim/Dropbox/' # home directory of repo
rooftops_dir = f'{home_dir}data/rooftops/v1/'
sys.path.append(home_dir)
from global_methods import *
from contexts.airsim import * # map_params dictionary


def airsim_to_astar(x, y, z, airsim_map):
    x_shift = map_params[airsim_map]['x_shift']
    y_shift = map_params[airsim_map]['y_shift']
    return y+y_shift, x+x_shift, -1*z
def astar_to_airsim(x, y, z, airsim_map):
    x_shift = map_params[airsim_map]['x_shift']
    y_shift = map_params[airsim_map]['y_shift']
    return y-x_shift, x-y_shift, -1*z

def get_astar_path(motion, airsim_map, start, target, max_iterations=40_000):
    x_bounds, y_bounds, z_bounds = get_bounds(airsim_map, 'astar')
    
    # convert AirSim coordinates to Astar/Unity coordinates (positive integers starting at 0,0 in top left bottom corner)
    start = airsim_to_astar(*start, airsim_map)
    target = airsim_to_astar(*target, airsim_map)
    
    # make astar object to call 
    roofpath_astar = pk_read(f'{rooftops_dir}rooftops_unity_{airsim_map}.p')
    astar = Astar(roofpath_astar, x_bounds, y_bounds, z_bounds)
    path, outer_iterations, result = astar.search(np.array(start), np.array(target), motion=motion, max_iterations=max_iterations)
    print('astar path result:', result, '. Astar took', outer_iterations,'number of iterations to finish. Increase max_iterations if failed and you can see that a path exists.')
    
    # convert Astar coordinates back to stupid AirSim coordinates
    start = astar_to_airsim(*start, airsim_map)
    target = astar_to_airsim(*target, airsim_map)
    for point in path:
        point['position'] = astar_to_airsim(*point['position'], airsim_map)
    return path

def load_paths(paths_path):
    paths_info = pk_read(paths_path)
    n_total_paths = len(paths_info['paths'])
    return paths_info, n_total_paths


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
    # rooftops is 2D grid in x,y coordinats with highest z-value cooresponding to rooftop at that point
    # roof_collision is allowed distance from above rootop before collision
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
