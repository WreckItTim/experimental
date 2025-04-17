import utils.global_methods as gm # sys.path.append('path/to/parent/repo/')
import map_data.map_methods as mm
import numpy as np

# takes start and target in airsim coordinates
# calculates path in standard coordinates
# outputs path in airsim coordinates
def get_astar_path(start, target, motion, airsim_map, 
                   rooftops_version='v1', region='all', astar_version='v1', 
                  max_iterations=40_000,):

    # read map bounds to search for path in
    x_bounds, y_bounds, z_bounds = mm.get_bounds(airsim_map, region) # region returns bounds in airsim coords
    bounds_min = mm.drone_to_standard(x_bounds[0], y_bounds[0], z_bounds[0], airsim_map)
    bounds_max = mm.drone_to_standard(x_bounds[1], y_bounds[1], z_bounds[1], airsim_map)
    x_bounds, y_bounds, z_bounds = [bounds_min[0], bounds_max[0]], [bounds_min[1], bounds_max[1]], [bounds_min[2], bounds_max[2]]

    # read rooftops object used to detect collisions
    rooftops = gm.pk_read(f'map_data/rooftops/{rooftops_version}/rooftops_standard_{airsim_map}.p')
    
    # convert AirSim coordinates to standard coordinates
    start = mm.drone_to_standard(*start, airsim_map)
    target = mm.drone_to_standard(*target, airsim_map)
    
    # make astar object and search 
    astar = Astar(rooftops, x_bounds, y_bounds, z_bounds, astar_version)
    path, outer_iterations, result = astar.search(np.array(start), np.array(target), motion=motion, max_iterations=max_iterations)
    print('astar path result:', result, '. Astar took', outer_iterations,'number of iterations to finish. Increase max_iterations if failed and you can see that a path exists.')
    
    # convert standard coordinates back to stupid AirSim coordinates
    for point in path:
        point['position'] =  mm.standard_to_drone(*point['position'], airsim_map)
    return path, outer_iterations, result

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
        self.name = f'{self.position[0]}_{self.position[1]}_{self.position[2]}_{self.direction}'
    
class Astar:
    # rooftops is 2D grid in x,y coordinats with highest z-value cooresponding to rooftop at that point
    # roof_collision is allowed distance from above rootop before collision
    def __init__(self, rooftops, x_bounds, y_bounds, z_bounds, astar_version='v1'):

        # get magnitudes for astar actions
        if astar_version in ['v1']:
            # magnitudes of movement forward to make
            self.magnitudes_horizontal = [2, 4, 8, 16, 32]
            # magnitudes of movement up or down
            self.magnitudes_vertical = [4]
            # distance between roof for vertical collision
            self.roof_collision = 2
            # reached goal within this distance is a success
            self.goal_tolerance = 4
            # cost of each action
            self.cost = 1

        # set other params
        self.rooftops = rooftops.copy()
        self.min_x = x_bounds[0]
        self.max_x = x_bounds[1]
        self.min_y = y_bounds[0]
        self.max_y = y_bounds[1]
        self.min_z = z_bounds[0]
        self.max_z = z_bounds[1]
        self.max_horizontal = max(self.magnitudes_horizontal)
        self.max_vertical = max(self.magnitudes_vertical)
    
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
    def search(self, start, end, motion='2d', max_iterations = 10_000):
        start_node = Node(start, 0, -1, None)

        to_visit = {}
        visited = {}
        to_visit[start_node.name] = start_node

        outer_iterations = 0
    
        def check_child(position, direction, check_visited=True, check_bounds=True, check_collision=True):
            name = f'{position[0]}_{position[1]}_{position[2]}_{direction}'
            
            # check if visited already
            if check_visited and name in visited:
                return False

            # check bounds
            x, y, z = position
            if check_bounds and (x > self.max_x or x < self.min_x or 
                y > self.max_y or y < self.min_y or 
                z > self.max_z or z < self.min_z
                ):
                return False

            # check collision
            if check_collision and z - self.roof_collision <= self.rooftops[x, y]:
                return False
            
            return True
            
        while len(to_visit) > 0:
            outer_iterations += 1    

            # get best node to look at next
            best_f = np.inf
            for name in to_visit:
                node = to_visit[name]
                if node.f < best_f:
                    current_node = node
                    best_f = node.f

            # bail if too many iterations
            if outer_iterations > max_iterations:
                return self.return_path(current_node), outer_iterations, 'failure'

            # move from to_visit to visited
            del to_visit[current_node.name]
            visited[current_node.name] = current_node

            # test if goal is reached or not, if yes then return the path
            if np.linalg.norm(current_node.position - end) < self.goal_tolerance:
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
            if check_child(current_position, proposed_direction):
                children.append(Node(current_position, proposed_direction, 0, current_node))

            # child node for rotating left
            proposed_direction = current_direction - 1
            if proposed_direction == -1:
                proposed_direction = 3
            if check_child(current_position, proposed_direction):
                children.append(Node(current_position, proposed_direction, 1, current_node))

            # children nodes for moving forward (facing current direction
            proposed_position = current_position.copy()
            for magnitude in range(1, self.max_horizontal+1):
                if current_direction == 0: # move forward
                    proposed_position[1] += 1
                if current_direction == 1: # move right
                    proposed_position[0] += 1
                if current_direction == 2: # move backward
                    proposed_position[1] -= 1
                if current_direction == 3:# move left
                    proposed_position[0] -= 1
                
                if not check_child(proposed_position, current_direction, check_visited=False):
                    break

                # do we add this child?
                if magnitude in self.magnitudes_horizontal:
                    action_idx = 2+self.magnitudes_horizontal.index(magnitude)
                    name = f'{proposed_position[0]}_{proposed_position[1]}_{proposed_position[2]}_{current_direction}'
                    if name not in visited:
                        children.append(Node(proposed_position, current_direction, action_idx, current_node))

            # child nodes for moving up or down
            if motion in ['3d']:
                for updown in [-1, 1]:
                    proposed_position = current_position.copy()
                    for magnitude in range(1, self.max_vertical+1):
                        proposed_position[2] += updown # move up or down
                        
                        if not check_child(proposed_position, current_direction, check_visited=False):
                            break
        
                        # do we add this child?
                        if magnitude in self.magnitudes_vertical:
                            if updown == -1:
                                action_idx = 2+len(self.magnitudes_horizontal)+self.magnitudes_vertical.index(magnitude)
                            if updown == 1:
                                action_idx = 2+len(self.magnitudes_horizontal)+len(self.magnitudes_vertical)+self.magnitudes_vertical.index(magnitude)
                            name = f'{proposed_position[0]}_{proposed_position[1]}_{proposed_position[2]}_{current_direction}'
                            if name not in visited:
                                children.append(Node(proposed_position, current_direction, action_idx, current_node))
                
            for child in children:

                # Create the f, g, and h values
                child.g = current_node.g + self.cost # 1 cost per move
                # estimate finish cost as number of moves it would take if moving straight to goal
                #child.h = np.linalg.norm(child.position - end) # eucl distance
                child.h = 0

                # heuristic takes into account rotation and movements
                x_tol_dis = child.position[0] - end[0]
                x_tol_dis_abs = np.abs(x_tol_dis) - self.goal_tolerance
                if x_tol_dis_abs > 0:
                    if child.direction != 1 and x_tol_dis < 0:
                        child.h += 1
                    elif child.direction != 3 and x_tol_dis > 0:
                        child.h += 1
                    if x_tol_dis_abs >= 2:
                        child.h += int(np.log2(x_tol_dis_abs))
                y_tol_dis = child.position[1] - end[1] 
                y_tol_dis_abs = np.abs(y_tol_dis) - self.goal_tolerance
                if y_tol_dis_abs > 0:
                    if child.direction != 0 and y_tol_dis < 0:
                        child.h += 1
                    elif child.direction != 2 and y_tol_dis > 0:
                        child.h += 1
                    if y_tol_dis_abs >= 2:
                        child.h += int(np.log2(y_tol_dis_abs))
                z_tol_dis = child.position[2] - end[2] 
                z_tol_dis_abs = np.abs(z_tol_dis) - self.goal_tolerance
                if z_tol_dis_abs > 0:
                    child.h += int(z_tol_dis_abs/4)
                    
                child.f = child.g + child.h

                # check against other child
                if child.name in to_visit and child.g > to_visit[child.name].g:
                    continue

                # add child to to_visit 
                to_visit[child.name] = child
            #print(to_visit.keys())
                
        return self.return_path(current_node), outer_iterations, 'no children'
