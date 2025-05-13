import utils.global_methods as gm # sys.path.append('path/to/parent/repo/')
import map_data.map_methods as mm
import numpy as np

def get_astar_path(start, target, motion, datamap, x_bounds, y_bounds, z_bounds,
                   astar_version='v1', max_iterations=10_000,):
    
    # make astar object and search 
    astar = Astar(datamap, x_bounds, y_bounds, z_bounds, astar_version)
    path, outer_iterations, result = astar.search(np.array(start), np.array(target), motion=motion, max_iterations=max_iterations)
    print('astar path result:', result, '. Astar took', outer_iterations,'number of iterations to finish. Increase max_iterations if failed and you can see that a path exists.')
    
    return path, outer_iterations, result


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
    # roof_collision is allowed distance from above rootop before collision
    def __init__(self, datamap, x_bounds, y_bounds, z_bounds, astar_version='v1'):

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
        self.datamap = datamap
        self.x_bounds = x_bounds.copy()
        self.y_bounds = y_bounds.copy()
        self.z_bounds = z_bounds.copy()
        self.max_horizontal = max(self.magnitudes_horizontal)
        self.max_vertical = max(self.magnitudes_vertical)
        self.astar_version = astar_version
    
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
    def search(self, start, end, motion='2d', max_iterations=10_000):
        start_node = Node(start, 0, 'Spawn', None)

        to_visit = {}
        visited = {}
        to_visit[start_node.name] = start_node

        outer_iterations = 0
    
        def check_child(position, direction, check_visited=True, check_bounds=True, check_collision=True):
            name = f'{position[0]}_{position[1]}_{position[2]}_{direction}'
            
            # check if visited already
            if check_visited and name in visited:
                return 

            # unpack positional vars
            x, y, z = position

            # check bounds
            if check_bounds and self.datamap.out_of_bounds(x, y, z, self.x_bounds, self.y_bounds, self.z_bounds):
                return False

            # check collision
            if check_collision and self.datamap.in_object(x, y, z, self.roof_collision):
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
            #print('current', current_position, current_direction)

            # child node for rotating right
            proposed_direction = current_direction + 1
            if proposed_direction == 4:
                proposed_direction = 0
            if check_child(current_position, proposed_direction):
                children.append(Node(current_position, proposed_direction, 'RotateRight', current_node))

            # child node for rotating left
            proposed_direction = current_direction - 1
            if proposed_direction == -1:
                proposed_direction = 3
            if check_child(current_position, proposed_direction):
                children.append(Node(current_position, proposed_direction, 'RotateLeft', current_node))

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
                    name = f'{proposed_position[0]}_{proposed_position[1]}_{proposed_position[2]}_{current_direction}'
                    if name not in visited:
                        children.append(Node(proposed_position, current_direction, f'Forward{magnitude}', current_node))

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
                                action = f'Downward{magnitude}'
                            if updown == 1:
                                action = f'Updward{magnitude}'
                            name = f'{proposed_position[0]}_{proposed_position[1]}_{proposed_position[2]}_{current_direction}'
                            if name not in visited:
                                children.append(Node(proposed_position, current_direction, action, current_node))
                
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
                    for i in range(len(self.magnitudes_horizontal)-1, -1, -1):
                        mag = self.magnitudes_horizontal[i]
                        while x_tol_dis_abs >= mag:
                            child.h += 1
                            x_tol_dis_abs -= mag
                if x_tol_dis_abs > 0:
                    child.h += 1
                y_tol_dis = child.position[1] - end[1] 
                y_tol_dis_abs = np.abs(y_tol_dis) - self.goal_tolerance
                if y_tol_dis_abs > 0:
                    if child.direction != 0 and y_tol_dis < 0:
                        child.h += 1
                    elif child.direction != 2 and y_tol_dis > 0:
                        child.h += 1
                    for i in range(len(self.magnitudes_horizontal)-1, -1, -1):
                        mag = self.magnitudes_horizontal[i]
                        while y_tol_dis_abs >= mag:
                            child.h += 1
                            y_tol_dis_abs -= mag
                if y_tol_dis_abs > 0:
                    child.h += 1
                z_tol_dis = child.position[2] - end[2] 
                z_tol_dis_abs = np.abs(z_tol_dis) - self.goal_tolerance
                if z_tol_dis_abs > 0:
                    for i in range(len(self.magnitudes_vertical)-1, -1, -1):
                        mag = self.magnitudes_vertical[i]
                        while z_tol_dis_abs >= mag:
                            child.h += 1
                            z_tol_dis_abs -= mag
                if z_tol_dis_abs > 0:
                    child.h += 1
                    
                child.f = child.g + child.h

                # check against other child
                if child.name in to_visit and child.g > to_visit[child.name].g:
                    continue

                # add child to to_visit 
                to_visit[child.name] = child
            #print(to_visit.keys())
                
        return self.return_path(current_node), outer_iterations, 'no children'
