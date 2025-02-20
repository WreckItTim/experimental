
paths are found randomly from an A-star algorithm with an admissible heuristic to insure shortest path
a random starting and target point are sampled from the given airsim map bounds then...
    starting positions always have a yaw of 0
x, y, z, yaw coordinates are discretized since A-star can not handle continuous coordinates
at each iteration (node), the A-star algo can take one of many actions that include:
    rotating either right or left, at a full 90 degree rotations
    moving forward with a step size equal to one of many discrete magnitudes, measured in integer meters
    moving up or down at a discrete step size of 4 meters
    can only move on even x,y coordinates and multiple-of-4 z coordiantes (to reduce complexity and data space)
    x, y, z follow drone coordinates:
        x is directed + forward and - backward from origin (0,0,0)
        y is directed + right and - left from origin (0,0,0)
        z is directed - up and + down from origin (0,0,0)
    yaw is an integer between [0, 3] where:
        0 >>>> forward facing, 0 radians
        1 >>>> right facing, pi/2 radians
        2 >>>> backward facing, pi radians
        3 >>>> left facing, -pi/2 radians
    a cell at the given x,y,z coordinate is considered blocked based on the voxels data returned from AirSim
        I have the voxels data saved as rooftops objects (this is used to visualize the map as well)
        this means that the highest z-coordinate for a given x,y position sets the rooftop
            all z-values below a given rooftop are considred blocked, so there is no flying underneath an object
        if you need access to the original voxels data, please let me know


each pickle file is named as such:

    airsimMap_motion_region.p

    airsimMap >>>> which airsim map paths were found on
        right now either Blocks or AirSimNH
    motion >>>> how the drone could move around map, 
        right now either 2d (horizontal plane only) or 3d (can go up and down)'
    region >>>> what part of the map paths were found on
        right now either train or test
        train is top half of map (with some overlap in middle)
        test is bottom half of map (with some overlap in middle) 


each pickle file is a dictionary following this structure:
    {
        'paths': list where each item in the list is another dictionary (see below)
        'levels': dictionary where the key is the difficulty level and value is a list of path indexes from above
        'linearitys': list of the euclidean distance between start and target the path at the same index
        'nonlinearitys': list of the number of non-forward actions taken during the path at the same index
        'level_ranges': list of ranges of linearity and nonlinearity values used to determine each difficulty level
            [linearity_min, linearity_max, nonlinearity_min, nonlinearity_max] for each level
        'linearity_bounds': (ignore) list of linearity values used to determine each sublevel for sampling
        'nonlinearity_bounds': (ignore) list of nonlinearity values used to determine each sublevel for sampling
    }

    paths list dictionary:
    {
        'position': [x, y, z]
        'direction': yaw
        'action': an integer value between [-1, 8] (see list of actions below)
    }

    actions (you can likely ignore these and just use the position and direction variables):
        -1 >>>> starting point, no action
        0 >>>> rotate right 90 degrees
        1 >>>> rotate left 90 degrees
        2 >>>> move forward 2 meters
        3 >>>> move forward 4 meters
        4 >>>> move forward 8 meters
        5 >>>> move forward 16 meters
        6 >>>> move forward 32 meters
        7 >>>> move downward 4 meters
        8 >>>> move upward 4 meters