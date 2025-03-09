
def get_bounds(airsim_map, region):
    # bounds drone can move in
    z_bounds = [-40, 0]
    if region in ['astar']:
        z_bounds = [4, 16] # set the valid bounds to explore for z-values in astar coords
    if airsim_map in ['Blocks']:
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
    if airsim_map in ['AirSimNH']:
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
    return x_bounds, y_bounds, z_bounds

# this is interms of rooftops v1
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