
observations directory structure:

visualization.ipynb >>>> use this jupyter python notebook to visualize data (this should be your first stop)
sensor_name / info.json >>>> human readable file with some details on the sensor data
sensor_name / airsim_map / file_name >>>> data collection files, explained below...


file_names:

data_dict__part_name.p
    this is the data dictionary you will be using the most
    it is indexed by [x][y][z][yaw_idx] where:
        yaw_idx==0 is facing forward (yaw of 0)
        yaw_idx==1 is facing right (yaw of pi/2)
        yaw_idx==2 is facing backward (yaw of pi)
        yaw_idx==3 is facing left (yaw of -pi/2)
        x, y, z follow drone coordinates (it is annoying):
        x is directed + forward and - backward from origin (0,0,0)
        y is directed + right and - left from origin (0,0,0)
        z is directed - up and + down from origin (0,0,0)
        typically, but not exact, the ground of the airsim_map is level and at z~=0
    the parts are currently 64x64x4 meter chunks from the given airsim_map, but might be different based on part_name
    
part_name follows this structure:
    id_repeat_xmin_xmax_xint_ymin_ymax_yint_zmin_zmax_zint
    where id is a unique id-name set by me, the first batch of data are just named 'aaa'
    repeat is the number of times data points were repeated at each location (this is for me to validate data)
    the min, max, int values represent the dataspace captured as range(min, max, int)
    data is collected at each data point for all 4 yaw_idx as noted above
        
point_list__part_name.p
    you can likely ignore this
    this is a list indexed in the order that each data point was captured
    note that airsim coordinates are not exact, so there may be some error in the desired position
    at each index is a dictionary following this structure
    {
        'x': x coordinate desired to capture data at, quite accurate
        'y': y coordinate desired to capture data at, quite accurate
        'z': z coordinate desired to capture data at, error within about +/- 0.1
        'yaw': yaw coordinate desired to capture data at, error within about +/- 1e-6
        'drone_x': x coordinate reported by AirSim right before data collection
        'drone_y': y coordinate reported by AirSim right before data collection
        'drone_z': z coordinate reported by AirSim right before data collection
        'drone_yaw': yaw coordinate reported by AirSim right before data collection
    }
        
log__part_name.p
    you can likely ignore this
    this is a list of dictionaries where each dictionary is a new entry
    an entry is logged by the data collection python file for mostly my purposes to validate data
    an entry can be one of many events that I deemed worthy of logging
        
data_list__part_name.p
    you can likely ignore this
    alternatively the data might be represented as a basic list rather than a dictionary
    I use this to recapture multiple observations at the same location to validate data
    if using data_list, you will need to match the same index between data_list and point_list

configuration.json 
    you can likely ignore this
    this has all of the config settings needed to run a full data collection run from my rl_drone repo
    you can load this file into my repo to rerun on a different set of coordinates
    view this if needed if curious about any end-to-end pipeline parameters used for data collection


sensors:

    DepthVX >>>> grayscale 1xHxW relative depths measured in meters w.r.t. camera lense on drone, uint8 (rounded down)
    SceneVX >>>> BGR 3xHxW camera capturing the scene in front of the drone, uint8
    SegmentationVX >>>> BGR 3xHxW where each color is a unique object, uint8
        the color map and object ids can be found in SegmentationVX / ...
    BoxesVX >>>> dictionary with all objects detected in scene, expressed as bounding box locations
        { object_name : [top_left_x_pixel, top_left_y_pixel, width_pixels, height_pixels] } -- inclusive
    MasksVX >>>> dictionary with all objects detected in scene HxW, expressed as numpy boolean masks
        { object_name : numpy_mask_array }


sensor_names:

    DepthV1 >>>> default resolution forward facing ground truth depth 1x144x256
    DepthV2 >>>> reduced resolution forward facing ground truth depth 1x36x64
    SceneV1 >>>> default resolution forward facing RGB scene camera 3x144x256
    SceneV3 >>>> RAIN default resolution forward facing RGB scene camera 3x144x256
    SceneV4 >>>> SNOW default resolution forward facing RGB scene camera 3x144x256
    SceneV5 >>>> FOG default resolution forward facing RGB scene camera 3x144x256
    SegmentationV1 >>>> default resolution forward facing object segmentation 3x144x256
    BoxesV1 >>>> dictionary from SegmentationV1
    MasksV1 >>>> dictionary from SegmentationV1


airsim_maps:

    Blocks - simple, small map with arbitrary shaped blocks, static objects only
    AirSimNH - medium map with realistic objects simulating a neighborhood, static objects only
