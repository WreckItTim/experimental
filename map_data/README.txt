
Directory breakdown:

    grids -- (ignore) intermediate values used to convert voxels to rooftops 
    observations -- (main attraction) contains all data collected from AirSim, please see observations/README.txt
    paths -- (ignore) random Astar paths found for each map, for each type of drone motion, for each region of map
    rooftops -- (ignore) 2d dictionaries indexed by x,y showing the z-value of highest object at that x,y location
    voxels -- (ignore) 3d 3rd party data structure extracted from AirSim maps to detect surface objects at x,y,z


Using data viewing/collection methods:
    1. can view and/or collect data at defined set of x,y,z,yaw coordinates or from a prefound path or from a desired path between two points.
    2. use make_animation=True to view video of data
        # can easily run out of memory if number of data points is too high (>100 or so)
    3. use return_data=True to return a dictionary of data for model training/testing. This dictionary is:
        {
            'observations':{sensor_name:set}, # for each desired sensor_name (e.g. SceneV1 DepthV2 etc.)
                                    # BGR color image set is a np.array of shape [N, 3, height, width]
                                    # grayscale image set is a np.array of shape [N, 1, height, width]
                                    # bounding boxes set is a list of length N of dictionaries: {obj_name:[x,y,w,h]}
                                    # boolean masks set is a list of length N of dictionaries: {obj_name:[x,y,w,h]}
            'coordinates':[], # np.array of shape [N,x,y,z,yaw] coordinates coorisponding to same idxs as above
        }




observations folder contains all of data
