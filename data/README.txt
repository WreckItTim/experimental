
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




currently known AirSim issues (and hopefully fixes):
    1. AirSim will randomly crash the longer it runs
        > added a crash handler to reboot AirSim on crash (using msgpacket timeouts to determine crash)
    2. if moving drone using physics, has tendency to irraticaly spin out of control
        > I added a stabelize feautre that 0s out the yaw, pitch, roll when this happens -- it seems to work from my observations
        > I found that using drone.teleport() is stable and quicker, however does not consider all physics in simulation
    3. if running for a long time AirSim will slow down, also causing other render-on-time-before-observation issues
        > decreased collection size for smaller data collection windows
    4. observation will view the previous spot in AirSim (this might be because it needs more time to render)
        > added a cache to view previous observation and recapture if the observation is the same as the last one
    5. Zixia pointed out that weather will not render immediately when moving to new location
        > Zixia uses a time.sleep(1) in between each data point to mitigate this issue and give time to render
    6. data can be saved as floats rather than uint8s
        > adjusted the rl_drone/observation.to_numpy() to account for this (was changing to floats)
    7. data collection can be on wrong map
        > quick fix is to collect data one map at a time
        > long term fixes -- TODO: add as_kill() command at beginning of each run and wait till process is dead
                    TODO: add more checks at end of a run to insure proper cleanup and map was closed
    8. yaw, pitch, roll can be tilted incorrectly -- this seems to affect a large number of consequitve observations
        > added a time.sleep(1) inbetween each data point collection. seems to have worked
        > is this the same issue as numbers 4 and 5?
    9. when moving with physics and if colliding with an object while trying to move forward, AirSim can hang up by repeadily crashing into object
        > added a timeout to throw error inbetween timesteps and reboot simulation
    10. segmentation has some unwanted colors appearing on rare occasion (they are not mapped to anything by me?)