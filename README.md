This is an experimental github repo to share airsim data and models

To use, download this repo then drag and drop any large file data you need from the dropbox into map_data/observations/...:

please message me for a Dropbox link

see the map_data/visualize_data.ipynb notebook for an example of how to fetch and visualize data from a given map, sensor, and coordinates and/or navigation path

see the supervised_learning/monocular_depth_example.ipynb notebook of how to fetch training/testing data to train a CNN model for monocular depth extraction

The monocular_depth notebook uses the data SceneV1 and DepthV1. For weather, you will need SceneV3, SceneV4, and SceneV5. You can see the bottom of the readme.txt file in the map_data/observations folder for all of the different data available in the catalogue.
