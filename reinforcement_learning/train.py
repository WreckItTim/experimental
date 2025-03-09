import global_methods as md
from configuration import Configuration
import os
import modules


airsim_map = 'Blocks' # name of AirSim release to load
motion = '2d' # 2d for horizontal motion only, 3d to unlock vertical motion
x_bounds, y_bounds, z_bounds = modules.bounds_v1(airsim_map, motion) # bounds that drone can move around map in
data_path = '/home/tim/Dropbox/data/' # CHANGE to your path -- parent directory to read all data from
working_directory = f'{data_path}models/navigation_airsim_blocks_dqn_2d/' # CHANGE to your path -- where to write trained model and config files to
evaluate_at_end = True # set to true will auto evaluate on test paths after training and save to f'{working_directory}test/'

# setup for run, set system vars and prepare file system
overwrite_model = True # this will overwrite previous working_directory and start new model from scratch
md.setup(working_directory, overwrite_model)

# paramount training hyper parameters
continue_training = False # True will load saved json configuration and any previous model and replay buffer saved to file
if os.path.exists(working_directory + 'configuration.json'):
	continue_training = True
total_timesteps = int(2e6) # maximum number of timesteps to train on
	# SB3 default is 1e6, Microsoft uses 5e5
buffer_size = int(1e5) # number of recent steps (observation, action, reward) tuples to store in memory to train on -- takes up memory
	# ^^ SB3 default is 1e6, Microsoft uses 5e5, I typically have to use less because of memory constraints
#exploration_fraction =  0.1 # set this as a percent of total_timeteps
stop_annealing = int(1e4) # alternatively set a specific number of steps to stop annealing the exploration rate at
exploration_fraction = stop_annealing / total_timesteps
	# SB3 default is 0.1*total_timesteps, Microsoft uses 5e4
start_level, max_level = 0, 5 # range of difficulty levels to sample training paths from during curriculum learning (inclusive)

# make controller to run configuration on (we will train a model)
from controllers.train import Train
controller = Train(
	model_component = 'Model',
	environment_component = 'Environment',
	continue_training = continue_training,
	total_timesteps = total_timesteps,
	)

# continue training will load runs folder and pick up where it left off
if continue_training:
	# load configuration (with all components), model, and replay buffer from file
	configuration = modules.continue_training_v1()
	configuration.set_controller(controller)

# if not continuing, then we will create a new configuration, and accompanying components, from scratch
else:
	# set meta data (anything you want here for notes, just writes to config file as a dict)
	meta = {}
	## make a new configuration file to add components to 
		# this obj will be used to serialize components, and log experiment history
		# any components created after making this configuration file will auto be added to it
		# components use the name of other components which will be created and connected later
		# this is done to handle different objects which need to be connected to eachother and connected in different orders
		# there is a baked in priority que for connecting components if you need to handle this differently
	configuration = Configuration(
		meta, 
		)
	configuration.set_controller(controller)
	
	# set global paramseters for config file
	configuration.parameters['airsim_map'] = airsim_map
	configuration.parameters['motion'] = motion
	
	# make environment to navigate in
	modules.environment_v1(x_bounds, y_bounds, z_bounds, data_path, airsim_map)
	
	# set reward function
	modules.rewards_v1(x_bounds, y_bounds, z_bounds, motion)
	
	# set action space
	modules.actions_v1(motion)

	# set observation space
	observations_path = f'{data_path}observations/' # parent directory path where data_dicts are stored for each observation
		# make sure data is saved like: f'{observation_path}observations/{sensor_name}/{airsim_map}/{data_dicts}.p'
	forward_sensor = 'DepthV2' # forward sensor name to use as input to navigation model
	modules.observations_v1(observations_path, forward_sensor, x_bounds, y_bounds, z_bounds, airsim_map, motion)

	# set learning parameters and model
	modules.learning_v1(buffer_size, exploration_fraction, data_path, airsim_map, motion, start_level, max_level)


	# **** additional modifier components here -- such as when and what to save ****
	
	#  SAVE MODEL AND/OR REPLAY BUFFER every checkpoint_freq many training episodes
	from modifiers.saver import Saver
	checkpoint_freq = total_timesteps # default is to just save at end of training
	Saver(
		base_component = 'Model', # Modifiers will execute after/before this given component is used
		parent_method = 'end', # Modifiers will execute after/before this given component's method is used
		order = 'post', # Modifiers will run code either before (pre) or after (post) given component call
		track_vars = [
					'model', 
					#'replay_buffer', # this can cost alot of memory so default is to not save
					],
		write_folder = working_directory + 'modeling/',
		frequency = checkpoint_freq, # save every this many episodes
		name = 'ModelingSaver',
	)

	# # UNCOMMENT THIS TO SAVE TRAINING OBSERVATIONS AND/OR STATES TO FILE AT EACH STEP
	# saver = Saver(
	# 	base_component = 'Environment',
	# 	parent_method = 'end',
	# 	track_vars = [
	# 				'observations', 
	# 				'states',
	# 				],
	# 	write_folder = working_directory + 'states/',
	# 	order = 'post',
	# 	save_config = False,
	# 	save_benchmarks = False,
	# 	frequency = checkpoint_freq,
	# 	name='Saver',
	# )


# CONNECT COMPONENTS
configuration.connect_all()

# WRITE CONFIGURATION
configuration.save()

# WRITE CONTROLLER
controller.save(working_directory + 'train_controller.json')

# RUN CONTROLLER
configuration.controller.run()

# clean up shop
configuration.disconnect_all()
md.speak('training complete!')

#  evaluate on trained model
if evaluate_at_end:
	md.evaulate(
		f'{working_directory}configuration.json',
		f'{working_directory}modeling/model_final.zip',
		f'{working_directory}test_final/',
	)
	md.speak('testing complete!')