import json
import time
import math
import os
import shutil
import pickle as pk
import platform
import random
import numpy as np
import torch as th

def pk_read(path):
	return pk.load(open(path, 'rb'))

def pk_write(obj, path):
	return pk.dump(obj, open(path, 'wb'))

def read_json(path):
	return json.load(open(path, 'r'))

def write_json(dictionary, path):
	json.dump(dictionary, open(path, 'w'), indent=2)

def get_timestamp():
	secondsSinceEpoch = time.time()
	time_obj = time.localtime(secondsSinceEpoch)
	timestamp = '%d_%d_%d_%d_%d_%d' % (
		time_obj.tm_year, time_obj.tm_mon, time_obj.tm_mday,  
		time_obj.tm_hour, time_obj.tm_min, time_obj.tm_sec
	)
	return timestamp

def setup(working_directory = None, 
		  overwrite_directory=False,
		  project_name='void',
		  experiment_name='void',
		  trial_name='void',
		  run_name='void',
		  ):
	read_local_parameters()
	if working_directory is not None:
		if overwrite_directory and os.path.exists(working_directory):
			shutil.rmtree(working_directory)
		set_read_write_paths(working_directory = working_directory)
		read_local_log()
	set_operating_system()
	set_local_parameter('project_name', project_name)
	set_local_parameter('experiment_name', experiment_name)
	set_local_parameter('trial_name', trial_name)
	set_local_parameter('run_name', run_name)

def set_operating_system():
	import platform
	OS = platform.system().lower()
	set_local_parameter('OS', OS)
	speak(f'detected operating system:{OS}')
	
# end all folder paths with /
def fix_directory(directory):
	if '\\' in directory:
		directory = directory.replace('\\', '/')
	if directory[-1] != '/':
		directory += '/'
	return directory

# set up folder paths for file io
def set_read_write_paths(working_directory):
	# make temp folder if not exists
	if not os.path.exists('temp/'):
		os.makedirs('temp/')
	# make working directory if not exists
	working_directory = fix_directory(working_directory)
	if not os.path.exists(working_directory):
		os.makedirs(working_directory)
	# save working directory path to local_parameters to be visible by all 
	set_local_parameter('working_directory', working_directory) # relative to repo
	# absoulte path on local computer to repo
	set_local_parameter('absolute_path',  os.getcwd() + '/')

# set up controller to run configuration on
def get_controller(controller_type, 
				total_timesteps = 1_000_000,
				continue_training = True,
				model_component = 'Model',
				environment_component = 'TrainEnvironment',
				drone_component = 'Drone',
				evaluator_component = 'Evaluator',
				actor_component = 'Actor',
				log_interval = -1,
				evaluator = 'Evaluator',
				project_name = 'void',
				):
	# create CONTROLLER - controls all components (mode)
	controller = None
	speak(f'CONTROLLER = {controller_type}')
	# debug mode will prompt user input for which component(s) to debug
	if controller_type == 'Debug':
		from controllers.debug import Debug
		controller = Debug(
			drone_component=drone_component,
			)
	# train will create a new or read in a previously trained model
	# set continue_training=True to pick up where learning loop last saved
	# or set continue_training=False to keep weights but start new learning loop
	elif controller_type == 'Train':
		from controllers.train import Train
		controller = Train(
			model_component = model_component,
			environment_component = environment_component,
			evaluator_component = evaluator_component,
			total_timesteps = total_timesteps,
			log_interval = log_interval,
			continue_training = continue_training,
			project_name = project_name,
			)
	# evaluate will read in a trained model and evaluate on given environment
	elif controller_type == 'Evaluate':
		from controllers.test import Evaluate
		controller = Evaluate(
			evaluator_component = evaluator_component,
			)
	# checks will run drift checks
	elif controller_type == 'AirSimChecks':
		from controllers.airsimchecks import AirSimChecks
		controller = AirSimChecks(
			drone_component = drone_component,
			actor_component = actor_component,
			)
	# will teleport drone to specified points and collect data
	elif controller_type == 'Data':
		from controllers.data import Data
		controller = Data(
			model_component = model_component,
			environment_component = environment_component,
			drone_component = drone_component,
			)
	else:
		from controllers.empty import Empty
		controller = Empty()
	return controller


# local PARAMS
local_parameters = {}
def read_local_parameters(path = 'local/local_parameters.json'):
	if os.path.exists(path):
		local_parameters.update(read_json(path))

def write_local_parameters(path = 'local/local_parameters.json'):
	write_json(local_parameters, path)

def del_local_parameter(key):
	if key in local_parameters:
		del local_parameters[key]

def set_local_parameter(key, value):
	local_parameters[key] = value

def get_local_parameter(key):
	if key not in local_parameters:
		return None
	else:
		return local_parameters[key]


# COMMUNICATE WITH USER
local_log = []
def add_to_log(msg):
	local_log.append(get_timestamp() + ': ' + msg)
	#print_local_log()
def print_local_log():
	file = open(get_local_parameter('working_directory') + 'log.txt', 'w')
	for item in local_log:
		file.write(item + "\n")
	file.close()
def read_local_log():
	path = get_local_parameter('working_directory') + 'log.txt'
	if os.path.exists(path):
		with open(path) as file:
			for line in file:
				local_log.append(line.rstrip())

def speak(msg):
	add_to_log(msg)
	print(msg)

def prompt(msg):
	speak(msg)
	return get_user_input()

def get_user_input():
	return input()

def error(msg):
	add_to_log(msg)
	raise Exception('ERROR:', msg)

def warning(msg):
	speak('WARNING:', msg)

def set_random_seed(random_seed):
	random.seed(random_seed)
	np.random.seed(random_seed)
	th.manual_seed(random_seed)    
	if th.cuda.is_available():
		th.cuda.manual_seed_all(random_seed) 
	set_local_parameter('random_seed', random_seed)
		
def get_random_seed(random_seed):
	return get_local_parameter('random_seed')


# STOPWATCH
# simple stopwatch to time whatevs, in (float) seconds
# keeps track of laps along with final time
class StopWatch:
	def __init__(self):
		self.start_time = time.time()
		self.last_time = self.start_time
		self.laps = []
	def lap(self):
		this_time = time.time()
		delta_time = this_time - self.last_time
		self.laps.append(delta_time)
		self.last_time = this_time
		return delta_time
	def stop(self):
		self.stop_time = time.time()
		self.delta_time = self.stop_time - self.start_time
		return self.delta_time


def yaw_to_idx(yaw):
	while yaw < 0:
		yaw += 2*np.pi
	while yaw >= 2*np.pi:
		yaw -= 2*np.pi
	yaw_idx = round(yaw/(np.pi/2))
	# if yaw_idx == -1:
	# 	yaw_idx = 3
	# if yaw_idx == 4:
	# 	yaw_idx = 0
	return yaw_idx

def db_setup(seed_mult=1_000, levels_in=0):
	OS = platform.system()
	if levels_in > 0:
		cwd_path = os.getcwd()
		delim = '\\' if OS in ['Windows'] else '/'
		parts = cwd_path.split(delim)
		dropbox_path = '/'.join(parts[:-1*levels_in]) + '/'
		home_path = '/'.join(parts[:-1*levels_in-1]) + '/'
	else:
		dropbox_path = os.getcwd() + '/'
		delim = '\\' if OS in ['Windows'] else '/'
		parts = cwd_path.split(delim)
		home_path = '/'.join(parts[:-1]) + '/'
	models_path = dropbox_path + 'models/'
	data_path = dropbox_path + 'data/'
	local_path = home_path + 'local/'
	temp_path = local_path + 'temp/'
	local_parameters = json.load(open(local_path + 'local_parameters.json', 'r'))
	instance_name = local_parameters['instance_name']
	speak('detected instance named ' + instance_name)
	instance_seeds = {
		'hephaestus':1*seed_mult,
		'apollo':2*seed_mult,
		'fox':3*seed_mult,
		'flareon':4*seed_mult,
		'ace':5*seed_mult,
		'magma':6*seed_mult,
		'pyro':7*seed_mult,
		'torch':8*seed_mult,
		'phoenix':9*seed_mult,
		'tron':10*seed_mult,
		'ifrit':11*seed_mult,
	}
	random_seed = instance_seeds[instance_name]
	set_random_seed(random_seed) 


	set_local_parameter('models_path', models_path)
	set_local_parameter('data_path', data_path)
	set_local_parameter('local_path', local_path)
	set_local_parameter('temp_path', temp_path)
	set_local_parameter('instance_name', instance_name)
	set_local_parameter('random_seed', random_seed)


	return instance_name, dropbox_path, local_path, data_path, models_path


def progress(name, progress):
	old_progress = get_local_parameter('progress')
	set_local_parameter('progress', progress)
	local_path = get_local_parameter('local_path')
	if old_progress is not None:
		old_path = f'{local_path}progress/{name} {old_progress}'
		if os.path.exists(old_path):
			os.remove(old_path)
	new_path = f'{local_path}progress/{name} {progress}'
	pk_write('', new_path)

def save_png(img, path, rgb=True):
	if rgb:
		# rgb to bgr
		img = img.copy()
		temp = img[0, :, :].copy()
		img[0, :, :] = img[2, :, :].copy()
		img[2, :, :] = temp
		# move channel first to channel last
		img = np.moveaxis(img, 0, 2)
	else:
		# remove channel for greyscale
		img = img[0]
	img = Image.fromarray(img)
	img.save(path)


# read_directory is the folder with trained model output
# eval_name will create sub_folder within read_directory with all output from evaluations
def evaluate2(config_path, model_path, working_directory):
	os.system(f'python _test.py {config_path} {model_path} {working_directory}')

def evaluate(test_name, start_level, max_level, num_per, model_name, results_dic=None):
	working_directory = get_local_parameter('working_directory')
	airsim_map = get_local_parameter('airsim_map')
	motion = get_local_parameter('motion')
	project_name = get_local_parameter('project_name')
	experiment_name = get_local_parameter('experiment_name')
	trial_name = get_local_parameter('trial_name')
	run_name = get_local_parameter('run_name')
	astar_name = get_local_parameter('astar_name')
	overwrite = True
	device = get_local_parameter('device')
	local_path = get_local_parameter('local_path')
	test_params = f'{airsim_map} {motion} {project_name} {experiment_name} {trial_name} {run_name} {test_name} {astar_name} {model_name} {start_level} {max_level} {num_per} {overwrite} {device}'
	os.system(f'python _client_test.py {test_params}')
	job_name = f'testNavi_{airsim_map}_{motion}_{project_name}_{experiment_name}_{trial_name}_{run_name}_{test_name}_{astar_name}_{model_name}_{start_level}_{max_level}_{num_per}_{overwrite}'
	progress_path = f'{local_path}progress/{job_name} complete'
	while not os.path.exists(progress_path):
		time.sleep(1)
	os.remove(progress_path)
	test_results = read_json(f'{working_directory}{test_name}/evaluation.json')
	if results_dic is not None:
		results = {
			'test_accuracy':float(100*np.mean(test_results['successes'])),
			'test_length':float(np.mean(test_results['lengths'])),
		}
		results.update(results_dic)
		write_json(results, working_directory+'results.json')
	return test_results