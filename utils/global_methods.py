import os
import pickle
import json
import time
import shutil
import pickle
import platform
import random
import platform
import datetime
import numpy as np
import torch as th
import psutil

def check_ram(msg=''):
    print(f'{msg} {psutil.virtual_memory().used*1e-9:0.4f} gb')

# global PARAMS
global_parameters = {}
def set_globals(params):
	global_parameters.update(params)
def set_global(key, value):
	global_parameters[key] = value
def get_global(key):
	if key not in global_parameters:
		return None
	else:
		return global_parameters[key]

def fix_directory(directory):
	if '\\' in directory:
		directory = directory.replace('\\', '/')
	if directory[-1] != '/':
		directory += '/'
	return directory
# setup paths and some global params
def setup_output_dir(output_dir, overwrite_directory=False):
	# set working directory
	if overwrite_directory and os.path.exists(output_dir):
		shutil.rmtree(output_dir)
	os.makedirs(output_dir, exist_ok=True)
	# make temp folder if not exists
	#os.makedirs(f'{output_dir}temp/', exist_ok=True)
	# set operation system
	set_global('OS', platform.system().lower())
	# save working directory path to global_parameters to be visible by all 
	set_global('output_dir', output_dir) # relative to repo
	# absoulte path on local computer to repo
	set_global('absolute_path',  os.getcwd() + '/')

# used for argvs but can be used for whatevs
	# inputs string following dictionary format of 'key1:value1 key2:value2 ... keyN:valueN'
	# set_global will update global params dictionary variable with arguments dict
def isint(s):
	for c in s:
		if c not in ['-','0','1','2','3','4','5','6','7','8','9']:
			return False
	return True
def isfloat(s):
	for c in s:
		if c not in ['-','0','1','2','3','4','5','6','7','8','9','.','e']:
			return False
	return True
def args_to_str(args):
	s = ''
	for key in args:
		s += f'{key}:{args[key]} '
	return s
def parse_arguments(arguments, set_global_arguments=True):
	dictionary = {}
	for keyvalue in arguments:
		parts = keyvalue.split(':')
		key = parts[0]
		value = ':'.join(parts[1:])
		print(key, value)
		if value[0]=='{':
			value = parse_arguments(value[1:-1].split(','), set_global_arguments=False)
		elif value[0]=='[':
			value = value[1:-1].split(',')
		elif value in ['True']:
			value = True
		elif value in ['False']:
			value = False
		elif isint(value):
			value = int(value)
		elif isfloat(value):
			value = float(value)
		dictionary[key] = value
		if set_global_arguments:
			set_global(key, value)
	return dictionary

# COMMUNICATE WITH USER
local_log = []
def add_to_log(msg):
	local_log.append(get_timestamp() + ': ' + msg)
	#print_local_log()
def print_local_log():
	file = open(get_global('output_dir') + 'log.txt', 'w')
	for item in local_log:
		file.write(item + "\n")
	file.close()
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

# **** common utility funcitons **** 
def set_random_seed(random_seed):
	random.seed(random_seed)
	np.random.seed(random_seed)
	th.manual_seed(random_seed)    
	if th.cuda.is_available():
		th.cuda.manual_seed_all(random_seed) 
	set_global('random_seed', random_seed)


# STOPWATCH
# simple stopwatch to time whatevs, in (float) seconds
# keeps track of laps along with final time
class Stopwatch:
	def __init__(self):
		self.start()
	def start(self):
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
def pk_read(path):
	return pickle.load(open(path, 'rb'))
def pk_write(obj, path):
	pickle.dump(obj, open(path, 'wb'))
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
def to_datetime(timestamp):
	format_string = '%Y_%m_%d_%H_%M_%S'
	datetime_object = datetime.datetime.strptime(timestamp, format_string)
	return datetime.datetime.timestamp(datetime_object)
def progress(name, progress):
	if name is None:
		return
	old_progress = get_global('progress')
	set_global('progress', progress)
	progress_dir = get_global('local_dir') + 'progress/'
	if old_progress is not None:
		old_path = f'{progress_dir}{name} {old_progress}'
		if os.path.exists(old_path):
			os.remove(old_path)
	new_path = f'{progress_dir}{name} {progress}'
	if not os.path.exists(progress_dir):
		os.makedirs(progress_dir)
	pk_write('', new_path)