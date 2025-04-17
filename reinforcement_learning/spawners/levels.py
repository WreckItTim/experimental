from spawners.spawner import Spawner
from component import _init_wrapper
import random
import numpy as np
import math
import utils.global_methods as gm
import pickle
import copy

# reads static list of paths, in a levels hiearchy (increasing difficulty)
# data structure of paths file (saved from pickle binary):
# {
# 	paths : [], # list of np.arrays where each np.array is a path of x,y,z points going from start to goal
# 	linearitys: [], # list of linear distances from start to goal for each path above
# 	nonlinearitys: [], # list of nonlinear values for each path = total_distance_traveled/linearity
# 	levels: {}, # dictionary, where each key is the level name (I just use the level numner as name) 
# 					# and where each value is a list of sublevels,
# 					# each sublevel above has a list of path indexes that belong to that sublevel
# }
class Levels(Spawner):
	# levels path is file path to data structure as outlined above
	# constructor
	@_init_wrapper
	def __init__(self,
				drone_component, # drone to spawn at given path start
				levels_path, # path to pickle file with paths info
				min_level, # inclusive, minimum path level to sample from
				max_level, #inclusive, maximum path level to sample from
				level = None, # current level to sample paths from at probability = level_proba, if None then = min_level
				random_path = False, # True will sample paths at current level, False will rotate through all paths
				yaw_type = 0, # 'face': faces goal, 'random': random full range, 'value': specific yaw value
				rotating_idx = 0, # used for static path selection
				path_splits = { # percent splits to apply to data at levels_path -- None will apply no split
					'train':[0, 0.6], # first [0-60)% of test paths
					'val':[0.6, 0.8], # first [60-80)% of test paths
					'test':[0.8, 1], # first [80-100]% of test paths
				}, # split_name : [inclusive, exclusive)    >>>>     unless =1 then will be all inclusive
									# I typically use [0.6, 0.2, 0.2] for a train, val, test split
									# this will also potentially be used by the curriculum learning object
				split_name = 'train', # name of path_splits to use, for example for 'train', 'val', 'test'
				level_proba = 1.0, # when getting a random path, this is the proba of sampling the current level 
										# -- otherwise randomly rolls a previous level
			):
		super().__init__()


	def connect(self):
		super().connect()
		self._level_info = gm.pk_read(self.levels_path)
		self._paths = self._level_info['paths']
		self._nonlinearitys = self._level_info['nonlinearitys']
		self._linearitys = self._level_info['linearitys']
		self._levels = self._level_info['levels']
		self._sublevels = self._level_info['sublevels']
		if self.level is None:
			self.level = self.min_level
		self.set_level(self.level)
		# get raw path idx values based on path_splits percents
		n_paths = len(self._paths) # all possible paths
		if self.path_splits is None:
			start_idx = 0
			end_idx = n_paths
		else:
			start_perc, end_perc = self.path_splits[self.split_name]
			start_idx = int(n_paths*start_perc)
			end_idx = int(n_paths*end_perc)-1 if end_perc < 1 else n_paths
		# place path idxs into bins based on level and sublevel
		self._path_bins = {}
		self.n_paths = 0 # keep track of paths we keep to sample from
		for path_idx in range(start_idx, end_idx):
			level = self._levels[path_idx]
			if level < self.min_level or level > self.max_level:
				continue
			sublevel = self._sublevels[path_idx]
			if level not in self._path_bins:
				self._path_bins[level] = {}
			if sublevel not in self._path_bins[level]:
				self._path_bins[level][sublevel] = []
			self._path_bins[level][sublevel].append(path_idx)
			self.n_paths += 1
		self.avail_levels = list(self._path_bins.keys())
		print('LEVELS', 'split name', self.split_name, 'nPaths', self.n_paths)

	def set_level(self, level):
		if level >= self.min_level and level <= self.max_level:
			self.level = level
			self.rotating_idx = 0
		
	def reset_rotating(self):
		self.rotating_idx = 0

	def reset_learning(self):
		if not self.random_path:
			self.reset_rotating()

	def spawn(self):
		if self.random_path:
			# randomize what level we sample from
			die_roll = random.random()
			if die_roll <= self.level_proba:
				level = self.level
			else:
				level = random.choice(self.avail_levels)
			# randomize what sublevel we sample from
			sublevels = list(self._path_bins[level].keys())
			sublevel = random.choice(sublevels)
			# sample path from given level and sublevel
			path_idx = random.choice(self._path_bins[level][sublevel])
		else:
			# get next path in static rotating list
			self.rotating_idx = self.rotating_idx+1 if self.rotating_idx+1 < self.n_paths else 0  
			path_idx = self.rotating_idx
		
		self._path_idx = path_idx	
		self._path = self._paths[path_idx]
		self._astar_length = len(self._path)
		self._nonlinearity = self._nonlinearitys[path_idx]
		self._linearity = self._linearitys[path_idx]
		self._level = self._levels[path_idx]
		self._sublevel = self._sublevels[path_idx]
		self._start_x, self._start_y, self._start_z = self._path[0]['position']
		self._goal_x, self._goal_y, self._goal_z = self._path[-1]['position']
		
		if self.yaw_type in ['face']:
			distance_vector = np.array([self._goal_x, self._goal_y, self._goal_z]) - np.array([self._start_x, self._start_y, self._start_z])
			self._start_yaw = math.atan2(distance_vector[1], distance_vector[0])
		elif self.yaw_type in ['random']:
			self._start_yaw = np.random.uniform(-1*math.pi, math.pi)
		else:
			self._start_yaw = self.yaw_type
			
		self._drone.teleport(self._start_x, self._start_y, self._start_z, self._start_yaw, ignore_collision=True, stabelize=True)
		self._drone.check_collision()
		
	def reset(self, state=None):
		self.spawn()
		if state is not None:
			state['path_idx'] = self._path_idx
			state['astar_length'] = self._astar_length
			state['path_linearity'] = self._linearity
			state['path_nonlinearity'] = self._nonlinearity
			state['level'] = self._level
			state['sublevel'] = self._sublevel
