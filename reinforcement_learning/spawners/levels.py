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
				split_name, # split name to start at
				level = None, # current level to sample paths from at probability = level_proba, if None then = min_level
				start_direction = 0, # #### 'face': faces goal, 'random': random full range, 'value': specific yaw value
				rotating_idx = 0, # used for static path selection
				path_splits = { # percent splits to apply to data at levels_path -- None will apply no split
					'train':[0, 0.8], # first [0-80)% of test paths
					'val':[0.8, 0.9], # first [80-90)% of test paths
					'test':[0.9, 1], # first [90-100]% of test paths
				}, # split_name : [inclusive, exclusive)    >>>>     unless =1 then will be all inclusive
									# I typically use [0.6, 0.2, 0.2] for a train, val, test split
									# this will also potentially be used by the curriculum learning object
				random_splits = { # True will randomize path, False will iterate through all paths
					'train':True,
					'val':False,
					'test':False,
				},
				current_splits = { # True with use current level only, False will use min-max levels
					'train':True,
					'val':True,
					'test':False,
				},
				level_proba = 1.0, # when getting a random path, this is the proba of sampling the current level 
										# -- otherwise randomly rolls a previous level
			):
		super().__init__()

	def connect(self):
		super().connect()
		self._level_info = gm.pk_read(self.levels_path)
		self._paths = self._level_info['paths']
		self._starts = self._level_info['starts']
		self._goals = self._level_info['targets']
		self._levels = self._level_info['levels']
		self._euclideans = self._level_info['euclideans']
		self._nturns = self._level_info['nturns']
		# get raw path idx values based on path_splits percents
		n_total_paths = len(self._paths) # all possible paths
		self._level_bins = {}
		for path_idx in range(n_total_paths):
			level = self._levels[path_idx]
			if level < self.min_level or level > self.max_level:
				continue
			if level not in self._level_bins:
				self._level_bins[level] = []
			self._level_bins[level].append(path_idx)
		self._split_paths = {}
		for split_name in self.path_splits:
			self._split_paths[split_name] = {}
			start_perc, end_perc = self.path_splits[split_name]
			for level in self._level_bins:
				self._split_paths[split_name][level] = []
				n_level_paths = len(self._level_bins[level])
				start_idx = int(n_level_paths*start_perc)
				end_idx = int(n_level_paths*end_perc)
				# place path idxs into bins based on level and sublevel
				for level_idx in range(start_idx, end_idx):
					path_idx = self._level_bins[level][level_idx]
					self._split_paths[split_name][level].append(path_idx)
				#print('LEVELS', split_name, level, len(self._split_paths[split_name][level]))
		if self.level is None:
			self.level = self.min_level
		self.set_active_split(self.split_name)

	def set_active_split(self, split_name):
		self.split_name = split_name
		self._path_bins = self._split_paths[split_name]
		self._use_current = self.current_splits[split_name]
		self._random_path = self.random_splits[split_name]
		self.set_level(self.level)

	def set_level(self, level):
		if level >= self.min_level and level <= self.max_level:
			self.level = level
			if not self._random_path:
				self._path_idxs = []
				if self._use_current:
					for path_idx in self._path_bins[level]:
						self._path_idxs.append(path_idx) 
				else:
					for level in self._path_bins:
						for path_idx in self._path_bins[level]:
							self._path_idxs.append(path_idx) 
				self._n_split_paths = len(self._path_idxs)
				self.reset_rotating()
			else:
				self._n_split_paths = len(self._split_paths[self.split_name][level])
		
	def reset_rotating(self):
		self.rotating_idx = 0

	def reset_learning(self):
		self.reset_rotating()

	def spawn(self):
		if self._random_path:
			# randomize what level we sample from
			level = self.level
			if self.level_proba < 1 and level > self.min_level and random.random() >= self.level_proba:
				level = random.randint(self.min_level, level-1)
			# sample path from given level
			path_idx = random.choice(self._path_bins[level])
		else:
			# get next path in static rotating list
			self.rotating_idx = self.rotating_idx+1 if self.rotating_idx+1 < self._n_split_paths else 0  
			path_idx = self._path_idxs[self.rotating_idx]
		
		self._path_idx = path_idx	
		self._path = self._paths[path_idx]
		self._start = self._starts[path_idx]
		self._goal = self._goals[path_idx]
		self._astar_length = len(self._path)-1 # ignore spawn
		self._euclidean = self._euclideans[path_idx]
		self._nturn = self._nturns[path_idx]
		self._level = self._levels[path_idx]
		self._start_x, self._start_y, self._start_z = [int(d) for d in self._start]
		self._goal_x, self._goal_y, self._goal_z = [int(d) for d in self._goal]
		
		# if self.yaw_type in ['face']:
		# 	distance_vector = np.array([self._goal_x, self._goal_y, self._goal_z]) - np.array([self._start_x, self._start_y, self._start_z])
		# 	self._start_yaw = math.atan2(distance_vector[1], distance_vector[0])
		# elif self.yaw_type in ['random']:
		# 	self._start_yaw = np.random.uniform(-1*math.pi, math.pi)
		# else:
		self._start_direction = self.start_direction
		
		#print('spawn drone at', [self._start_x, self._start_y, self._start_z, self._start_direction])
		self._drone.teleport(self._start_x, self._start_y, self._start_z, self._start_direction, ignore_collision=True, check_bounds=False, stabelize=True)
		
	def reset(self, state=None):
		self.spawn()
		if state is not None:
			state['path_idx'] = self._path_idx
			state['astar_length'] = self._astar_length
			state['euclidean'] = self._euclidean
			state['nturn'] = self._nturn
			state['level'] = self._level
