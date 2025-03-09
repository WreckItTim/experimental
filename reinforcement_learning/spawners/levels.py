from spawners.spawner import Spawner
from component import _init_wrapper
import random
import numpy as np
import math
import global_methods as md
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
				drone_component,
				levels_path,
				start_level,
				max_level,
				random_path = False,
				yaw_type = 0, # 'face': faces goal, 'random': random full range, value: specific yaw
				rotating_idx = 0,
				paths_per_sublevel=1,
				level = None,
			):
		super().__init__()


	def connect(self):
		super().connect()
		self._levels = md.pk_read(self.levels_path)
		self._paths = self._levels['paths']
		self._nonlinearitys = self._levels['nonlinearitys']
		self._linearitys = self._levels['linearitys']
		self._sublevels = self._levels['levels']
		self._idx_levels = self._levels['idx_levels']
		self._idx_sublevels = self._levels['idx_sublevels']
		if self.level is None:
			self.level = self.start_level
		self.set_level(self.level)
		if not self.random_path:
			self._static_path_idxs = []
			for level in range(self.start_level, self.max_level+1):
				for sublevel in self._levels['levels'][level]:
					for i in range(self.paths_per_sublevel):
						if i < len(sublevel):
							self._static_path_idxs.append(sublevel[i])

	def set_level(self, level):
		if level <= self.max_level:
			self.level = level
			self.rotating_idx = 0
			self.reset_subs()

	def reset_subs(self):
		self._avaiable_subs = [i for i in range(len(self._sublevels[self.level]))]
		
	def reset_rotating(self):
		self.rotating_idx = 0

	def reset_learning(self):
		if self.random_path:
			self.reset_subs()
		else:
			self.reset_rotating()

	def spawn(self):
		if self.random_path:
			sublevel_idx = random.choice(self._avaiable_subs)
			del self._avaiable_subs[self._avaiable_subs.index(sublevel_idx)]
			if len(self._avaiable_subs) == 0:
				self.reset_subs()
			path_idx = random.choice(self._sublevels[self.level][sublevel_idx])
		else:
			path_idx = self._static_path_idxs[self.rotating_idx]
			self.rotating_idx = self.rotating_idx+1 if self.rotating_idx+1 < len(self._static_path_idxs) else 0   
		
		self._path_idx = path_idx	
		self._path = self._paths[path_idx]
		self._astar_length = len(self._path)
		self._nonlinearity = self._nonlinearitys[path_idx]
		self._linearity = self._linearitys[path_idx]
		self._level = self._idx_levels[path_idx]
		self._sublevel = self._idx_sublevels[path_idx]
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
			state['astar_length'] = self._astar_length
			state['path_linearity'] = self._linearity
			state['path_nonlinearity'] = self._nonlinearity
			state['level'] = self._nonlinearity
			state['sublevel'] = self._nonlinearity
			state['path_idx'] = self._path_idx
