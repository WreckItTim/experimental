from component import Component
import utils.global_methods as gm


class Spawner(Component):
	# constructor
	def __init__(self):
		pass

	# set start and target location on reset()
	def reset(self, state=None):
		raise NotImplementedError

	# get x,y,z,yaw of last spawned starting location
	def get_start(self):
		return [self._start_x, self._start_y, self._start_z]

	# get x,y,z of last spawned target location
	def set_goal(self, goal):
		self._goal_x = goal[0]
		self._goal_y = goal[1]
		self._goal_z = goal[2]

	# get x,y,z of last spawned target location
	def get_goal(self):
		return [self._goal_x, self._goal_y, self._goal_z]

	# this is used some times in arbitrary components
	def get_position(self):
		return self.get_goal()

	def get_yaw(self):
		return self._start_yaw

	def get_direction(self):
		return self._start_direction

	def connect(self):
		super().connect()