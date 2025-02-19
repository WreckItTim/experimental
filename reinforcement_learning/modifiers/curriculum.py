from modifiers.modifier import Modifier
from component import _init_wrapper
import rl_utils as _utils
import numpy as np

# checks against percent of training episodes resulting in success
class Curriculum(Modifier):
	@_init_wrapper
	def __init__(self,
				base_component, # componet with method to modify
				parent_method, # name of parent method to modify
				order, # modify 'pre' or 'post'?
				spawner_component, # which component to level up
				start_level,
				max_level,
				level_up_criteria=0.9, # percent of succesfull paths needed to level up
				level_up_buffer=100, # number of previous episodes to look at to measure level_up_criteria
				terminate_on_max = False,
				eval_num_per_level = 10, # number of evaluation paths to run at each sublevel
				level = None,
				track_record = [],
				track_idx = 0,
				reached_max = False,
				frequency = 1, # use modifiation after how many calls to parent method?
				eval_on_levelup = False, # save a model at each new level up and evaluate
				model_component = None, # model being trained (for saving purposes, otherwise can set to None)
				update_progress = False,
				counter = 0, # keepts track of number of calls to parent method
			): 
		super().__init__(base_component, parent_method, order, frequency, counter)
		self.connect_priority = -1 # goal component needs to connect first to get starting goal range
		if level is None:
			self.level = start_level

	def connect(self):
		super().connect()
		if len(self.track_record) == 0:
			self.reset_track()
		self._project_name = _utils.get_local_parameter('project_name')
		self._experiment_name = _utils.get_local_parameter('experiment_name')
		self._trial_name = _utils.get_local_parameter('trial_name')
		self._run_name = _utils.get_local_parameter('run_name')
		self._job_name = _utils.get_local_parameter('job_name')
		self.progress()

	# reset learning loop 
	def reset_learning(self):
		self.reset_level()
		self.reset_track()

	# modifier activate()	
	def activate(self, state):
		if self.check_counter(state):
			self.update_track(state)
			if self.evaluate(state):
				if self.terminate_on_max:
					self.terminate()

	def disconnect(self, state=None):
		super().disconnect()
		self.terminate()

	# evaluates all episodes for this next set
	def terminate(self):
		self._configuration.controller.stop()

	# evaluates all episodes for this next set
	def evaluate(self, state):
		if self.check_levelup(state):
			return self.level_up()
		return False

	def progress(self):
		if self.update_progress:
			_utils.progress(self._job_name, f'level {self.level} episodes {self.counter}')

	## helper functions
	def update_track(self, state):
		self.track_record[self.track_idx] = 0 # overwrite to 0 to assum failure
		if state['reached_goal'] == True: 
			self.track_record[self.track_idx] = 1 # overwrite to 1 after observing success
		self.increment_track()
	def increment_track(self):
		self.track_idx = self.track_idx+1 if self.track_idx+1 < len(self.track_record) else 0   
	def reset_track(self):
		self.track_record = [0]*self.level_up_buffer
		self.track_idx = 0
	def reset_level(self):
		self.reached_max = False
		self.level = self.start_level
		self._spawner.set_level(self.level)
	def level_up(self):
		if self.eval_on_levelup and not self.reached_max:
			model_name = f'model_level_{self.level}'
			test_name = f'test_level_{self.level}'
			working_directory = _utils.get_local_parameter('working_directory')
			model_write_path = f'{working_directory}modeling/{model_name}.zip'
			self._model.save_model(model_write_path)
			self._configuration.save()
			_utils.evaluate2(
				f'{working_directory}configuration.json',
				f'{working_directory}modeling/{model_name}.zip',
				f'{working_directory}{test_name}/',
			)
			#_utils.evaluate(f'test_level_{self.level}', self.start_level, self.level, self.eval_num_per_level, model_name)
		if self.level < self.max_level:
			self.level += 1
			self._spawner.set_level(self.level)
			self.reset_track()
			_utils.speak(f'LEVELED UP!!! from {self.level-1} to {self.level}')
			return False
		else:
			if not self.reached_max:
				_utils.speak(f'max level reached, {self.level}')
			self.reached_max = True
			return True
	def check_levelup(self, state):
		percent_success = float(np.mean(self.track_record))
		termination_reason = state['termination_reason']
		if self.counter%1_000 == 0:
			self.progress()
			_utils.speak(f'job:{self._job_name} level:{self.level} episode:{self.counter} track:{percent_success} end:{termination_reason} ')
		return percent_success >= self.level_up_criteria
