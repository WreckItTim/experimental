from modifiers.modifier import Modifier
from component import _init_wrapper
import utils.global_methods as gm
import reinforcement_learning.reinforcement_methods as rm
import numpy as np

# checks against percent of training episodes resulting in success
class Curriculum(Modifier):
	@_init_wrapper
	def __init__(self,
				base_component, # Modifier param, componet with method to modify
				parent_method, # Modifier param, name of parent method to modify
				order, # Modifier param, modify 'pre' or 'post'?
				spawner_component, # which component to level up
				model_component, # model being trained
				min_level, # min curric level 
				max_level, # max curric level
				frequency = 1, # Modifier param, use modifiation after how many calls to parent method -- KEEP AT 1
				counter = 0, # Modifier param, keepts track of number of calls to parent method
				level = None, # current level in curric learning, if None then will start at min_level
				update_progress = False, # write progress to file (given global progress_path)
				# specific parameters for how we track curriculum learning
				reached_max = False, # we have leveled up to the maximum level if True
				terminate_on_max = False, # when reach max level, stop learning?
				eval_frequencies = {}, # evalute the given path_split at split_name every these many episodes
											# key names correspond to spawner.path_splits
											# {'split_name' : eval_freq}
				eval_use_current_as_min_level = {}, # when evaluating at given split_name, is the min_level = self.min_level or self.level
				 										# keep same indexes as eval_frequencies keys
														# default if not in dictionary is min_level = self.min_level
				eval_use_current_as_max_level = {}, # when evaluating at given split_name, is the max_level = self.max_level or self.level
				 										# keep same indexes as eval_frequencies keys
														# default if not in dictionary is max_level = self.max_level
				eval_accuracies = {}, # keep track of learning curve accuracies indexed by eval_frequencies keys
				# if using the track_record to level up curric
				use_track = False, # True will level up when 
				track_freq = 1, # every this many episodes, run a check to see if we level up
				track_record = [], # keeps track of results from recent training episodes (0 for fail 1 for success)
				track_idx = 0, # current rotating index of track_record
				track_length = 100, # number of training episode results to store in track_record
				track_criteria = 0.9, # percent of succesfull paths in track_record needed to level up
				# if using early_stopping to level up curric
				use_early = False, # True will use early stopping to level up based on some evaluation set if wait > patience
				early_freq = 1_000, # every this many episodes, run a check to see if we level up
				early_split_name = 'val', # which eval_accuracies key value to use to trigger early stopping
				early_criteria = 0.8, # wait till accuracy hits this much before check for early stopping
				early_start = False, # this is controlled inside of learning keep at False unless you do not want a burn in period
				patience = 10, # number of epochs to wait before triggering early stopping
				epsilon = 0.01, # min improvement required inbetween epochs to reset wait counter
				wait = 0, # counter to keep track of number of epoch that have not improved by epsilon
				best_accuracy = 0, # keep track of the best accuracy for comparing improvements and logging models
			): 
		super().__init__(base_component, parent_method, order, frequency, counter)
		self.connect_priority = -1 # goal component needs to connect first to get starting goal range
		if level is None:
			self.level = min_level
		
	def connect(self, state=None):
		super().connect(state)
		self._job_name = gm.get_global('job_name')

	def disconnect(self, state=None):
		super().disconnect(state)

	# reset entire learning loop 
	def reset_learning(self):
		self.reset_level()
		self.reset_criterias()

	# reset track, or early stopping, or whateber it is used as criteria for leveling up
	def reset_criterias(self):
		if self.use_track:
			self.reset_track()
		if self.use_early:
			self.reset_early()

	# modifier activate()	
	def activate(self, state):
		if self.check_counter(state): # super Modifier class call (do we check level up?)
			if self.counter <= 1:
				self._stopwatch = gm.Stopwatch()
			# run external evaluation script?
			output_str = f'episodes-{self.counter}'
			evald = False
			stopwatch = gm.Stopwatch()
			for split_name in self.eval_frequencies:
				if self.counter%self.eval_frequencies[split_name] == 0:
					evald = True
					min_level = self.min_level
					if split_name in self.eval_use_current_as_min_level and self.eval_use_current_as_min_level[split_name]:
						min_level = self.level
					max_level = self.max_level
					if split_name in self.eval_use_current_as_max_level and self.eval_use_current_as_max_level[split_name]:
						max_level = self.level
					accuracy = self.eval_accuracy(f'eval_{split_name}', 'model_eval', min_level, max_level, split_name)
					if split_name not in self.eval_accuracies:
						self.eval_accuracies[split_name] = []
					self.eval_accuracies[split_name].append(accuracy)
					output_str += f' {split_name}-{100*accuracy:0.2f}%'
			if evald:
				eval_time = stopwatch.stop()
				train_time = self._stopwatch.stop() - eval_time
				output_str += f' eval_time-{eval_time/60:0.2f}min train_time-{train_time/60:0.2f}min'
				gm.speak(output_str)
				self._stopwatch = gm.Stopwatch()
			# update criterias?
			if self.use_track:
				self.update_track(state)
			# do we level up?
			if self.check_levelup(state):
				self.level_up()
				if self.reached_max and self.terminate_on_max:
					self.terminate_learning()
			# write progress to file?
			if self.update_progress:
				self.progress()

	# stop all learning algorithm processes / stops controller 
	def terminate_learning(self):
		self._configuration.controller.stop()

	def progress(self, acc_name='val'):
		progress_string = f'level {self.level} episodes {self.counter}'
		for split_name in self.eval_accuracies:
			last_accuracy = self.eval_accuracies[split_name][-1]
			progress_string += f' {split_name} {last_accuracy:.2f}'
		gm.progress(self._job_name, progress_string)

	def eval_accuracy(self, test_name, model_name, min_level, max_level, split_name):
		self.save_model(model_name)
		output_dir = gm.get_global('output_dir')
		self._configuration.save()
		accuracy = rm.evaluate_navi({
			'config_path':f'{output_dir}configuration.json',
			'model_path':f'{output_dir}modeling/{model_name}.zip',
			'output_dir':f'{output_dir}{test_name}/',
			'device':gm.get_global('device'),
			'min_level':min_level,
			'max_level':max_level,
			'split_name':split_name,
		})
		return accuracy

	def save_model(self, model_name):
		output_dir = gm.get_global('output_dir')
		model_write_path = f'{output_dir}modeling/{model_name}.zip'
		self._model.save_model(model_write_path)

	## helper functions
	def update_track(self, state):
		self.track_record[self.track_idx] = 0 # overwrite to 0 to assume failure
		if state['reached_goal'] == True: 
			self.track_record[self.track_idx] = 1 # overwrite to 1 after observing success
		self.track_idx = self.track_idx+1 if self.track_idx+1 < len(self.track_record) else 0 
	def reset_track(self):
		self.track_record = [0]*self.track_length
		self.track_idx = 0
	def reset_early(self):
		self.wait = 0
		self.best_accuracy = 0
		self.early_start = False
	def reset_level(self):
		self.reached_max = False
		self.level = self.min_level
		self._spawner.set_level(self.level)
	def level_up(self):
		if self.level < self.max_level:
			self.level += 1
			self._spawner.set_level(self.level)
			self.reset_criterias()
			gm.speak(f'LEVELED UP!!! from {self.level-1} to {self.level}')
		else:
			if not self.reached_max:
				gm.speak(f'max level reached, {self.level}')
			self.reached_max = True
	def check_levelup(self, state):
		go_level_up = False
		if self.use_track and self.counter%self.track_freq==0:
			percent_success = float(np.mean(self.track_record))
			go_level_up = percent_success >= self.track_criteria
		if self.use_early and self.counter%self.early_freq==0:
			if len(self.eval_accuracies[self.early_split_name]) == 0:
				return
			last_accuracy = self.eval_accuracies[self.early_split_name][-1]
			if not self.early_start:
				if last_accuracy >= self.early_criteria:
					self.early_start = True
					self.best_accuracy = last_accuracy
			else:
				if last_accuracy > self.best_accuracy + self.epsilon:
					self.best_accuracy = last_accuracy
					self.wait = 0
					self.save_model('model_best')
				else:
					self.wait += 1
					if self.wait > self.patience:
						go_level_up = True
		return go_level_up
