from modifiers.modifier import Modifier
from component import _init_wrapper
import utils.global_methods as gm
import reinforcement_learning.reinforcement_methods as rm
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

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
				eval_controller_component,
				frequency = 1, # Modifier param, use modifiation after how many calls to parent method -- KEEP AT 1
				counter = 0, # Modifier param, keepts track of number of calls to parent method
				level = None, # current level in curric learning, if None then will start at min_level
				update_progress = False, # write progress to file (given global progress_path)
				# specific parameters for how we track curriculum learning
				reached_max = True, # we have leveled up to the maximum level if True
				terminate_on_max = True, # when reach max level, stop learning?
				eval_frequencies = {}, # evalute the given path_split at split_name every these many episodes
											# key names correspond to spawner.path_splits
											# {'split_name' : eval_freq}
				eval_accuracies = {}, # keep track of learning curve accuracies indexed by eval_frequencies keys
				eval_lengths = {}, # keep track of learning curve accuracies indexed by eval_frequencies keys
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
				early_acc = 0,
				early_best = 0,
				patience = 10, # number of epochs to wait before triggering early stopping
				wait = 0, # counter to keep track of number of epoch that have not improved
				best_test = 0, # just for funsies
				view_progress = False,
				eval_on_levelup = False,
			): 
		super().__init__(base_component, parent_method, order, frequency, counter)
		self.connect_priority = -777 # other components needs to connect first
		if level is None:
			self.level = min_level
		self._fig, self._axs = plt.subplots(ncols=2, figsize=(8,4))
		
	def connect(self, state=None):
		super().connect(state)
		self._job_name = gm.get_global('job_name')
		self._stopwatch = gm.Stopwatch()

	def disconnect(self, state=None):
		super().disconnect(state)

	# reset entire learning loop 
	def reset_learning(self):
		self.reset_level()
		self.reset_criterias()
		self.best_test = 0

	# reset track, or early stopping, or whateber it is used as criteria for leveling up
	def reset_criterias(self):
		if self.use_track:
			self.reset_track()
		if self.use_early:
			self.reset_early()

	def plot_learning(self):
		clear_output(wait = True)
		self._axs[0].cla()
		self._axs[1].cla()
		for split_name in self.eval_accuracies:
			self._axs[0].plot(self.eval_accuracies[split_name], label=split_name)
			self._axs[1].plot(self.eval_lengths[split_name], label=split_name)
		self._axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
		self._axs[0].set_xlabel('Epoch')
		self._axs[0].set_ylabel('Navi Accuracy')
		self._axs[1].set_xlabel('Epoch')
		self._axs[1].set_ylabel('Path Length')
		plt.tight_layout()
		display(self._fig)

	# modifier activate()	
	def activate(self, state):
		if self.check_counter(state): # super Modifier class call (do we check level up?)
			# run external evaluation script?
			if self.counter%self.early_freq == 0:
				stopwatch = gm.Stopwatch()
				accuracy, length = self.eval_accuracy('val', use_current=True)
				self.early_acc = accuracy
				self.early_len = length
				output_str = f'lvl-{self.level} eps-{self.counter} Test-{self.best_test:0.2f} early-{self.early_best:0.2f} chk-{self.early_start} wait-{self.wait}'
				evald = False
				for split_name in self.eval_frequencies:
					if self.counter%self.eval_frequencies[split_name] == 0 or self.counter == 1:
						evald = True
						accuracy, length = self.eval_accuracy(split_name)
						if split_name not in self.eval_accuracies:
							self.eval_accuracies[split_name] = []
							self.eval_lengths[split_name] = []
						self.eval_accuracies[split_name].append(accuracy)
						self.eval_lengths[split_name].append(length)
						output_str += f' {split_name}-{100*accuracy:0.2f}%'
						# write progress to file?
						if self.update_progress:
							if split_name == 'test':
								self.best_test = max(self.best_test, accuracy)
							self.save_model('model_eval')
							self._configuration.save()
				if evald:
					eval_time = stopwatch.stop()
					train_time = self._stopwatch.stop() - eval_time
					output_str += f' eval_time-{eval_time/60:0.2f}min train_time-{train_time/60:0.2f}min'
					if self.view_progress:
						self.plot_learning()
					self._stopwatch = gm.Stopwatch()
				gm.speak(output_str)
				gm.progress(self._job_name, output_str)
				# update criterias?
				if self.use_track:
					self.update_track(state)
				# do we level up?
				if self.check_levelup(state):
					self.level_up()
					if self.reached_max and self.terminate_on_max:
						self.terminate_learning()

	# stop all learning algorithm processes / stops controller 
	def terminate_learning(self):
		self._configuration.controller.stop()

	def eval_accuracy(self, split_name, use_current=False):
		# set spawner to given set
		random_before = self._spawner.random_splits[split_name]
		current_before = self._spawner.current_splits[split_name]
		self._spawner.random_splits[split_name] = False
		self._spawner.current_splits[split_name] = use_current
		self._spawner.set_active_split(split_name)
		# evaluate
		evaluation = self._eval_controller.run()
		accuracy = float(np.mean(evaluation['successes']))
		lengths = []
		for i in range(len(evaluation['successes'])):
			success = evaluation['successes'][i]
			if success:
				length = evaluation['lengths'][i]
				lengths.append(length)
		length = float(np.mean(lengths))
		# set spawner back
		self._spawner.random_splits[split_name] = random_before
		self._spawner.current_splits[split_name] = current_before
		self._spawner.set_active_split('train')
		return accuracy, length

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
		self.early_acc = 0
		self.early_best = 0
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
			if self.eval_on_levelup:
				stopwatch = gm.Stopwatch()
				output_str = f'lvl-{self.level} eps-{self.counter} Test-{self.best_test:0.2f} early-{self.early_best:0.2f} chk-{self.early_start} wait-{self.wait}'
				for split_name in ['train', 'val', 'test']:
					accuracy, length = self.eval_accuracy(split_name)
					if split_name not in self.eval_accuracies:
						self.eval_accuracies[split_name] = []
						self.eval_lengths[split_name] = []
					self.eval_accuracies[split_name].append(accuracy)
					self.eval_lengths[split_name].append(length)
					output_str += f' {split_name}-{100*accuracy:0.2f}%'
					# write progress to file?
					if self.update_progress:
						if split_name == 'test':
							self.best_test = max(self.best_test, accuracy)
						self.save_model('model_eval')
						self._configuration.save()
				eval_time = stopwatch.stop()
				train_time = self._stopwatch.stop() - eval_time
				output_str += f' eval_time-{eval_time/60:0.2f}min train_time-{train_time/60:0.2f}min'
				if self.view_progress:
					self.plot_learning()
				self._stopwatch = gm.Stopwatch()
				gm.speak(output_str)
				gm.progress(self._job_name, output_str)
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
			if self.early_acc > self.early_best:
				self.early_best = self.early_acc
				self.wait = 0
				self.save_model('model_best')
			if not self.early_start:
				if self.early_best >= self.early_criteria:
					self.early_start = True
			else:
				self.wait += 1
				if self.wait > self.patience:
					go_level_up = True
		return go_level_up
