from controllers.controller import Controller
from component import _init_wrapper
import utils.global_methods as gm
import numpy as np

# simply runs an evaluation set on the given evaluation environment
class Test(Controller):
	# constructor
	@_init_wrapper
	def __init__(self,
				environment_component, # environment to run eval in
				model_component, # used to make predictions
				spawner_component, # used to spawn drone based on astar paths
				results_directory=None,
				ckpt_freq=1000,
				job_name=None,
				 ):
		super().__init__()

	# runs control on components
	def run(self):
		#self._configuration.reset_all()
		return self.evaluate_set()

	def connect(self):
		super().connect()

	# steps through one evaluation episode
	def evaluate_episode(self):
		# reset environment, returning first observation
		observation_data, initial_state = self._environment.reset()
		# start episode
		done = False
		rewards = []
		gamma = 0.99
		while(not done):
			rl_output = self._model.predict(observation_data)
			# take next step
			observation_data, reward, done, truncated, state = self._environment.step(rl_output)
			rewards.append(reward)
		q = 0
		for i, reward in enumerate(rewards):
			q += reward * gamma**(len(rewards)-i-1)
		length = (len(rewards)) / initial_state['astar_length']
		# end of episode
		if 'reached_goal' not in state or 'termination_reason' not in state:
			return False, 'unknown', q, np.nan
		return state['reached_goal'], state['termination_reason'], q, length

	# evaluates all episodes for this next set
	def evaluate_set(self):
		# loop through all episodes
		successes = []
		lengths = []
		termination_reasons = []
		qs = []
		episode_idx = 0
		#self._spawner.reset_learning()
		#print('eval set', self._spawner.split_name, self._spawner._n_split_paths)
		while(episode_idx < self._spawner._n_split_paths):
			# step through next episode
			success, termination_reason, q, length = self.evaluate_episode()
			#gm.speak(f'episode:{episode} goal:{success} q:{q} termination:{termination_reason}')
			successes.append(success)
			lengths.append(length)
			termination_reasons.append(termination_reason)
			qs.append(q)
			if episode_idx % self.ckpt_freq == 0:
				perc_done = 100*episode_idx/self._spawner._n_split_paths
				#gm.speak(f'evaluation percent done: {perc_done:.2f}%')
				if self.job_name is not None: 
					gm.progress(self.job_name, f'{perc_done:.2f}%')
			episode_idx += 1
		results_dic = {
			'successes':successes,
			'lengths':lengths,
			'termination_reasons':termination_reasons,
			'qs':qs,
		}
		if self.results_directory is not None:
			results_path = self.results_directory + 'evaluation.json'
			gm.write_json(results_dic, results_path)
		return results_dic

