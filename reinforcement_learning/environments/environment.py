# abstract class used to handle observations to input into rl algo
from component import Component
import gymnasium
import numpy as np
import utils.global_methods as gm

# OpenAI Gym enviornment needed to run Stable_Baselines3
class Environment(gymnasium.Env, Component):
	@staticmethod
	def show_state(state):
		action = state['transcribed_action']
		x = int(state['drone_position'][0])
		y = int(state['drone_position'][1])
		z = int(state['drone_position'][2])
		distance = int(state['distance'])
		episode = state['nEpisodes']
		step = state['nSteps']
		total_reward = round(state['total_reward'], 4)
		rewards = []
		for key in state:
			if 'reward_from_' in key:
				rewards.append(round(state[key], 4))
		print(f'episode:{episode} step:{step} action:{action}  position:({x},{y},{z})  distance:{distance}  total_reward:{total_reward}  rewards:{rewards}') 

	# constructor
	def __init__(self):
		self.connect_priority = -3 # order: sensors, observers, environment, model

	def connect(self, state=None):
		super().connect()

	## methods that are expected to be defined and called from OpenAI Gym and Stable_Baselines3

	# step called after observation and input action to take
	# take action then create next state to progress for next step
	# must return observation, reward, done, info
		# observation - input to rl model after taken action 
		# reward - calcuated reward at state after taken action
		# done - True or False if the episode is done or not 
		# info - auxilary diction of info for whatever
		
	def step(self, rl_output):
		raise NotImplementedError

	# called at begin of episode to prepare for next, when step() returns done=True
	# returns first observation for new episode
	def reset(self):
		raise NotImplementedError

	# called at end of episode to prepare for next, when step() returns done=True
	# returns first observation for new episode
	def end(self):
		raise NotImplementedError

	# Crashes happen -- this will undo a step when triggered and reconnect to the map
		# this is especially needed for AirSim which is particulary unstable 
	def handle_crash(self):
		self._map.connect(from_crash=True)
		#gm.print_local_log()

	# just makes the rl_output from SB3 more readible
	def clean_rl_output(self, rl_output):
		if np.issubdtype(rl_output.dtype, np.integer):
			return int(rl_output)
		if np.issubdtype(rl_output.dtype, np.floating):
			return rl_output.astype(float).tolist()