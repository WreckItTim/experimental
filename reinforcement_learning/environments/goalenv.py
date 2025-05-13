from environments.environment import Environment
from component import _init_wrapper
import numpy as np
import utils.global_methods as gm
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from IPython.display import clear_output

# an environment is the heart of RL algorithms
# the Goal flavor wants the drone to go to Point A to Point B
# steps until termination then resets
# if a saver modifier is used, will write observations and states
class GoalEnv(Environment):
	# even though we do not render, this field is necesary for sb3
	metadata = {"render.modes": ["rgb_array"]}

	# constructor
	# if continuing training, step_counter and episode_counter will likely be > 0
	@_init_wrapper
	def __init__(self, 
				drone_component, 
				actor_component, 
				observer_component, 
				rewarder_component, 
				spawner_component,
				model_component,
				map_component,
				others_components=None,
				step_counter=0, 
				episode_counter=0, 
				save_counter=0,
				crash_handler=True,
				debug=False,
			):
		super().__init__()
		self._last_observation_name = 'None'
		self._all_states = {}
		self._observations = {}
		self._track_save = False
		self._freq_changed = False
		if crash_handler:
			import msgpackrpc
			self._exception = msgpackrpc.error.TimeoutError
			self.step = self.step_with_crash
			self.reset = self.reset_with_crash
			self.end = self.end_with_crash
		else:
			self.step = self._step
			self.reset = self._reset
			self.end = self._end

	def connect(self, state=None):
		super().connect(state)
		# even though we do not directly use the observation or action space, these fields are necesary for sb3
		self.observation_space = self._observer.get_space()
		self.action_space = self._actor.get_space()
		
	# this will toggle if keep track of observations and states
	# note this is expensive, so must dump using save() from time to time
	# best route to do this is a saver modifier
	# track_vars are strings with which variables to save
	def set_save(self,
			  track_save,
			  track_vars=[
				  'observations', 
				  'states',
				  ],
			  ):
		self._track_save = track_save
		self._track_vars = track_vars.copy()


	# if reset learning loop
	def reset_learning(self):
		self.step_counter = 0 # total steps
		self.episode_counter = 0
		self.save_counter = 0
		self._all_states = {}
		self._observations = {}

	# save observations and states
	# these are done in chunks rather than individual files
		# this saves memory, reduces zip time, and avoids clutter
	# pass in write_folder to state
	def save(self, state):
		write_folder = state['write_folder']
		if not self._track_save:
			gm.warning('called GoalEnv.save() without setting _track_save=True, nothing to save')
			return
		part_name = 'part_' + str(self.save_counter)
		if 'states' in self._track_vars and len(self._all_states) > 0:
			path = write_folder + 'states__' + part_name + '.json'
			gm.write_json(self._all_states, path)
			self._all_states = {}
		if 'observations' in self._track_vars and len(self._observations) > 0:
			path = write_folder + 'observations__' + part_name + '.npz'
			np.savez(path, **self._observations)
			self._observations = {}
		self.save_counter += 1

	def display_obs(self, prev_obs_data, initial_state, previous_state, state):
		step = state['nSteps']
		print()
		print(f'state at timestep {step}')
		if isinstance(prev_obs_data, dict):
			img = prev_obs_data['img']
			vec = prev_obs_data['vec']
			ncols = self._nPast
			fig, axs = plt.subplots(ncols=ncols, figsize=(10, 5))
			if ncols == 1:
				axs = [axs]
			for t in range(self._nPast):
				ax = axs[t]
				ax.set_xticks([])
				ax.set_yticks([])
				ax.set_title(f'img @ t-{t}')
				ax.imshow(img[t])
			#plt.suptitle(f't = {step}')
			plt.tight_layout()
			plt.show()    
		else:
			vec = prev_obs_data
		
		j = 0
		for t in range(self._nPast):
			for i, vec_sensor in enumerate(self._vector_sensor_names):
				self._obs_df[vec_sensor][t] = round(vec[j], 4)
				j += 1
		pdf = pd.DataFrame(self._obs_df)
		pdf.index.name = 't'
		display(pdf)

		# fig, ax = plt.subplots()
		# ax.table(cellText=pdf.values, colLabels=pdf.columns, cellLoc='center', loc='center')
		# plt.show()
		
		fig, ax = plt.subplots(figsize=(4, 4))
		start_pos = initial_state['drone_position']
		goal_pos = initial_state['goal_position']
		drone_pos = previous_state['drone_position']
		direction = previous_state['direction']
		datamap = gm.get_global('datamap')
		datamap.view_map(fig, ax, x=drone_pos[0], y=drone_pos[1], z=drone_pos[2], direction=direction, 
					start=start_pos, target=goal_pos, path=self._path, show_z=False, resolution=None)
		ax.set_title(f'{start_pos} to {goal_pos}')
		plt.show()

	def display_rew(self, state):
		plt.show()    
		
		for t in range(self._nPast-1, 0, -1):
			for i, reward in enumerate(self._reward_names):
				self._rew_df[reward][t] = self._rew_df[reward][t-1]
			self._rew_df['action'][t] = self._rew_df['action'][t-1]
		for i, reward in enumerate(self._reward_names):
			if f'reward_from_{reward}' in state:
				self._rew_df[reward][0] = round(state[f'reward_from_{reward}'], 4)
			else:
				self._rew_df[reward][0] = '-'
		self._rew_df['action'][0] = state['transcribed_action']
		pdf = pd.DataFrame(self._rew_df)
		pdf.index.name = 't'
		display(pdf)

		# fig, ax = plt.subplots()
		# ax.table(cellText=pdf.values, colLabels=pdf.columns, cellLoc='center', loc='center')
		# plt.show()

	def _step(self, rl_output, state=None, return_state=True):
		# next step
		self._nSteps += 1 # total number of steps
		self.step_counter += 1 # total number of steps
		this_step = 'step_' + str(self._nSteps)
		if state is None:
			state = self._states[this_step] = {}
		else:
			state = self._states[this_step] = state.copy()
		self._states[this_step]['nSteps'] = self._nSteps
		# clean and save rl_output to state
		self._states[this_step]['rl_output'] = self.clean_rl_output(rl_output)
		if self.debug:
			self.display_obs(self._last_observation_data, self._states[f'step_0'], self._states[f'step_{self._nSteps-1}'], state)
			state['rl_output'] = int(input())
		# take action
		actor_done = self._actor.step(self._states[this_step])
		# save state kinematics
		self._states[this_step]['drone_position'] = self._drone.get_position()
		self._states[this_step]['direction'] = self._drone.get_direction()
		# get observation
		self._states[this_step]['observation_name'] = self._last_observation_name
		observation_data, observation_name = self._observer.step(self._states[this_step])
		self._last_observation_name = observation_name
		# take step for other components
		if self._others is not None:
			for other in self._others:
				other.step(self._states[this_step])
		# assign rewards (stores total rewards and individual rewards in state)
		# also checks if we are done with episode
		total_reward, rewarder_done = self._rewarder.step(self._states[this_step])
		done = actor_done or rewarder_done
		self._states[this_step]['done'] = done
		truncated = False
		if self.debug:
			self._last_observation_data = observation_data
			self._path.append(state['drone_position'])
			self.display_rew(state)
			if done:
				print('terminated', state['termination_reason'])
				input()
				clear_output()
		# save data?
		if self._track_save and 'observations' in self._track_vars:
			self._observations[observation_name] = observation_data.copy()
		if self._track_save and 'states' in self._track_vars and done: 
			self._all_states['episode_' + str(self.episode_counter)] = self._states.copy()
		if done: 
			self.end(self._states[this_step])
			if 'truncated' in self._states[this_step] and self._states[this_step]['truncated']:
				truncated = True
		if return_state:
			return observation_data, total_reward, done, truncated, self._states[this_step].copy()
		return observation_data, total_reward, done, truncated, this_step
		
	# activate needed components
	def step_with_crash(self, rl_output, state=None):
		try:
			observation_data, total_reward, done, truncated, this_step = self._step(rl_output, state, return_state=False)
			return_state = self._states[this_step].copy()
		except self._exception as e:
			gm.speak(f'*** crashed ** step {self._nSteps}')
			self.handle_crash()
			gm.speak('*** recovered **')
			# this is a hot fix, until SB3 has a way to remove erroneous steps
			# however in the grand scheme this one step on the replay buffer shouldnt have such a drastic affect
			done = True # finish this episode
			truncated = True
			total_reward = 0 # 0 reward
			observation_data = self._observer.null_data # fill with erroneous data (Zeros)
			return_state = {}
		# state is passed to stable-baselines3 callbacks
		### return observation_data, total_reward, done, self._states[this_step].copy()
		return observation_data, total_reward, done, truncated, return_state

	def _reset(self, state=None, seed=None, increment=True, return_state=True):
		if increment:
			self.episode_counter += 1
		# init state(s)
		self._nSteps = 0 # steps this episode
		this_step = 'step_' + str(self._nSteps)
		if state is None:
			self._states = {this_step:{}}
		else:
			self._states = {this_step:state.copy()}
		self._states[this_step]['nSteps'] = self._nSteps

		# reset drone and goal components, several reset() methods may be blank
		# order may matter here, currently no priority queue set-up, may need later
		self._model.reset(self._states[this_step])
		self._map.reset(self._states[this_step])
		self._drone.reset(self._states[this_step])
		self._spawner.reset(self._states[this_step])
		
		# set initial state
		start_x, start_y, start_z = self._spawner.get_start()
		self._states[this_step]['drone_position'] = [start_x, start_y, start_z]
		start_direction = self._spawner.get_direction()
		self._states[this_step]['direction'] = start_direction
		goal_x, goal_y, goal_z = self._spawner.get_goal()
		self._states[this_step]['goal_position'] = [goal_x, goal_y, goal_z]

		# reset rest of components
		if self._others is not None:
			for other in self._others:
				other.reset(self._states[this_step])
		self._actor.reset(self._states[this_step])
		self._observer.reset(self._states[this_step])
		self._rewarder.reset(self._states[this_step])

		# get first observation
		observation_data, observation_name = self._observer.step(self._states[this_step])
		self._last_observation_name = observation_name

		if self.debug:
			self._last_observation_data = observation_data
			self._obs_df = {}
			for vec_sensor in self._vector_sensor_names:
				self._obs_df[vec_sensor] = [0]*self._nPast

			self._rew_df = {}
			self._rew_df['action'] = ['-']*self._nPast
			for reward in self._reward_names:
				self._rew_df[reward] = ['-']*self._nPast

			self._path = []

		
		# save data?
		if self._track_save and 'observations' in self._track_vars:
			self._observations[observation_name] = observation_data.copy()

		if return_state:
			return observation_data, self._states[this_step].copy()
		return observation_data, this_step

	# called at beginning of each episode to prepare for next
	# returns first observation for new episode
	# spawn_to will overwrite previous spawns and force spawn at that x,y,z,yaw
	def reset_with_crash(self, state=None, seed=None):
		self.episode_counter += 1
		try_again = True
		while(try_again):
			try:
				observation_data, this_step = self._reset(state, seed, increment=False, return_state=False)
				try_again = False
			except self._exception as e:
				gm.speak('*** crashed ** reset')
				self.handle_crash()
				gm.speak('*** recovered **')

		### return observation_data
		return observation_data, self._states[this_step].copy()

	def _end(self, state=None):
		# end all components
		self._model.end(state)
		self._map.end(state)
		self._drone.end(state)
		self._spawner.end(state)
		if self._others is not None:
			for other in self._others:
				other.end(state)
		self._actor.end(state)
		self._observer.end(state)
		self._rewarder.end(state)

	# called at the end of each episode for any clean up, when done=True
	# normally only reset() is used in OpenAI Gym environments
	# but end() is much easier/clear/necessary for modifiers and multiple envs
	# off-by-one errors aggregate when switching between multiple envs
	def end_with_crash(self, state=None):
		try_again = True
		while(try_again):
			try:
				self._end(state)
				try_again = False
			except self._exception as e:
				gm.speak('*** crashed ** end')
				self.handle_crash()
				gm.speak('*** recovered **')