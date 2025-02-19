from controllers.controller import Controller
from component import _init_wrapper
import numpy as np
import math
import rl_utils as _utils
import os
import pickle
import time
import msgpackrpc

# collects data by taking queries from file
class Host(Controller):
	# constructor
	@_init_wrapper
	def __init__(self,
				 drone_component, # agent to move around map
				 map_component, # map to handle crash with
				 map_name, # name where to store observaitons
				 read_directory, # directory path to read queries from
				 write_directory # directory path to write observation to
				 ):
		super().__init__()
		os.makedirs(read_directory, exist_ok=True)
		os.makedirs(write_directory, exist_ok=True)
	
	def set_state(self, state_name, state_type):
		if state_type in ['4vec']:
			x, y, z, yaw = [int(part) for part in state_name.split('_')]
			yaw *= math.pi/2
			self._drone.teleport(x, y, z, yaw, ignore_collision=True)
		
	# runs control on components
	def run(self):
		# constantly check for new queries
		while(True):
			time.sleep(0.05)
			
			# read query files
			query_files = [file for file in os.listdir(self.read_directory) if file[:6] == 'query_']
			
			# if any queries are detected then translate and execute
			if len(query_files) > 0:
				for q_idx, query_file in enumerate(query_files):
							
					# get query from file name
					query = query_file.split('.')[0].replace('query__', '')
					parts = query.split('__')
					
					# get sensor name from query
					sensor_name = parts[0]
					sensor_path = self.write_directory + sensor_name + '/' + self.map_name + '/'
					os.makedirs(sensor_path, exist_ok=True)
				
					# get state values from query
					state_name = parts[1] # unique identifier for state
					state_type = parts[2] # name of instructions on how to interpret state_name
					
					# check observation path
					observation_path = sensor_path + state_name + '.p'
								
					# get sensor to query
					sensor = self._configuration.get_component(sensor_name)
					
					# query to airsim
					while True: # crash handler
						try: # crash handler
					
							# set state
							self.set_state(state_name, state_type)
							
							# make observation
							observation = sensor.step()
							
							# success without crash
							break
						
						except msgpackrpc.error.TimeoutError as e: # crash handler
							_utils.speak(f'*** host crashed **') # crash handler
							self._map.connect(from_crash=True) # crash handler
							_utils.speak('*** host recovered **') # crash handler
							_utils.print_local_log() # crash handler
						
					# write observation as numpy array (pickle may lead to future dependency issues)
					pickle.dump(observation.to_numpy(), open(observation_path, 'wb'))
					print(f'written observation to {observation_path}')
					
					# remove query from queue
					os.remove(self.read_directory + query_file)
