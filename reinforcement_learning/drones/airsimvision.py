# drone launched through airsim api - requires an airsim relase map
import setup_path # need this in same directory as python code for airsim
import airsim
import utils.global_methods as gm
from drones.drone import Drone
from component import _init_wrapper
import math
import numpy as np
import time

class AirSimVision(Drone):
	@_init_wrapper
	def __init__(self,
			  airsim_component,
			  handle_crashes=True,
			):
		super().__init__()
		self._timeout = 10 # number of seconds to wait for communication

	# resets on episode
	def reset(self, state=None):
		self._airsim._client.confirmConnection()

	# teleports to position, yaw in radians
	def teleport(self, x, y, z, yaw, ignore_collision=True, stabelize=True):
		while yaw > math.pi:
			yaw -= 2*math.pi
		while yaw <= -1*math.pi:
			yaw += 2*math.pi
		camera_pose = airsim.Pose(airsim.Vector3r(x, y, z), airsim.to_quaternion(0, 0, yaw)) #radians
		self._airsim._client.simSetCameraPose("0", camera_pose)
