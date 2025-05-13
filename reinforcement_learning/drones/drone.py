from component import Component

# abstract class to handle drone
# there are several parent methods here left to define by the programmer
# not all need defined, depends on implementation
class Drone(Component):
	# constructor
	def __init__(self):
		pass

	# setup anything that needs to be done to communicate with drone
	def connect(self, state=None):
		super().connect(state)

	# clean up any resources as needed when done with communication
	def disconnect(self, state=None):
		super().disconnect(state)

	# # updates number of collisions and returns same 
	# def check_collision(self):
	# 	raise NotImplementedError

	# # take off!
	# def take_off(self):
	# 	raise NotImplementedError

	# # land!
	# def land(self):
	# 	raise NotImplementedError

	# # teleports to exact position (not this should only be defined in simulation drones, however you can define in real world drones if you have an alternative solution)
	# def teleport(self, x, y, z, yaw, ignore_collision=True):
	# 	raise NotImplementedError

	# # sets exact yaw (not this should only be defined in simulation drones, however you can define in real world drones if you have an alternative solution)
	# def set_yaw(self, degrees):
	# 	raise NotImplementedError

	# # moves to relative position at given speed (units defined within drone) 
	# def move(self, point, speed):
	# 	raise NotImplementedError

	# # moves to absolute position at given speed (units defined within drone) 
	# def move_to(self, point, speed):
	# 	raise NotImplementedError

	# # rotates along the z-axis
	# def rotate(self, degrees):
	# 	raise NotImplementedError

	# # return response from all active sensors
	# def sense(self):
	# 	raise NotImplementedError

	# # issue command to drone via string
	# def command(self):
	# 	raise NotImplementedError

	# # get current position of drone
	# def get_position(self):
	# 	raise NotImplementedError

	# # get rotation about the z-axis (yaw)
	# def get_yaw(self):
	# 	raise NotImplementedError

	# # enter hover mode
	# def hover(self):
	# 	raise NotImplementedError
		