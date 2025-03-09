# takes an image on each observation
from sensors.sensor import Sensor
from observations.image import Image
import numpy as np
import cv2
from cv2 import VideoCapture
from component import _init_wrapper
import global_methods as md

# see https://microsoft.github.io/AirSim/image_apis/
class PortCamera(Sensor):
	# constructor
	@_init_wrapper
	def __init__(self, 
			  port='udp://0.0.0.0:11111', 
			  is_gray=False,
			  write=True,
			  transformers_components=None,
			  offline = False,
			  out_size = (255,144),
			  memory='all', # none all part
			  ):
		super().__init__(offline, memory)
		self._camera = None
		
	def create_obj(self, data):
		observation = Image(
			_data = data, 
			is_gray=self.is_gray,
		)
		return observation
	
	def get_null(self):
		null_data = np.zeros(self.out_size).atype(np.uint8)
		return self.create_obj(null_data)

	def connect(self):
		super().connect()
		self._camera = VideoCapture(self.port)

	def disconnect(self):
		if self._camera is not None:
			self._camera.release()

	# takes a picture with camera
	def step(self, state=None):
		ret = False
		while not ret:
			ret, img_array = self._camera.read()
		observation = self.create_obj(img_array)
		cv2.imwrite(md.get_global_parameter('working_directory') + 'tello_imgs/' + observation._name + '_pre.png', img_array)
		transformed = self.transform(observation)
		return transformed
