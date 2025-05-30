from controllers.controller import Controller
from component import _init_wrapper

# trains a reinforcment learning algorithm
# launches learn() from the mdoel object
	# which links to the train environment
	# learn() calls step() and reset()
class Train(Controller):
	# constructor
	@_init_wrapper
	def __init__(self, 
				model_component,
				environment_component,
				continue_training = False,
				total_timesteps = 10_000_000,
				log_interval = -1,
				tb_log_name = None,
		):
		super().__init__()

	# runs control on components
	def run(self):
		# this will continually train model from same learning loop
		# meaning if you load in an old config, se this to true to not reset before training
		# this is how curriculum learning is done on other maps
		# or to pick up from a checkpoint
		# will read old model weights
		if self.continue_training:
			_num_timesteps = self._environment.step_counter
			_episode_num = self._environment.episode_counter
			self._model._sb3model.num_timesteps = _num_timesteps
			self._model._sb3model._episode_num = _episode_num
			_total_timesteps = self.total_timesteps - _num_timesteps
		# otherwise will train model from a new learning loop 
		# calls reset_learning() 
		# will use new model weights
		else:
			_total_timesteps = self.total_timesteps
			self._configuration.reset_all()
		# learn baby learn
		self._model.learn(
			total_timesteps = _total_timesteps,
			log_interval = self.log_interval,
			reset_num_timesteps = not self.continue_training,
			tb_log_name = self.tb_log_name,
			)

	def stop(self):
		self._model.stop_learning()