# abstract class used to handle RL model
from sb3models.sb3model import SB3Model
from stable_baselines3 import SAC as sb3SAC
from component import _init_wrapper

class SAC(SB3Model):
	# constructor
	@_init_wrapper
	def __init__(self,
			environment_component,
			policy = 'MlpPolicy',
			learning_rate = 1e-3,
			buffer_size = 1_000_000,
			learning_starts = 100,
			batch_size = 256,
			tau = 0.005,
			gamma = 0.99,
			train_freq = 1,
			gradient_steps = 1,
			action_noise = None,
			replay_buffer_class = None,
			replay_buffer_kwargs = None,
			optimize_memory_usage = False,
			ent_coef = "auto",
			target_update_interval = 1,
			target_entropy = "auto",
			use_sde = False,
			sde_sample_freq = -1,
			use_sde_at_warmup = False,
			tensorboard_log = None,
			policy_kwargs = None,
			verbose = 0,
			seed = None,
			device ='cpu',
			init_setup_model = True,
			read_model_path=None, 
			read_replay_buffer_path=None, 
			read_weights_path=None, 
			use_slim = False,
			convert_slim = False,
		):
		kwargs = locals()
		_model_arguments = {key:kwargs[key] for key in kwargs.keys() if key not in [
			'self', 
			'__class__',
			'environment_component',
			'init_setup_model',
			'read_model_path',
			'read_replay_buffer_path',
			'read_weights_path',
			'use_slim',
			'convert_slim',
			]}
		_model_arguments['_init_setup_model'] = kwargs['init_setup_model']
		self.sb3Type = sb3SAC
		self.sb3Load = sb3SAC.load
		self._has_replay_buffer = True
		super().__init__(
				   read_model_path=read_model_path, 
				   read_replay_buffer_path=read_replay_buffer_path, 
				   read_weights_path=read_weights_path, 
				   _model_arguments=_model_arguments,
				   use_slim = use_slim,
				   convert_slim = convert_slim,
				   )