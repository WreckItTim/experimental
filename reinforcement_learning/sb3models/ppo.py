# abstract class used to handle RL model
from sb3models.sb3model import SB3Model
from stable_baselines3 import PPO as sb3PPO
from component import _init_wrapper

class PPO(SB3Model):
	# constructor
	@_init_wrapper
	def __init__(self, 
			environment_component,
			policy = 'MlpPolicy',
			learning_rate = 1e-3,
			n_steps = 2048,
			batch_size = 64,
			n_epochs = 10,
			gamma = 0.99,
			gae_lambda = 0.95,
			clip_range = 0.2,
			clip_range_vf = None,
			normalize_advantage = True,
			ent_coef = 0.0,
			vf_coef = 0.5,
			max_grad_norm = 0.5,
			use_sde = False,
			sde_sample_freq = -1,
			target_kl = None,
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
		self.sb3Type = sb3PPO
		self.sb3Load = sb3PPO.load
		self._has_replay_buffer = False
		super().__init__(
				   read_model_path=read_model_path, 
				   read_replay_buffer_path=read_replay_buffer_path, 
				   read_weights_path=read_weights_path, 
				   _model_arguments=_model_arguments,
				   use_slim = use_slim,
				   convert_slim = convert_slim,
				   )