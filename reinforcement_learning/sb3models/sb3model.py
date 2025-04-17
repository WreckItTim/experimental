# abstract class used to handle RL model
from component import Component
import utils.global_methods as gm
from os.path import exists
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import BaseCallback
import torch as th
from torch.nn import functional as F
import torch.nn as nn
from torch import Tensor
import copy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.type_aliases import TensorDict
import gymnasium as gym
from gymnasium import spaces

class CombinedExtractor_tim(BaseFeaturesExtractor):
    """
    Combined features extractor for Dict observation spaces.
    Builds a features extractor for each key of the space. Input from each space
    is fed through a separate submodule (CNN or MLP, depending on input shape),
    the output features are concatenated and fed through additional MLP network ("combined").

    :param observation_space:
    :param cnn_output_dim: Number of features to output from each CNN submodule(s). Defaults to
        256 to avoid exploding network sizes.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        cnn_output_dim: int = 256,
        normalized_image: bool = False,
        tim_cnn_class = NatureCNN,
        tim_cnn_kwargs = None,
    ) -> None:
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super().__init__(observation_space, features_dim=1)

        if isinstance(tim_cnn_class, str):
            tim_cnn_class = str_to_class(tim_cnn_class)

        extractors: dict[str, nn.Module] = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace, normalized_image=normalized_image):
                if tim_cnn_kwargs is None:
                    extractors[key] = tim_cnn_class(subspace, features_dim=cnn_output_dim, normalized_image=normalized_image)
                else:
                    extractors[key] = tim_cnn_class(subspace, features_dim=cnn_output_dim, normalized_image=normalized_image, **tim_cnn_kwargs)
                total_concat_size += cnn_output_dim
            else:
                # The observation key is a vector, flatten it if needed
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return th.cat(encoded_tensor_list, dim=1)

# this is modified from sb3 to change the size
class NatureCNN_tim(BaseFeaturesExtractor):
    """
    CNN from DQN Nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param observation_space: The observation space of the environment
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
        normalized_image: bool = False,
        scale:float = 1,
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "NatureCNN must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, int(32*scale), kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(int(32*scale), int(64*scale), kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(int(64*scale), int(64*scale), kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

# allow us to control when to terminate training from SB3 learn()
class StopLearning(BaseCallback):
    def __init__(self, check_learning, verbose: int = 0):
        super().__init__(verbose)
        self.check_learning = check_learning
    def _on_step(self) -> bool:
        return self.check_learning()

# CUSTOM SLIM LAYERS
class Slim(nn.Linear):
    def __init__(self, max_in_features: int, max_out_features: int, bias: bool = True,
                 device=None, dtype=None,
                slim_in=True, slim_out=True) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(max_in_features, max_out_features, bias, device, dtype)
        self.max_in_features = max_in_features
        self.max_out_features = max_out_features
        self.slim_in = slim_in
        self.slim_out = slim_out
        self.slim = 1
        
    def forward(self, input: Tensor) -> Tensor:
        if self.slim_in:
            self.in_features = int(self.slim * self.max_in_features)
        if self.slim_out:
            self.out_features = int(self.slim * self.max_out_features)
        #print(f'B4-shape:{self.weight.shape}')
        weight = self.weight[:self.out_features, :self.in_features]
        if self.bias is not None:
            bias = self.bias[:self.out_features]
        else:
            bias = self.bias
        y = F.linear(input, weight, bias)
        #gm.speak(f'RHO:{self.slim} IN:{weight.shape} OUT:{y.shape}')
        return y

# convert a neural network model to slimmable
def convert_to_slim(model):
    #after calling, set as such: new_model = copy.deepcopy(model) ..
    nLinearLayers = 0
    for module in model.modules():
        if 'Linear' in str(type(module)):
            nLinearLayers += 1
    modules = []
    onLinear = 0
    for module in model.modules():
        if 'Sequential' in str(type(module)):
            continue
        elif 'Linear' in str(type(module)):
            onLinear += 1
            max_in_features = module.in_features
            max_out_features = module.out_features
            bias = module.bias is not None
            slim_in, slim_out = True, True
            if onLinear == 1:
                slim_in = False
            if onLinear == nLinearLayers:
                slim_out = False
            new_module = Slim(max_in_features, max_out_features,
                            bias=bias, slim_in=slim_in, slim_out=slim_out)
            modules.append(new_module)
        else:
            modules.append(module)
    new_model = nn.Sequential(*modules)
    new_model.load_state_dict(copy.deepcopy(model.state_dict()))
    return new_model

def str_to_class(class_str):
    if class_str == 'CombinedExtractor_tim':
        return CombinedExtractor_tim
    if class_str == 'NatureCNN_tim':
        return NatureCNN_tim

class SB3Model(Component):
    # WARNING: child init must set sb3Type, and should have any child-model-specific parameters passed through model_arguments
        # child init also needs to save the training environment (make environment_component a constructor parameter)
    # NOTE: env=None as training and evaluation enivornments are handeled by controller
    def __init__(self, 
                read_model_path=None, 
                read_replay_buffer_path=None, 
                read_weights_path=None, 
                _model_arguments=None,
                use_slim = False,
                convert_slim = False,
            ):
        self._model_arguments = _model_arguments
        self._sb3model = None
        self.connect_priority = -4 # order: sensors, observers, environment, model
        self._is_slim = False
        self._continue_learning = True

    def stop_learning(self):
        self._continue_learning = False
    
    def check_learning(self):
        return self._continue_learning

    def connect(self):
        super().connect()
        self._model_arguments['env'] = self._environment
        # read sb3 model from file
        if self.read_model_path is not None and exists(self.read_model_path):
            self.load_model(self.read_model_path,
                tensorboard_log = self._model_arguments['tensorboard_log'],
            )
            self._sb3model.set_env(self._model_arguments['env'])
            self._sb3model.learning_starts = self._model_arguments['learning_starts']
            self._sb3model.train_freq = self._model_arguments['train_freq']
            self._sb3model._convert_train_freq()
            #gm.speak(f'loaded model from file {self.read_model_path}')
        # create sb3 model from scratch
        else:
            if 'policy_kwargs' in self._model_arguments:
                if 'features_extractor_class' in self._model_arguments['policy_kwargs']:
                    if isinstance(self._model_arguments['policy_kwargs']['features_extractor_class'], str):
                        str_class = self._model_arguments['policy_kwargs']['features_extractor_class']
                        self._model_arguments['policy_kwargs']['features_extractor_class'] = str_to_class(str_class)
            self._sb3model = self.sb3Type(**self._model_arguments)
            #gm.speak('created new model from scratch')
        # convert all linear modules to slim ones
        if self.convert_slim:
            self._is_slim = True
            self._sb3model.actor.mu = convert_to_slim(self._sb3model.actor.mu)
            #self._sb3model.actor_target.mu = convert_to_slim(self._sb3model.actor_target.mu)
            self._sb3model.slim = 1
            #gm.speak('converted model to slimmable')
        # load weights?
        if self.read_weights_path is not None:
            self.load_weights(self.read_weights_path)
            #gm.speak(f'read weights from path {self.read_weights_path}')
        # use slim layers
        if self.use_slim:
            self._is_slim = True
            self._sb3model.slim = 1
            #gm.speak('using slimmable model')
        # read replay buffer from file
        if self.read_replay_buffer_path is not None and exists(self.read_replay_buffer_path):
            self.load_replay_buffer(self.read_replay_buffer_path)
            #gm.speak(f'loaded replay buffer from file {self.read_replay_buffer_path}')
        
            
    # this will toggle if to checkpoint model and replay buffer
    def set_save(self,
              track_save,
              track_vars=[
                  'model', 
                  'replay_buffer',
                  ],
              ):
        self._track_save = track_save
        self._track_vars = track_vars.copy()
    
    # save sb3 model and replay_buffer to path
    # pass in write_folder to state
    def save(self, state):
        write_folder = state['write_folder']
        if 'model' in self._track_vars:
            self.save_model(write_folder + 'model_final.zip')
        if 'replay_buffer' in self._track_vars and self._has_replay_buffer:
            self.save_replay_buffer(write_folder + 'replay_buffer.zip')
    def save_model(self, path):
        self._sb3model.save(path)
    def save_replay_buffer(self, path):
        self._sb3model.save_replay_buffer(path)

    # load sb3 model from path, must set sb3Load from child
    def load_model(self, path, tensorboard_log=None):
        if not exists(path):
            gm.error(f'invalid Model.load_model() path:{path}')
        else:
            device=self._model_arguments['device']
            self._sb3model = self.sb3Load(path, 
                                 device=device, 
                                 tensorboard_log=tensorboard_log,
                                 )

    # load weights from pytorch .pt file (currently supports only the actor network)
    def load_weights(self, actor_path):
        device = self._model_arguments['device']
        state_dict = torch.load(actor_path)
        if actor_path is not None and exists(actor_path):
            self._sb3model.actor.mu.load_state_dict(copy.deepcopy(state_dict), map_location=device)
            self._sb3model.actor_target.mu.load_state_dict(copy.deepcopy(state_dict), map_location=device)
        #self._sb3model.critic.load_state_dict(torch.load(critic_path))
        #self._sb3model.critic_target.load_state_dict(torch.load(critic_path))

    # load sb3 replay buffer from path
    def load_replay_buffer(self, path):
        device=self._model_arguments['device']
        if not exists(path):
            gm.error(f'invalid Model.load_replay_buffer() path:{path}')
        elif self._has_replay_buffer:
            self._sb3model.load_replay_buffer(path)
        else:
            gm.warning(f'trying to load a replay buffer to a model that does not use one')

    # runs learning loop on model
    def learn(self, 
        total_timesteps = 10_000,
        log_interval = -1,
        reset_num_timesteps = False,
        tb_log_name = None,
        ):
        callback = StopLearning(self.check_learning)
        # call sb3 learn method
        self._sb3model.learn(
            total_timesteps,
            callback = callback,
            log_interval = log_interval,
            tb_log_name = tb_log_name,
            reset_num_timesteps = reset_num_timesteps,
        )
        
    # makes a single prediction given input data
    def predict(self, rl_input):
        rl_output, next_state = self._sb3model.predict(rl_input, deterministic=True)
        return rl_output

    # reset slim factors
    def reset(self, state = None):
        if self._is_slim:
            for module in self._sb3model.actor.modules():
                if 'Slim' in str(type(module)):
                    module.slim = 1
            self._sb3model.slim = 1