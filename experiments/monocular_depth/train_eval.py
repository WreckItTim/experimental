
import os
root_dir = '/home/tim/Dropbox/experimental/' # your path here where to parent directory where repos are
local_dir = '/home/tim/local/'
os.chdir(root_dir)
import sys
sys.path.append(root_dir)
import map_data.map_methods as mm
import supervised_learning.supervised_methods as sm
from supervised_learning.slim_methods import SlimConv2d, SlimBatchNorm2d, SlimConvTranspose2d#, SlimGroupNorm
from supervised_learning.slim_methods import forward_slim_train, forward_slim_val, foward_slim_predictions
from torch import nn
import utils.global_methods as gm
import sys
import matplotlib.cm as cm
import torch as th
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib import colors as mcolors
import multiprocessing as mp
initial_locals = locals().copy() # will exclude these parameters from config parameters written to file

# read params from command line
random_seed = 42
scale = 1 # scales number of channels in each layer
job_name = 'null'
device = 'cuda:0'
pytorch_threads = 8 # scale this down with more active processes running at same time
num_workers = 2 # scale this down with RAM constraints
batch_size = 16 # scale this down with VRAM constraints
pin_memory = False # toggle on if training data is small enough to fit in VRAM
overwrite = False # will clear all files in run_dir and restart everything from square 1
continue_training = True # will check if checkpoint files exist and load them and continue training if they do
scale = 1 # multiply the number of channels in the original depth network size by this
use_slim_cnn = False # True will make neural network with custom slim layers
use_slim_train = False # True will use custom forward_slim_...
use_slim_soft = False # True will add soft targets to loss function
rhos = [0.25, 0.5, 1.0]
version = 'v0'
map_name = 'AirSimNH' # this is the neighorhood map with houses and cars and roads and stuff -- airsim_map can equal 'Blocks' or 'AirSimNH'
train_sample_size = 10_000 # number of data instances to fetch (None to use all available)
test_sample_size = 10_000 # number of data instances to fetch (None to use all available)
max_epochs = 100
patience = 100
model_structure = 'DGNLNet'
optimizer_name = 'th.optim.Adam' # th.optim.SGD
optimizer_params = {}
loss_name = 'th.nn.MSELoss' # th.nn.L1Loss
scheduler_name = None # th.optim.lr_scheduler.StepLR
scheduler_params = {}
use_groupNorm = False
scale_rho = False
checkpoint_freq = 1
if len(sys.argv) > 1:
    arguments = gm.parse_arguments(sys.argv[1:])
    locals().update(arguments)

# set directory to write all results to
run_dir = f'{root_dir}models/monocular_depth/{version}/seed_{random_seed}/'
os.makedirs(run_dir, exist_ok=True)

# save all params to file
all_locals = locals()
new_locals = {k:v for k, v in all_locals.items() if (not k.startswith('__') and k not in initial_locals and k not in ['initial_locals','all_locals'])}
params = new_locals.copy()
gm.pk_write(params, f'{run_dir}params.p')

# convert params as needed
if optimizer_name is not None:
    optimizer_func = eval(optimizer_name)
if loss_name is not None:
    loss_func = eval(loss_name)()
scheduler_func = None
if scheduler_name is not None:
    scheduler_func = eval(scheduler_name)

# output job start
print('running job with params', params)
gm.set_global('local_dir', local_dir)
gm.progress(job_name, 'started')

# the numpy arrays need to match the same floating type used by pytorch here
th.set_default_dtype(th.float32)

# remove some annoying tensorflow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# set random seeds for replicability
gm.set_random_seed(random_seed)

# WARNING --- set file io -- will write model and evalutions to below paths (WILL OVERWRITE SAVED DATA, change if not desirable)
save_model_to = f'{run_dir}model.pth'
save_train_metrics_to = f'{run_dir}train_metrics.json'
save_eval_metrics_to = f'{run_dir}eval_metrics.json'

# if not overwrite and os.path.exists(f'{run_dir}r2s.p'):
#     gm.progress(job_name, 'already completed')
#     sys.exit(0)

# read data
datamap = mm.DataMap(map_name)
data = datamap.get_data(
    sensor_names = [ # name of sensors to fetch data, see README for available sensors -- names such as 'SceneV1' and 'SegmentationV1'
        'SceneV1',  # SceneV1 is monocular forward facing RGB at 144x256 resolution
        'DepthV1',  # DepthV1 is forward facing 2d depth map at 144x256 resolution
    ],
    region = 'train', # training is done on the top half of the map -- region can equal 'train' 'test' or 'all'
    sample_size = train_sample_size, # number of data instances to fetch (None to use all available)
)
X = data['observations']['SceneV1'] # RGB
Y = data['observations']['DepthV1'] # to depth
coordinates = data['coordinates'] # list of x, y, z, yaw coordinates at each index corresponding to data

# split into train and validation sets
percent_train, percent_val = 0.8, 0.2
N = len(coordinates)
n_train, n_val = int(percent_train*N), int(percent_val*N)
if n_val > 0:
    X_train = X[:n_train]
    Y_train = Y[:n_train]
    X_val = X[n_train:]
    Y_val = Y[n_train:]
else:
    X_train = X
    Y_train = Y
    X_val = None
    Y_val = None

# extract parameters from input/ouput data
n_input_channels = X_train.shape[1] # this is used in code below to make CNN model

print('Train shape:', X_train.shape, Y_train.shape)
if n_val > 0:
    print('Val shape:', X_val.shape, Y_val.shape)


X_mean = 94.11807077041952 #np.mean(X_train)
X_std = 80.77841523336176 #np.std(X_train)
Y_min = 0
Y_max = 255

def x_preproc_func(x, _mean, _std):
    return (x.astype(np.float32)-_mean)/_std
def y_preproc_func(y, _min, _max):
    return (y.astype(np.float32)-_min)/(_max-_min)
x_preproc_params={'_mean':X_mean, '_std':X_std}
y_preproc_params={'_min':Y_min, '_max':Y_max}
def unprocess_func(p, _min, _max):
    p = (p*(_max-_min)+_min)
    p[p<=1] = 1
    p[p>=255] = 255
    p = p.astype(np.uint8)
    return p
unprocess_params={'_min':Y_min, '_max':Y_max}

# add custom cnn layers to network?
if use_slim_cnn:
    Conv2d = SlimConv2d
    Norm = SlimBatchNorm2d
    if use_groupNorm:
        Norm = SlimGroupNorm
    ConvTranspose2d = SlimConvTranspose2d
    in_channels_key = 'max_in_channels'
    out_channels_key = 'max_out_channels'
else:
    Conv2d = nn.Conv2d
    Norm = nn.BatchNorm2d
    if use_groupNorm:
        Norm = GroupNorm
    ConvTranspose2d = nn.ConvTranspose2d
    in_channels_key = 'in_channels'
    out_channels_key = 'out_channels'

# make scalable network structures
# DGNLNet https://ieeexplore.ieee.org/abstract/document/9318521
if model_structure in ['DGNLNet']:
    model_func = sm.create_cnn # this is my custom method for creating a CNN (of couse you do not have to use it)
    model_params = {
        'block_layers':[
                [
                    [Conv2d, {in_channels_key:n_input_channels, out_channels_key:int(32*scale), 'kernel_size':4, 'stride':2, 'padding':1}],
                    [Norm, {'max_channels':int(32*scale), 'rhos':rhos} if use_slim_cnn else {'num_features':int(32*scale)}],
                    [nn.SELU],
                ],
                [
                    [Conv2d, {in_channels_key:int(32*scale), out_channels_key:int(64*scale), 'kernel_size':4, 'stride':2, 'padding':1}],
                    [Norm, {'max_channels':int(64*scale), 'rhos':rhos} if use_slim_cnn else {'num_features':int(64*scale)}],
                    [nn.SELU],
                ],
                [
                    [Conv2d, {in_channels_key:int(64*scale), out_channels_key:int(128*scale), 'kernel_size':4, 'stride':2, 'padding':1}],
                    [Norm, {'max_channels':int(128*scale), 'rhos':rhos} if use_slim_cnn else {'num_features':int(128*scale)}],
                    [nn.SELU],
                ],
                [
                    [Conv2d, {in_channels_key:int(128*scale), out_channels_key:int(256*scale), 'kernel_size':4, 'stride':2, 'padding':1}],
                    [Norm, {'max_channels':int(256*scale), 'rhos':rhos} if use_slim_cnn else {'num_features':int(256*scale)}],
                    [nn.SELU],
                ],
                [
                    [Conv2d, {in_channels_key:int(256*scale), out_channels_key:int(256*scale), 'kernel_size':3, 'dilation':2, 'padding':2}],
                    [Norm, {'max_channels':int(256*scale), 'rhos':rhos} if use_slim_cnn else {'num_features':int(256*scale)}],
                    [nn.SELU],
                ],
                [
                    [Conv2d, {in_channels_key:int(256*scale), out_channels_key:int(256*scale), 'kernel_size':3, 'dilation':4, 'padding':4}],
                    [Norm, {'max_channels':int(256*scale), 'rhos':rhos} if use_slim_cnn else {'num_features':int(256*scale)}],
                    [nn.SELU],
                ],
                [
                    [Conv2d, {in_channels_key:int(256*scale), out_channels_key:int(256*scale), 'kernel_size':3, 'dilation':2, 'padding':2}],
                    [Norm, {'max_channels':int(256*scale), 'rhos':rhos} if use_slim_cnn else {'num_features':int(256*scale)}],
                    [nn.SELU],
                ],
                [
                    [ConvTranspose2d, {in_channels_key:int(256*scale), out_channels_key:int(128*scale), 'kernel_size':4, 'stride':2, 'padding':1}],
                    [Norm, {'max_channels':int(128*scale), 'rhos':rhos} if use_slim_cnn else {'num_features':int(128*scale)}],
                    [nn.SELU],
                ],
                [
                    [ConvTranspose2d, {in_channels_key:int(128*scale), out_channels_key:int(64*scale), 'kernel_size':4, 'stride':2, 'padding':1}],
                    [Norm, {'max_channels':int(64*scale), 'rhos':rhos} if use_slim_cnn else {'num_features':int(64*scale)}],
                    [nn.SELU],
                ],
                [
                    [ConvTranspose2d, {in_channels_key:int(64*scale), out_channels_key:int(32*scale), 'kernel_size':4, 'stride':2, 'padding':1}],
                    [Norm, {'max_channels':int(32*scale), 'rhos':rhos} if use_slim_cnn else {'num_features':int(32*scale)}],
                    [nn.SELU],
                ],
                [
                    [ConvTranspose2d, {in_channels_key:int(32*scale), out_channels_key:int(32*scale), 'kernel_size':4, 'stride':2, 'padding':1}],
                    [Norm, {'max_channels':int(32*scale), 'rhos':rhos} if use_slim_cnn else {'num_features':int(32*scale)}],
                    [nn.SELU],
                    [Conv2d, {in_channels_key:int(32*scale), out_channels_key:int(32*scale), 'kernel_size':3, 'padding':1}],
                    [nn.SELU],
                    [Conv2d, {in_channels_key:int(32*scale), out_channels_key:int(1), 'kernel_size':1, 'stride':1, 'padding':0}],
                    [nn.Sigmoid],
                ],
           ]
    }
    if use_slim_cnn:
        model_params['block_layers'][0][0][1]['slim_in'] = False
        model_params['block_layers'][-1][-2][1]['slim_out'] = False

forward_train_func = sm.forward_train
forward_train_extra_params = {}
forward_val_func = sm.forward_val
forward_val_extra_params = {}
if use_slim_train:
    forward_train_func = forward_slim_train
    forward_train_extra_params = {'rhos':rhos, 'soft_targets':use_slim_soft, 'scale_rho':scale_rho}
    forward_val_func = forward_slim_val
    forward_val_extra_params = {'rhos':rhos}

model, train_losses, val_losses, epoch_times = sm.one_shot(
    model_func, model_params, run_dir, X_train, Y_train, X_val, Y_val, optimizier_func=optimizer_func,
    optimizer_params={}, minimize_error=True, criterion=loss_func, patience=patience, max_epochs=max_epochs,
    augmentors=None, sample_size=None, device=device, batch_size=batch_size, pytorch_threads=pytorch_threads,
    num_workers=num_workers, pin_memory=pin_memory, checkpoint_freq=checkpoint_freq,
    random_seed=random_seed, show_curve_freq=0, continue_training=continue_training,
    x_preproc_func=x_preproc_func, x_preproc_params=x_preproc_params,
    y_preproc_func=y_preproc_func, y_preproc_params=y_preproc_params, 
    forward_train_func=forward_train_func, forward_train_extra_params=forward_train_extra_params,
    forward_val_func=forward_val_func, forward_val_extra_params=forward_val_extra_params,
    scheduler_func=scheduler_func, scheduler_params=scheduler_params,
)

th.save(model, save_model_to)

train_metrics = {
    'train_losses':train_losses, 
    'val_losses':val_losses, 
    'epoch_times':epoch_times,
    'train_time':float(np.sum(epoch_times)),
    'n_epochs':len(epoch_times)-1,
}
gm.write_json(train_metrics, save_train_metrics_to)

if use_slim_cnn:
    r2s = {'train':{}, 'val':{}, 'test':{}}
else:
    r2s = {}

def eval_set(X, Y, set_name):
    DL = sm.preproc2(X, batch_size=batch_size, num_workers=num_workers,
                      x_preproc_func=x_preproc_func, x_preproc_params=x_preproc_params,
                    )
    if use_slim_cnn:
        Pslim = foward_slim_predictions(model, DL, device, rhos=rhos)
        for rho in rhos:
            r2s[set_name][rho] = sm.r2_score(Y, unprocess_func(Pslim[rho], **unprocess_params).astype(np.float32))
    else:
        P = sm.forward_predictions(model, DL, device)
        r2s[set_name] = sm.r2_score(Y, unprocess_func(P, **unprocess_params).astype(np.float32))

# eval train/val data
eval_set(X_train, Y_train, 'train')
del X_train
del Y_train
eval_set(X_val, Y_val, 'val')
del X_val
del Y_val
del data
datamap.clear_cache()

# eval test data
data = datamap.get_data(
    sensor_names = [ # name of sensors to fetch data, see README for available sensors -- names such as 'SceneV1' and 'SegmentationV1'
        'SceneV1',  # SceneV1 is monocular forward facing RGB at 144x256 resolution
        'DepthV1',  # DepthV1 is forward facing 2d depth map at 144x256 resolution
    ],
    region = 'all', # region can equal 'train' 'test' or 'all'
                               # or 'houses_{train/test/all}' for only the center portion of the map 
    sample_size = test_sample_size, # number of data instances to fetch (None to use all available)
    pull_from_end = True, # sample from end of pre-shuffled data points to keep holdout test set seperate
)
X_test = data['observations']['SceneV1'] # RGB
Y_test = data['observations']['DepthV1'] # to depth
coordinates_eval = data['coordinates']
eval_set(X_test, Y_test, 'test')
del X_test
del Y_test
del data
datamap.clear_cache()

gm.pk_write(r2s, f'{run_dir}r2s.p')

if use_slim_cnn:
    r2_test = r2s['test'][1.0]
else:
    r2_test = r2s['test']
gm.progress2(job_name, f'results {run_dir} {r2_test:0.2f} r2')
gm.progress(job_name, 'complete')