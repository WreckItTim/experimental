from sklearn.metrics import r2_score as r2_score_flat
from torch.utils.data import Dataset, DataLoader
from torch import nn
from functools import partial
import multiprocessing as mp
from IPython.display import clear_output
from utils.global_methods import * # sys.path.append('path/to/parent/repo/') from parent code
import copy
import numpy as np
import matplotlib.pyplot as plt

print('cpu count:', os.cpu_count())
print('gpu count:', th.cuda.device_count())
print('cuda avail:', th.cuda.is_available())
    
# metric evaluations given true inputs and predicted ones, includes distros
    # i.e. rmse, r2, kld
def mean_squared_error(Y, P):
    return ((Y.flatten()-P.flatten())**2).mean()
def root_mean_squared_error(Y, P):
    return math.sqrt(mean_squared_error(Y, P))
def r2_score(Y, P):
    return r2_score_flat(Y.flatten(), P.flatten())
def mean_absolute_percent_error_tensor(Y, P):
    Y = Y.flatten()
    P = P.flatten()
    return ((Y-P)/Y).abs().mean()
def mean_absolute_percent_error_numpy(Y, P):
    Y = Y.flatten()
    P = P.flatten()
    return float(np.abs((Y-P)/Y).mean())
# calculates error metrics per instance
def r2_scores(Y, P):
    return [r2_score(Y[i], P[i]) for i in range(len(Y))]
def mean_absolute_percent_errors_numpy(Y, P):
    return [mean_absolute_percent_error_numpy(Y[i], P[i]) for i in range(len(Y))]



# preprocess a set of features and labels:
    # splits into train,val,test based on indexes
    # standradizes features, normalizes labels
    # converts to, and returns, torch dataloaders
class MyDataset(Dataset):
    def __init__(self, X, Y, sample_size=None, x_preproc_func=None, x_preproc_params={}, y_preproc_func=None, y_preproc_params={},):
        self.X = X.copy()
        self.Y = Y.copy()
        if sample_size is None:
            sample_size = len(X)
        self.sample_size = sample_size
        self.x_preproc_func = x_preproc_func
        self.x_preproc_params = x_preproc_params
        self.y_preproc_func = y_preproc_func
        self.y_preproc_params = y_preproc_params
    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.x_preproc_func is not None:
            x = self.x_preproc_func(x, **self.x_preproc_params)
        if self.y_preproc_func is not None:
            y = self.y_preproc_func(y, **self.y_preproc_params)
        return x, y
    def __len__(self):
        return self.sample_size
# no y
class MyDataset2(Dataset):
    def __init__(self, X, sample_size=None, x_preproc_func=None, x_preproc_params={}):
        self.X = X.copy()
        if sample_size is None:
            sample_size = len(X)
        self.sample_size = sample_size
        self.x_preproc_func = x_preproc_func
        self.x_preproc_params = x_preproc_params
    def __getitem__(self, index):
        x = self.X[index]
        if self.x_preproc_func is not None:
            x = self.x_preproc_func(x, **self.x_preproc_params)
        return x
    def __len__(self):
        return self.sample_size
# each X/Y train/val is a list of numpy arrays with same shape[1] but not necessarily shape[0]
    # the index of the list corresponds to a different group (which is variable and up to implementation)
    # for example each group can be a different watershed site
def preproc(X, Y, batch_size=32, shuffle=False, drop_last=False, num_workers=0, pin_memory=False, sample_size=None, 
            x_preproc_func=None, x_preproc_params={}, y_preproc_func=None, y_preproc_params={},):
    # convert to torch DataLoader
    DS = MyDataset(X, Y, sample_size, x_preproc_func, x_preproc_params, y_preproc_func, y_preproc_params)
    DL = DataLoader(DS, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers, pin_memory=pin_memory)
    return DL
# no Y
def preproc2(X, batch_size=32, shuffle=False, drop_last=False, num_workers=0, pin_memory=False, sample_size=None, 
            x_preproc_func=None, x_preproc_params={},):
    # convert to torch DataLoader
    DS = MyDataset2(X, sample_size, x_preproc_func, x_preproc_params)
    DL = DataLoader(DS, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers, pin_memory=pin_memory)
    return DL

# makes a pytorch MLP block  (a list of modules)
    # layers is the number of nodes in [input, hidden1, ..., hiddenN, output] so must have a length of atleast 2
    # dropout is None to not use or a list with nodes to drop in each layer above (not including output)
def mlp_modules(layers, dropout=None, hid_activation=nn.ELU, out_activation=nn.Sigmoid, with_bias=True):
    modules = []
    for idx in range(len(layers)-1):
        if dropout is not None and dropout[idx] > 0:
            modules.append(nn.Dropout(dropout[idx]))
        modules.append(nn.Linear(layers[idx], layers[idx + 1], bias=with_bias))
        if idx < len(layers)-2:
            modules.append(hid_activation())
    if out_activation is not None:
        modules.append(out_activation())
       
    return modules

# creates a pytorch MLP model
def create_mlp(layers, dropout=None, hid_activation=nn.ELU, out_activation=nn.Sigmoid, with_bias=True):
    modules = mlp_modules(layers, dropout, hid_activation, out_activation, with_bias)
    model = nn.Sequential(*modules)
    
    return model

# creates several blocks of conv>activation>pool, followed by fully connected MLP
def create_cnn(block_layers):
    
    # bulid CNN blocks
    blocks = []
    for i in range(len(block_layers)):
        block = []
        for j in range(len(block_layers[i])):
            layer_func = block_layers[i][j][0]
            if layer_func is None:
                continue
            if len(block_layers[i][j]) == 1:
                layer = layer_func()
            else:
                layer_params = block_layers[i][j][1]
                layer = layer_func(**layer_params)
            if isinstance(layer, list):
                block = block + layer
            else:
                block.append(layer)
        blocks.append(nn.Sequential(*block))

    # make model
    model = nn.Sequential(*blocks)
    
    return model

def forward_val(model, DL, device, criterion, mem_optim=True):
    losses = []
    for i, data in enumerate(DL):
        x, y = data
        with th.no_grad():
            p = model(x.to(device=device))
            loss = float(criterion(p, y.to(device=device)).detach().cpu())
            losses.append(loss)
        if mem_optim:
            del x, y, p # clear mem from gpu
    return float(np.mean(losses))
def forward_train(model, DL, device, criterion, mem_optim=True):
    losses = []
    for i, data in enumerate(DL):
        x, y = data
        model.optimizer.zero_grad()
        p = model(x.to(device=device))
        loss = criterion(p, y.to(device=device))
        loss.backward()
        model.optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if mem_optim:
            del x, y, p # clear mem from gpu
    return float(np.mean(losses))
def forward_predictions(model, DL, device, mem_optim=True, DL_includes_y=False,
                       unprocess_func=None, unprocess_params={}):
    P = []
    idx = 0
    for i, data in enumerate(DL):
        if DL_includes_y:
            x, y = data
        else:
            x = data
        with th.no_grad():
            p = model(x.to(device=device)).cpu().detach().numpy()
            if unprocess_func is not None:
                p = unprocess_func(p, **unprocess_params)
            P.append(p)
        if mem_optim:
            del x, p # clear mem from gpu
    return np.vstack(P)
# train neural network model
def train(model, DL_train, DL_val=None,
          device='cpu', criterion=nn.MSELoss(), minimize_error=True,
          patience=10, max_epochs=10_000, augmentors=None, show_curve=True, show_curve_freq=1,
          pytorch_threads=1, checkpoint_freq=0, run_path=None, output_progress=True,
          forward_train_func=forward_train, forward_train_extra_params={}, forward_val_func=forward_val, forward_val_extra_params={},
         ):
    def _forward_train(DL):
        if augmentors is not None:
            X_old = DL.dataset.X.copy()
            Y_old = DL.dataset.Y.copy()
            for augmentor in augmentors:
                DL.dataset.X = augmentor.augment(DL.dataset.X)
        model.train()
        mean_loss = forward_train_func(model, DL, device, criterion, **forward_train_extra_params)
        if augmentors is not None:
            DL.dataset.X = X_old
            DL.dataset.Y = Y_old
        return mean_loss

    def _forward_val(DL):
        model.eval()
        mean_loss = forward_val_func(model, DL, device, criterion, **forward_val_extra_params)
        return mean_loss

    th.set_num_threads(pytorch_threads)
    best_weights = copy.deepcopy(model.state_dict())
    wait = 0
    train_errors = [_forward_val(DL_train)]
    val_errors = None
    if DL_val is not None:
        val_errors = [_forward_val(DL_val)]
        best_error = val_errors[0]
    train_times = []
    error_multiplier = 1 if minimize_error else -1
    for epoch in range(max_epochs):
        if show_curve and epoch % show_curve_freq == 0:
            clear_output()
            plt.plot(train_errors, label='train')
            if val_errors is not None:
                plt.plot(val_errors, label='val')
            plt.legend()
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.show()
        sw = Stopwatch()
        train_error = _forward_train(DL_train)
        train_errors.append(train_error)
        if DL_val is not None:
            val_error = _forward_val(DL_val)
            val_errors.append(val_error)
            if error_multiplier*val_error < error_multiplier*best_error:
                best_error = val_error
                best_weights = copy.deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1
        epoch_time = sw.stop()
        train_times.append(epoch_time)
        print(epoch, train_error, val_error, epoch_time)
        if wait > patience:
            break
        if checkpoint_freq > 0 and epoch % checkpoint_freq == 0:
            th.save(model, run_path+'model_ckpt.pt')
            pk.dump(train_errors, open(run_path+'train_errors_ckpt.p', 'wb'))
            pk.dump(val_errors, open(run_path+'val_errors_ckpt.p', 'wb'))
        if output_progress:
            progress(get_global('job_name'), f'epoch {epoch} loss {round(val_error,4)} time {round(epoch_time,2)}')
    model.load_state_dict(best_weights)
    return model, train_errors, val_errors, train_times

def get_test_predictions(model, DL, device, sample_size=0, test_dropout=False, augmentors=None):
    # get test predictions
    test_times = []
    sw = Stopwatch() 
    model.eval()
    p_test = forward_predictions(model, DL, device)
    test_times.append(sw.stop())
    if sample_size > 0:
        if augmentors is not None:
            X_old = DL.dataset.X.copy()
            Y_old = DL.dataset.Y.copy()
        ps_shape = [sample_size] + list(DL.dataset.Y.shape)
        ps_test = np.zeros(ps_shape, dtype=DL.dataset.Y.dtype)
        for l in range(sample_size):
            sw = Stopwatch() 
            model.eval()
            if test_dropout:
                for m in model.modules():
                    if m.__class__.__name__.startswith('Dropout'):
                        m.train()
            if augmentors is not None:
                for augmentor in augmentors:
                    DL.dataset.X = augmentor.augment(DL.dataset.X)       
            ps_test[l] = forward_predictions(model, DL, device)
            test_times.append(sw.stop())
            if augmentors is not None:
                DL.dataset.X = X_old.copy()
                DL.dataset.Y = Y_old.copy()
        return p_test, ps_test, test_times
    return p_test, None, test_times

# add number of elements to move up by (will rotate into valid indexes)
def rotating_idx(start_i, jump_i, end_i):
    i = start_i + jump_i
    if(i < end_i):
        return i
    else:
        over_i = i - end_i
        return rotating_idx(0, over_i, end_i)
# get list of index values where possible that end < start (rotating)
def get_idx(sI, eI, n):
    if eI > sI:
        idx = [i for i in range(sI, eI)]
    else:
        idx = [i for i in range(sI, n)]
        idx = idx + [i for i in range(eI)]
    return idx
# runs one iteration of a cross-validation fold
# can use this as 1-fold cross-validation (normal train-val-test split)
# np_features and np_labels should be shuffled as indexes are sequential
# groups are groups to split data by
# X/Y train/val/test are dictionaries containing data indexed by their group name/number
    # to elimiate groups simply index by row number
def one_fold(model_func, model_params, X_train, Y_train, X_val=None, Y_val=None,
             optimizier_func=th.optim.Adam, optimizer_params={}, minimize_error=True,
             fold_num=0, splits=[0.6, 0.2, 0.2], criterion=nn.MSELoss(), patience=10, max_epochs=1_000, 
             augmentors=None, sample_size=None, device='cpu', batch_size=32, pytorch_threads=1, num_workers=0, pin_memory=False,
             checkpoint_freq=0, folder_path='dummy/', random_seed=42, 
             x_preproc_func=None, x_preproc_params={}, y_preproc_func=None, y_preproc_params={},
             forward_train_func=forward_train, forward_train_extra_params={}, forward_val_func=forward_val, forward_val_extra_params={},
            ):
    # set random seeds for replicability
    random.seed(random_seed)
    np.random.seed(random_seed)
    th.manual_seed(random_seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(random_seed)    
    
    # fetch index values to use for fold for each train, val, test
    # get number of data points in each set
    nGroups = len(X_train)
    nTrain = int(splits[0] * nGroups)
    nVal = int(splits[1] * nGroups)
    nTest = nGroups - nTrain - nVal # account for an uneven split by giving extra to test
        # nTest gets extra otherwise will miss cross-validation test data from rotation
            # this causes some overlap in test data (a few points will get tested twice)
            # the most recent test points are the ones saved
    # start idxs for fold 0
    sTrain, eTrain = 0, nTrain
    sVal, eVal = eTrain, eTrain + nVal
    sTest, eTest = eVal, eVal + nTest
    # rotate to idxs based on current fold number
    nRot = fold_num*nTest
    sTrain, eTrain = rotating_idx(sTrain, nRot, nGroups), rotating_idx(eTrain, nRot, nGroups)
    sVal, eVal = rotating_idx(sVal, nRot, nGroups), rotating_idx(eVal, nRot, nGroups)
    sTest, eTest = rotating_idx(sTest, nRot, nGroups), rotating_idx(eTest, nRot, nGroups)
    # get list of group idxs to select for train/val/test
    train_idx = get_idx(sTrain, eTrain, nGroups)
    val_idx = get_idx(sVal, eVal, nGroups)
    test_idx = get_idx(sTest, eTest, nGroups)
    
    # fetch train, val, test data
    
    DL_train = preproc(np.vstack([X_train[i] for i in train_idx]), np.vstack([Y_train[i] for i in train_idx]), 
                       batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=pin_memory,
                      sample_size=sample_size, x_preproc_func=x_preproc_func, x_preproc_params=x_preproc_params, 
                                                y_preproc_func=y_preproc_func, y_preproc_params=y_preproc_params,)
    DL_val = None if X_val is None else preproc(np.vstack([X_val[i] for i in val_idx]), np.vstack([Y_val[i] for i in val_idx]), 
                       batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                      sample_size=sample_size, x_preproc_func=x_preproc_func, x_preproc_params=x_preproc_params, 
                                                y_preproc_func=y_preproc_func, y_preproc_params=y_preproc_params,)
    
    # make model
    #model_path = folder_path + 'model_' + str(fold_num) + '_.pt'
    run_path = folder_path
    #if checkpoint_freq > 0 and os.path.exists(model_path):
    #    model = th.load(model_path)
    #else:
    model = model_func(**model_params)
    model.to(device)

    # make optimizer
    optimizer_params['params'] = model.parameters()
    model.optimizer = optimizier_func(**optimizer_params)

    # train model
    model, train_errors, val_errors, train_times = train(model=model, DL_train=DL_train, DL_val=DL_val, 
        device=device, criterion=criterion, minimize_error=minimize_error, 
        patience=patience, max_epochs=max_epochs, augmentors=augmentors,
        pytorch_threads=pytorch_threads, checkpoint_freq=checkpoint_freq, run_path=run_path,
        forward_train_func=forward_train_func, forward_train_extra_params=forward_train_extra_params, 
        forward_val_func=forward_val_func, forward_val_extra_params=forward_val_extra_params,)
    
    return (model, train_idx, val_idx, test_idx, train_errors, val_errors, train_times)


def one_shot(model_func, model_params, run_path, X_train, Y_train, X_val=None, Y_val=None,
             optimizier_func=th.optim.Adam, optimizer_params={}, minimize_error=True,
             criterion=nn.MSELoss(), patience=10, max_epochs=1_000, 
             augmentors=None, sample_size=None, device='cpu', batch_size=32, pytorch_threads=1, num_workers=0, pin_memory=False,
             checkpoint_freq=0, random_seed=42, show_curve=True, show_curve_freq=1,
             x_preproc_func=None, x_preproc_params={}, y_preproc_func=None, y_preproc_params={},
             forward_train_func=forward_train, forward_train_extra_params={}, forward_val_func=forward_val, forward_val_extra_params={},
            ):
    # set random seeds for replicability
    random.seed(random_seed)
    np.random.seed(random_seed)
    th.manual_seed(random_seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(random_seed)    
    
    # fetch train, val
    DL_train = preproc(X_train, Y_train, 
                       batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=pin_memory,
                      sample_size=sample_size, x_preproc_func=x_preproc_func, x_preproc_params=x_preproc_params, 
                                                y_preproc_func=y_preproc_func, y_preproc_params=y_preproc_params,)
    DL_val = None if X_val is None else preproc(X_val, Y_val, 
                       batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                      sample_size=sample_size, x_preproc_func=x_preproc_func, x_preproc_params=x_preproc_params, 
                                                y_preproc_func=y_preproc_func, y_preproc_params=y_preproc_params,)
    
    # make model
    model = model_func(**model_params)
    model.to(device)

    # make optimizer
    optimizer_params['params'] = model.parameters()
    model.optimizer = optimizier_func(**optimizer_params)

    # train model
    model, train_errors, val_errors, train_times = train(model=model, DL_train=DL_train, DL_val=DL_val, 
        device=device, criterion=criterion, minimize_error=minimize_error, 
        patience=patience, max_epochs=max_epochs, augmentors=augmentors, show_curve=show_curve, show_curve_freq=show_curve_freq,
        pytorch_threads=pytorch_threads, checkpoint_freq=checkpoint_freq, run_path=run_path, 
        forward_train_func=forward_train_func, forward_train_extra_params=forward_train_extra_params, 
        forward_val_func=forward_val_func, forward_val_extra_params=forward_val_extra_params,)
    
    return model, train_errors, val_errors, train_times

def predict_model(model, device, X, x_preproc_func=None, x_preproc_params={}, 
                  batch_size=32, pytorch_threads=1, num_workers=0, pin_memory=False,
                 unprocess_func=None, unprocess_params={}):
    DL = preproc2(X, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                 x_preproc_func=x_preproc_func, x_preproc_params=x_preproc_params)
    th.set_num_threads(pytorch_threads)
    stopwatch = Stopwatch()
    P = forward_predictions(model, DL, device, unprocess_func=unprocess_func, unprocess_params=unprocess_params)
    predict_time = stopwatch.stop()
    return P, predict_time

def string_params(params):
    params2 = {}
    for name in params:
        param = params[name]
        params2[name] = str(param)
    return params2

# class Parallel:
#     def __init__(self):
#         pass

#     def instance_error_job_setup(error_names, error_arrays):
#         setatt
#     def instance_error_job(error_funcs, )
    
#### MAIN TRAINER CLASS to handle paralell training/evaluation of nueral nets
class Trainer:
    # each X/Y train/val/test is a list of numpy arrays with same shape[1] but not necessarily shape[0]
        # the index of the list corresponds to a different group (which is variable and up to implementation)
        # for example each group can be a different watershed site
    # experiment_name is subset of parent folder
    def __init__(self, experiment_name, X, Y, X_val=None, Y_val=None, parent_folder='results/',
                     x_preproc_func=None, x_preproc_params={}, y_preproc_func=None, y_preproc_params={}, unprocess_func=None, unprocess_params={}, 
             forward_train_func=forward_train, forward_train_extra_params={}, forward_val_func=forward_val, forward_val_extra_params={},):
        # set params
        self.X = X
        self.Y = Y
        self.X_val = X_val
        self.Y_val = Y_val
        self.experiment_name = experiment_name
        self.parent_folder = parent_folder
        self.x_preproc_func = x_preproc_func
        self.x_preproc_params = x_preproc_params.copy()
        self.y_preproc_func = y_preproc_func
        self.y_preproc_params = y_preproc_params.copy()
        self.unprocess_func = unprocess_func
        self.unprocess_params = unprocess_param.copy()
        self.forward_train_func = forward_train_func
        self.forward_train_extra_params = forward_train_extra_params.copy()
        self.forward_val_func = forward_val_func
        self.forward_val_extra_params = forward_val_extra_params.copy()
        self.set_dir()
        
    def combine_runs(self, include_model=False, fold=0):
        run_names = [file_name.split('.')[0] for file_name in os.listdir(self.runs_path) if file_name[0] != '.']
        curves, idxs, logs, models = {}, {}, {}, {}
        for run_name in run_names:
            run_path = self.runs_path + run_name + '/'
            curves[run_name] = pk.load(open(run_path + 'curve_' + str(fold) + '_.p', 'rb'))
            idxs[run_name] = pk.load(open(run_path + 'idx_' + str(fold) + '_.p', 'rb'))
            logs[run_name] = pd.read_csv(run_path + 'log.csv').iloc[0].to_dict()
            if include_model:
                models[run_name] = th.load(run_path + 'model_' + str(fold) + '_.pt')
        pk.dump(curves, open(self.experiment_path + 'curves.p', 'wb'))
        pk.dump(idxs, open(self.experiment_path + 'idxs.p', 'wb'))
        pk.dump(logs, open(self.experiment_path + 'logs.p', 'wb'))
        if include_model:
            pk.dump(models, open(self.experiment_path + 'models.p', 'wb'))
        
    def combine_predictions(self):
        run_names = [file_name.split('.')[0] for file_name in os.listdir(self.predictions_path) if file_name[0] != '.']
        predictions = {}
        for run_name in run_names:
            prediction_path = self.predictions_path + run_name + '.p'
            Ps = pk.load(open(prediction_path, 'rb'))
            predictions[run_name] = Ps
        pk.dump(predictions, open(self.experiment_path + 'predictions.p', 'wb'))
        
    def combine_benchmarks(self):
        run_names = [file_name.split('.')[0] for file_name in os.listdir(self.benchmarks_path) if file_name[0] != '.']
        benchmarks = {}
        for run_name in run_names:
            benchmark_path = self.benchmarks_path + run_name + '.p'
            benchmark = pk.load(open(benchmark_path, 'rb'))
            benchmarks[run_name] = benchmark
        pk.dump(benchmarks, open(self.experiment_path + 'benchmarks.p', 'wb'))
            
    def purge_errors(self, from_func='all'):
        error_names = [file_name for file_name in os.listdir(self.errors_path) if file_name[0] != '.']
        for error_name in error_names:
            func_name, run_name = error_name.split('.')[0].split(' ')
            if from_func in ['all'] or func_name in from_func:
                run_path = self.runs_path + run_name
                error_path = self.errors_path + error_name
                shutil.rmtree(run_path)
                os.remove(error_path)
                print('purged', run_name)
        
    def set_dir(self):
        self.experiment_path = self.parent_folder + self.experiment_name + '/'
        if not os.path.exists(self.experiment_path):
            os.makedirs(self.experiment_path)
        
        self.runs_path = self.experiment_path + 'runs/'
        if not os.path.exists(self.runs_path):
            os.makedirs(self.runs_path)
        
        self.predictions_path = self.experiment_path + 'predictions/'
        if not os.path.exists(self.predictions_path):
            os.makedirs(self.predictions_path)
        
        self.benchmarks_path = self.experiment_path + 'benchmarks/'
        if not os.path.exists(self.benchmarks_path):
            os.makedirs(self.benchmarks_path)
            
        self.log_path = self.experiment_path + self.experiment_name + '.csv'
            
    def read_log(self):
        return pd.read_csv(self.log_path)
    
    # runs all jobs in parallel (considers as batches)
    def run_jobs(self, job_params, n_threads, batch_size=-1):
        # get optimal batch size
        if batch_size == -1:
            batch_size = int(len(job_params)/n_threads)
            if batch_size*n_threads < len(job_params):
                batch_size += 1
            
        # group job params into batches
        batch_job_params, batch = [], []
        for i in range(len(job_params)):
            batch.append(job_params[i])
            if len(batch) % batch_size == 0 or i == len(job_params)-1:
                batch_job_params.append([batch]) # mp executes a method in a thread by unpacking a list of params
                batch = []
        
        # make multiprocessing (mp) pool and execute threads
        pool = mp.Pool(processes=n_threads)
        batch_results = pool.starmap(self.batch_job, batch_job_params)
        
        # flatten batch results
        parallel_batch_results = []
        for batch in batch_results:
            parallel_batch_results = parallel_batch_results + batch
        
        return parallel_batch_results

    # use this to run a job in paralel by passing in params dict (otherwise takes a list by default)
    def single_job(self, job_name, params):
        job_func = getattr(self, job_name)
        return job_func(**params)
    
    # use this to run a batch of jobs
    def batch_job(self, batch):
        results = []
        for job in batch:
            job_name, params = job
            results.append(self.single_job(job_name, params))
        return results
                
    def prediction_job(self, run_name, fold=0, device='cpu', multi_process_gpu=False, n_gpus=4, sets = ['train', 'val', 'eval']):
        run_path = self.runs_path + run_name + '/'
        prediction_path = self.predictions_path + run_name + '.p'
        idx_path = run_path + 'idx_' + str(fold) + '_.p'
        model_path = run_path + 'model_' + str(fold) + '_.pt'
        if os.path.exists(prediction_path):
            return
        
        if multi_process_gpu:
            params = locals()
            # get unique cuda device for multiprocessing
            multiprocess_ID = int(mp.current_process().name.split('-')[1]) % n_gpus
            device = 'cuda:' + str(multiprocess_ID)
            
        idxs = pk.load(open(idx_path, 'rb'))
        model = th.load(model_path, map_location=device)

        Ps = {}
        for s in sets:
            if s == 'eval':
                x = np.vstack(self.X_val)
            else:
                x = np.vstack(self.X[idxs[s]])
            DL = preproc2(x, y, x_preproc_func=self.x_preproc_func, x_preproc_params=self.x_preproc_params, 
                         y_preproc_func=self.y_preproc_func, y_preproc_params=self.y_preproc_params)
            P = forward_predictions(model, DL, device)
            if self.unprocess_func is not None:
                P = self.unprocess_func(P, **self.unprocess_params)
            Ps[s] = P
            
        pk.dump(Ps, open(prediction_path, 'wb'))

    def benchmark_job(self, run_name, err_funcs, fold=0, device='cpu', multi_process_gpu=False, n_gpus=4, b_val=True,):
        prediction_path = self.predictions_path + run_name + '.p'
        run_path = self.runs_path + run_name + '/'
        benchmark_path = self.benchmarks_path + run_name + '.p'
        idx_path = run_path + 'idx_' + str(fold) + '_.p'
        if os.path.exists(benchmark_path):
            return
        
        if multi_process_gpu:
            params = locals()
            # get unique cuda device for multiprocessing
            multiprocess_ID = int(mp.current_process().name.split('-')[1]) % n_gpus
            device = 'cuda:' + str(multiprocess_ID)
        
        P = pk.load(open(prediction_path, 'rb'))
        idxs = pk.load(open(idx_path, 'rb'))

        benchmarks = {err_name:{} for err_name in err_funcs}
        for s in P:
            p = P[s]
            if s == 'eval':
                y = self.Y_val
            else:
                y = self.Y[idxs[s]]
            for err_name in err_funcs:
                benchmarks[err_name][s] = err_funcs[err_name](y, p)

        pk.dump(benchmarks, open(benchmark_path, 'wb'))
    
    def lightweight_job(self, run_name, delete_model=True, force_overwrite=False, fold=0, 
                        cross_validation_params={}, prediction_params={}, benchmark_params={}, ):
        run_path = self.runs_path + run_name + '/'
        model_path = run_path + 'model_' + str(fold) + '_.pt'
        prediction_path = self.predictions_path + run_name + '.p'
        benchmark_path = self.benchmarks_path + run_name + '.p'
        self.cross_validation(**cross_validation_params)
        self.prediction_job(**prediction_params)
        self.benchmark_job(**benchmark_params)
        
    # splits data into folds
    def cross_validation(self, 
                         # cross-validation params
                         run_name, nProcesses = 4, nFolds = 4, 
                         force_overwrite = False, save_all = True, save_log = True,
                         # one-fold params
                         model_func=create_mlp, model_params={}, optimizier_func=th.optim.Adam, optimizer_params={}, 
                         minimize_error=True, fold_num=0, splits=[0.5, 0.25, 0.25], criterion=nn.MSELoss(), patience=10, max_epochs=1_000,
                         augmentors=None, sample_size=None,
                         device='cpu', batch_size=32, pytorch_threads=1, num_workers=0, pin_memory=False, 
                         checkpoint_freq=0, random_seed=42, n_gpus=4,
                        ):
        params = locals()
        # get unique cuda device for multiprocessing
        if device in ['multi_cuda']:
            params['multiprocess_ID'] = int(mp.current_process().name.split('-')[1]) % n_gpus
            device = 'cuda:' + str(params['multiprocess_ID'])
            params['device'] = device
        del params['self']
        run_path = self.runs_path + run_name + '/'
            
        if force_overwrite and os.path.exists(run_path):
            shutil.rmtree(run_path)
        
        if save_all or save_log:
            if not os.path.exists(run_path):
                os.makedirs(run_path)  

        if os.path.exists(run_path + 'log.csv') and not force_overwrite:
            return
    
        # run cross validation
        sw = Stopwatch()
        outputs = [one_fold(
            model_func, model_params, self.X, self.Y, self.X, self.Y,
            optimizier_func, optimizer_params, minimize_error,
            fold_num, splits, criterion, patience, max_epochs,
            augmentors, sample_size, device, batch_size, pytorch_threads, num_workers, pin_memory,
            checkpoint_freq, run_path, random_seed, 
            self.x_preproc_func, self.x_preproc_params, self.y_preproc_func, self.y_preproc_params,
            self.forward_train_func, self.forward_train_extra_params, self.forward_val_func, self.forward_val_extra_params,
        ) for fold_num in range(nFolds)]

        # aggregate results over all folds
        results = {
            'params':params.copy(),
            'models':[],
            'idxs':[],
            'curves':[],
        }
        
        for i, output in enumerate(outputs):
            model = output[0]
            train_idx = output[1]
            val_idx = output[2]
            test_idx = output[3]
            train_errors = output[4]
            val_errors = output[5]
            train_times = output[6]
            
            idx = {'train':train_idx, 'val':val_idx, 'test':test_idx}
            curve = {'train':train_errors, 'val':val_errors}
            times = {'train':train_times}

            results['models'].append(model)
            results['idxs'].append(idx)
            results['curves'].append(curve)

            if save_all:
                th.save(model, run_path + 'model_' + str(i) + '_.pt')
                pk.dump(idx, open(run_path + 'idx_' + str(i) + '_.p', 'wb'))
                pk.dump(curve, open(run_path + 'curve_' + str(i) + '_.p', 'wb'))

        # end cross validation
        learn_time = sw.stop()

        if save_all or save_log:
            # log results
            log_params = {
                'experiment_name': self.experiment_name,
                'learn_time':learn_time,
                'epoch_time':np.mean(train_times),
            }
            log_params.update(string_params(params))
            df = pd.DataFrame({name:[log_params[name]] for name in log_params})
            df.to_csv(run_path + 'log.csv', index=False)