import models
from torch import nn
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from multiprocessing.pool import ThreadPool as Pool
import multiprocessing as mp
import pickle
from IPython.display import clear_output
import copy
import time
import random
import json
from torch.nn import functional as F

def read_local_params():
    return json.load(open('local/local_params.json', 'r'))

# simple stopwatch to time whatevs, in (float) seconds
# keeps track of laps along with final time
class Stopwatch:
    def __init__(self):
        self.start_time = time.time()
        self.last_time = self.start_time
        self.laps = []
    def lap(self):
        this_time = time.time()
        delta_time = this_time - self.last_time
        self.laps.append(delta_time)
        self.last_time = this_time
        return delta_time
    def stop(self):
        self.stop_time = time.time()
        self.delta_time = self.stop_time - self.start_time
        return self.delta_time

# loads a state dic by idx # rather than name
def load_state_dict_idx(model, state_dict):
    model_state_dict = {}
    model_keys = list(model.state_dict().keys())
    for idx, key in enumerate(state_dict):
        model_state_dict[model_keys[idx]] = state_dict[key].clone()
    model.load_state_dict(model_state_dict)
    return model

# reads imgs from file
def get_imgs(img_paths, img_type=0):
    imgs = [None] * len(img_paths)
    for i, img_path in enumerate(img_paths):
        if img_type == 0:
            img = np.expand_dims(cv2.imread(img_path, img_type).astype(np.float32), axis=0)
        if img_type == 1:
            img = cv2.cvtColor(cv2.imread(img_path, img_type).astype(np.float32), cv2.COLOR_BGR2RGB).swapaxes(0, 2).swapaxes(1, 2)
        imgs[i] = img
    return imgs

# reads imgs from file
def get_imgs2(img_paths, img_type, img_shape, imgs=None):
    if imgs is None:
        imgs = np.zeros(img_shape, dtype=np.float32)
    for i, img_path in enumerate(img_paths):
        if img_type == 0:
            img = np.expand_dims(cv2.imread(img_path, img_type).astype(np.float32), axis=0)
        if img_type == 1:
            img = cv2.cvtColor(cv2.imread(img_path, img_type).astype(np.float32), cv2.COLOR_BGR2RGB).swapaxes(0, 2).swapaxes(1, 2)
        imgs[i] = img
    return imgs

def concat_list(list_of_lists):
    concat = []
    for l in list_of_lists:
        concat = concat + l
    return concat

def get_mean_std(X):
    X_mean = np.zeros((3, 1, 1), dtype=np.float32)
    X_mean[0, 0, 0] = np.mean(X[:,0,:,:]/255)
    X_mean[1, 0, 0] = np.mean(X[:,1,:,:]/255)
    X_mean[2, 0, 0] = np.mean(X[:,2,:,:]/255)
    X_std = np.zeros((3, 1, 1), dtype=np.float32) 
    X_std[0, 0, 0] = np.std(X[:,0,:,:]/255)
    X_std[1, 0, 0] = np.std(X[:,1,:,:]/255)
    X_std[2, 0, 0] = np.std(X[:,2,:,:]/255)
    return X_mean, X_std

# preprocess a set of features and labels
# convert to torch DataLoader
class MyDataset(Dataset):
    def __init__(self, X, Y, with_norm=False, mean=None, std=None):
        self.X = X
        self.Y = Y
        self.with_norm = with_norm
        if with_norm:
            self.mean = mean
            self.std = std
    def __getitem__(self, index):
        if self.with_norm:
            return ((self.X[index]/255-self.mean)/self.std), (self.Y[index]/255)
        return (self.X[index]/255), (self.Y[index]/255)
    def __len__(self):
        return len(self.Y)

def make_dataloader(X, Y, with_norm=False, mean=None, std=None, batch_size=16, shuffle=False, drop_last=False, pin_memory=False, num_workers=0):
    DS = MyDataset(X, Y, with_norm, mean, std)
    DL = DataLoader(DS, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, pin_memory=pin_memory, num_workers=num_workers)
    return DL

# compress images using jpeg and quality
def jpg_compress(imgs, quality, save_to=None, img_names=None,):
    jpgs = [None] * len(imgs)
    if save_to is not None:
        if not os.path.exists(save_to):
            os.makedirs(save_to)
    for idx in range(len(imgs)):
        img = imgs[idx]
        write_path = 'local/temp.jpg'
        if save_to is not None:
            img_name = img_names[idx].split('.')[0] # truncate file type
            write_path = save_to + img_name + '.jpg'
        img = cv2.cvtColor(img.swapaxes(0, 2).swapaxes(0, 1), cv2.COLOR_RGB2BGR)
        result = cv2.imwrite(write_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        jpg = cv2.cvtColor(cv2.imread(write_path, 1).astype(np.float32), cv2.COLOR_BGR2RGB).swapaxes(0, 2).swapaxes(1, 2)
        jpgs[idx] = jpg
    return np.stack(jpgs)

# compress images using jpeg and quality without creating new memory (**WARNING** will overwrite old imgs array)
def jpg_compress2(imgs, quality, save_to=None, img_names=None,):
    if save_to is not None:
        if not os.path.exists(save_to):
            os.makedirs(save_to)
    for idx in range(len(imgs)):
        img = imgs[idx]
        write_path = 'local/temp.jpg'
        if save_to is not None:
            img_name = img_names[idx].split('.')[0] # truncate file type
            write_path = save_to + img_name + '.jpg'
        img = cv2.cvtColor(img.swapaxes(0, 2).swapaxes(0, 1), cv2.COLOR_RGB2BGR)
        result = cv2.imwrite(write_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        jpg = cv2.cvtColor(cv2.imread(write_path, 1).astype(np.float32), cv2.COLOR_BGR2RGB).swapaxes(0, 2).swapaxes(1, 2)
        imgs[idx] = jpg
    return imgs

# evaluate a model
def get_predictions(model, DL, device, quantize=False, split_point=0):
    x_min, x_max = -2, 22
    scale = (x_max - x_min)/255
    Y, P = [], []
    model.eval()
    for i, data in enumerate(DL):
        x, y = data
        x = x.to(device)
        if not quantize:
            p = model(x)
            Y.append(y)
            P.append(p.cpu().detach().numpy())
        else:
            for m_idx, module in enumerate(model):
                x = module(x)
                if m_idx == split_point:
                    x = torch.quantize_per_tensor(x, scale, x_min, torch.qint8)
                    x = torch.dequantize(x)
            Y.append(y)
            P.append(x.cpu().detach().numpy())
    return np.vstack(Y), np.vstack(P)

# evaluates model with given torch dataloader
    # with_grad will update model after evaluation
def evaluate(model, DL, device, quantize=False, split_point=0,
             with_grad=False, criterion=None, 
            ):
    x_min, x_max = -2, 22
    scale = (x_max - x_min)/255
    Y, P = [], []
    for i, data in enumerate(DL):
        x, y = data
        x = x.to(device)
        if not quantize:
            p = model(x)
            Y.append(y)
            P.append(p.cpu().detach().numpy())
        else:
            for m_idx, module in enumerate(model):
                x = module(x)
                if m_idx == split_point:
                    x = torch.quantize_per_tensor(x, scale, x_min, torch.qint8)
                    x = torch.dequantize(x)
            p = x
            Y.append(y)
            P.append(p.cpu().detach().numpy())
        # update params with grad?
        if with_grad:
            model.optimizer.zero_grad()
            loss = criterion(p, y.to(device=device))
            loss.backward()
            model.optimizer.step()
    Y, P = np.vstack(Y), np.vstack(P)
    return np.mean(np.abs(P-Y)/(Y))

# evaluates model with given torch dataloader
    # with_grad will update model after evaluation
def evaluate2(model, DL, device, quantize=False, split_point=0,
             with_grad=False, criterion=None, 
            ):
    x_min, x_max = -2, 22
    scale = (x_max - x_min)/255
    Y, P = [], []
    for i, data in enumerate(DL):
        x, y = data
        x = x.to(device)
        if not quantize:
            p = model(x)
            Y.append(y)
            P.append(p.cpu().detach().numpy())
        else:
            for m_idx, module in enumerate(model):
                x = module(x)
                if m_idx == split_point:
                    x = torch.quantize_per_tensor(x, scale, x_min, torch.qint8)
                    x = torch.dequantize(x)
            p = x
            Y.append(y)
            P.append(p.cpu().detach().numpy())
        # update params with grad?
        if with_grad:
            model.optimizer.zero_grad()
            loss = criterion(p, y.to(device=device))
            loss.backward()
            model.optimizer.step()
    Y, P = np.vstack(Y), np.vstack(P)
    return np.mean(np.abs(P-Y))

# train neural network model
def train(device, model, criterion, DL_train, DL_val, DL_test, test_name,
         patience=7, max_epochs=10_000, show_lc = False, print_lc = False, pytorch_threads=1, checkpoint=False, model_path=None,
         quantize=False, split_point=0,
         ):
    def forward_train(DL):      
        model.train()
        return evaluate(model, DL, device, quantize=quantize, split_point=split_point, with_grad=True, criterion=criterion)

    def forward_eval(DL):     
        model.eval()
        return evaluate(model, DL, device, quantize=quantize, split_point=split_point)
    
    def show_curve(train_errs, val_errs):
        clear_output()
        plt.plot(train_errs, label='train')
        plt.plot(val_errs, label='val')
        plt.xlabel('Epoch')
        plt.ylabel('MAPE')
        plt.title(test_name)
        plt.legend()
        plt.show()

    torch.set_num_threads(pytorch_threads)
    sw = Stopwatch()
    train_err = forward_eval(DL_train)
    best_err = val_err = forward_eval(DL_val)
    train_times = [sw.stop()]
    best_weights = copy.deepcopy(model.state_dict())
    train_errs, val_errs = [train_err], [val_err]
    if show_lc:
        show_curve(train_errs, val_errs)
    if print_lc:
        print(0, train_err, val_err, train_times[0])
    wait = 0
    for epoch in range(max_epochs):        
        sw = Stopwatch()
        train_err = forward_train(DL_train)
        val_err = forward_eval(DL_val)
        train_errs.append(train_err)
        val_errs.append(val_err)
        delta_t = sw.stop()
        if show_lc:
            show_curve(train_errs, val_errs)
        if print_lc:
            print(epoch+1, train_err, val_err, delta_t)
        if val_err < best_err:
            best_err = val_err
            best_weights = copy.deepcopy(model.state_dict())
            wait = 0
            if checkpoint and model_path is not None:
                torch.save(model, model_path)
        else:
            wait += 1
        train_times.append(delta_t)
        if wait > patience:
            model.optimizer.param_groups[0]['lr'] = model.optimizer.param_groups[0]['lr']/10
            model.optimizer.param_groups[1]['lr'] = model.optimizer.param_groups[1]['lr']/10
            if model.optimizer.param_groups[1]['lr'] < 1e-7:
                break
            wait = 0
    model.load_state_dict(best_weights)
    if model_path is not None:
        torch.save(model, model_path)
    test_err = forward_eval(DL_test)
    return model, train_errs, val_errs, train_times, test_err

def forward_head(parent, student, split_point, DL, device, quantize=False):
    x_min, x_max = -2, 22
    scale = (x_max - x_min)/255
    for i, data in enumerate(DL):
        PARENT, STUDENT = [], []
        x_0, y = data
        x = x_0.to(device)
        for m_idx, module in enumerate(parent):
            x = module(x)
            if m_idx > split_point:
                PARENT.append(x)
        x = x_0.to(device)
        for m_idx, module in enumerate(student):
            x = module(x)
            if quantize and m_idx == split_point:
                x = torch.quantize_per_tensor(x, scale, x_min, torch.qint8)
                x = torch.dequantize(x)
            if m_idx > split_point:
                STUDENT.append(x)
        student.optimizer.zero_grad()
        loss = sum(F.mse_loss(STUDENT[i], PARENT[i]) for i in range(len(PARENT))).mean()
        loss.backward()
        student.optimizer.step()
        
def forward_tail(parent, student, DL, device, quantize=False, split_point=0):
    x_min, x_max = -2, 22
    scale = (x_max - x_min)/255
    for i, data in enumerate(DL):
        x, y = data
        x = x.to(device=device)
        p_parent = parent(x)
        
        if not quantize:
            p_student = student(x)
        else:
            for m_idx, module in enumerate(student):
                x = module(x)
                if m_idx == split_point:
                    x = torch.quantize_per_tensor(x, scale, x_min, torch.qint8)
                    x = torch.dequantize(x)
            p_student = x
        
        y = y.to(device=device)
        student.optimizer.zero_grad()
        loss = sum((F.mse_loss(p_student, y), F.mse_loss(p_student, p_parent))).mean()
        loss.backward()
        student.optimizer.step()

# train bottleneck neural network model
def train_bottleneck(device, parent, student, DL_train, DL_val, DL_test, test_name, part='head', quantize=True, split_point=0,  
         patience=7, max_epochs=10_000, show_lc = False, print_lc = False, pytorch_threads=1, checkpoint=False, model_path=None,
         ):
    parent.eval()
    def forward_train(DL):      
        student.train()
        return evaluate(student, DL, device, quantize=quantize, split_point=split_point, with_grad=True, criterion=criterion)

    def forward_eval(DL):     
        student.eval()
        return evaluate(student, DL, device, quantize=quantize, split_point=split_point)
    
    def show_curve(train_errs, val_errs):
        clear_output()
        plt.plot(train_errs, label='train')
        plt.plot(val_errs, label='val')
        plt.xlabel('Epoch')
        plt.ylabel('MAPE')
        plt.title(test_name)
        plt.legend()
        plt.show()

    torch.set_num_threads(pytorch_threads)
    sw = Stopwatch()
    train_err = forward_eval(DL_train)
    best_err = val_err = forward_eval(DL_val)
    train_times = [sw.stop()]
    best_weights = copy.deepcopy(student.state_dict())
    train_errs, val_errs = [train_err], [val_err]
    if show_lc:
        show_curve(train_errs, val_errs)
    if print_lc:
        print(0, train_err, val_err, train_times[0])
    wait = 0
    for epoch in range(max_epochs):
        sw = Stopwatch()
        if part == 'head':
            _ = forward_head(parent, student, split_point, DL_train, device, quantize=quantize)
        if part == 'tail':
            _ = forward_tail(parent, student, DL_train, device, quantize=quantize, split_point=split_point)
        train_err = forward_eval(DL_train)
        val_err = forward_eval(DL_val)
        train_errs.append(train_err)
        val_errs.append(val_err)
        delta_t = sw.stop()
        if show_lc:
            show_curve(train_errs, val_errs)
        if print_lc:
            print(epoch+1, train_err, val_err, delta_t)
        if val_err < best_err:
            best_err = val_err
            best_weights = copy.deepcopy(student.state_dict())
            wait = 0
            if checkpoint and model_path is not None:
                torch.save(student, model_path)
        else:
            wait += 1
        train_times.append(delta_t)
        if wait > patience:
            student.optimizer.param_groups[0]['lr'] = student.optimizer.param_groups[0]['lr']/10
            student.optimizer.param_groups[1]['lr'] = student.optimizer.param_groups[1]['lr']/10
            if student.optimizer.param_groups[1]['lr'] < 1e-7:
                break
            wait = 0
    student.load_state_dict(best_weights)
    if model_path is not None:
        torch.save(student, model_path)
    test_err = forward_eval(DL_test)
    return student, train_errs, val_errs, train_times, test_err

def revolve_idx(idx, max_idx):
    idx += 1
    if idx >= max_idx:
        idx = 0
    return idx
def create_jobs(all_params, instances, overwrite=False):
    remaining_jobs = instances.copy()
    jobs = {instance:[] for instance in instances}
    queue = []
    loop = 0
    while len(queue) < len(all_params):
        instance = list(remaining_jobs.keys())[loop]
        queue.append(instance)
        remaining_jobs[instance] -= 1
        if remaining_jobs[instance] <= 0:
            del remaining_jobs[instance]
            loop += 1
        if len(remaining_jobs) <= 0:
            remaining_jobs = instances.copy()
            loop = 0
        else:
            loop = revolve_idx(loop, len(remaining_jobs))
    q_idx = 0
    for p_idx, params in enumerate(all_params):
        if not overwrite and os.path.exists(params[1] + 'params.p'):
            continue
        jobs[queue[q_idx]].append(p_idx)
        q_idx = revolve_idx(q_idx, len(queue))
    return jobs


def create_bottleneck(model, block_num, bottle_head, bottle_tail, device):
    student = nn.Sequential(*list(model.children())).to(device)
    student[block_num] = bottle_head
    student[block_num+1] = bottle_tail
    return student

def freeze(model, parameter_idxs):
    for p_idx, parameter in enumerate(model.parameters()):
        if p_idx in parameter_idxs:
            parameter.requires_grad = False
def unfreeze(model):
    for parameter in model.parameters():
        parameter.requires_grad = True

class Trainer:
    def __init__(self, X_train, Y_train, X_val, Y_val, X_test, Y_test, train_list, val_list, test_list, X_train_list, X_val_list, X_test_list, X_mean=None, X_std=None):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val
        self.X_test = X_test
        self.Y_test = Y_test
        self.train_list = train_list
        self.val_list = val_list
        self.test_list = test_list
        self.X_train_list = X_train_list
        self.X_val_list = X_val_list
        self.X_test_list = X_test_list
        self.X_mean = X_mean
        self.X_std = X_std

    def run_job(self, job_name, *args):
        job_func = getattr(self, job_name)
        return job_func(*args)

    def bottlefit_job(self, test_folder='dummy_test_folder/', model_name='V1', split_point=0, quantize=False, compression=64, head_block=None, tail_block=None, run_num=0, device_name='cuda', random_seed=42):
        sw = Stopwatch()
        params = locals().copy()
        del params['self']
        model_func = getattr(models, model_name)        
        device = torch.device(device_name)
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
        if not os.path.exists(test_folder):
            os.makedirs(test_folder)

        # get standardization stats
        X_mean, X_std = self.X_mean, self.X_std
        if self.X_train is None or self.X_std is None:
            X_mean, X_std = get_mean_std(self.X_train)

        # convert to dataloader
        DL_train = make_dataloader(self.X_train, self.Y_train, with_norm=True, mean=X_mean, std=X_std, shuffle=True, drop_last=True)
        DL_val = make_dataloader(self.X_val, self.Y_val, with_norm=True, mean=X_mean, std=X_std)
        DL_test = make_dataloader(self.X_test, self.Y_test, with_norm=True, mean=X_mean, std=X_std)

        # read parent
        #parent = model_func().to(device)
        #state_dict = torch.load('models/' + model_name + '.pt', map_location=device)
        #parent.load_state_dict(state_dict)
        parent = torch.load('models/' + model_name + '.pt', map_location=device)
        
        # create student
        model_path = test_folder + 'model.pt'
        student = create_bottleneck(parent, split_point, head_block, tail_block, device).to(device)
        lr = 5e-4
        student.optimizer = torch.optim.Adam([
            {'params': [param for name, param in student.named_parameters() if name[-4:] == 'bias' and param.requires_grad], 'lr': 2 * lr},
            {'params': [param for name, param in student.named_parameters() if name[-4:] != 'bias' and param.requires_grad], 'lr': lr}
        ])
        n_params = len([p for p in student.parameters()])
        head_param_idxs = []
        for i in range(split_point+2):
            for j in range(3):
                head_param_idxs.append(3*i+j)
        tail_param_idxs = [i for i in range(n_params) if i not in head_param_idxs]
        
        # train head
        freeze(student, tail_param_idxs)
        student, train_errs1, val_errs1, train_times1, test_err1 = train_bottleneck(device, parent, student, DL_train, DL_val, DL_test, test_folder, 
                                                                                    part='head', quantize=quantize, split_point=split_point, show_lc=False, print_lc=False, model_path=model_path)
        unfreeze(student)
        
        # train tail
        freeze(student, head_param_idxs)
        student, train_errs2, val_errs2, train_times2, test_err2 = train_bottleneck(device, parent, student, DL_train, DL_val, DL_test, test_folder, 
                                                                                    part='tail', quantize=quantize, split_point=split_point, show_lc=False, print_lc=False, model_path=model_path)
        unfreeze(student)
        
        curves_path = test_folder + 'curves.p'
        run_time = sw.stop()
        curves = {
            'train_errs':train_errs1+train_errs2, 
            'val_errs':val_errs1+val_errs2, 
            'train_times':train_times1+train_times2, 
            'test_err':test_err2,
            'run_time':run_time,
        }
        pickle.dump(curves, open(curves_path, 'wb'))
        params_path = test_folder + 'params.p'
        pickle.dump(params, open(params_path, 'wb'))
        return f'test_mape = {round(test_err2, 4)} in time = {sw.stop()} seconds'

    def basic_job(self, test_folder='dummy_test_folder/', model_name='ResNet152', split_point=4, quantize=False, midpoint=128, run_num=0, device_name='cuda', random_seed=42):
        sw = Stopwatch()
        params = locals().copy()
        del params['self']
        model_func = getattr(models, model_name)        
        device = torch.device(device_name)
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
        if not os.path.exists(test_folder):
            os.makedirs(test_folder)
            
        # get standardization stats
        X_mean, X_std = self.X_mean, self.X_std
        if self.X_train is None or self.X_std is None:
            X_mean, X_std = get_mean_std(X_train)

        # convert to dataloader
        DL_train = make_dataloader(self.X_train, self.Y_train, with_norm=True, mean=X_mean, std=X_std, shuffle=True, drop_last=True)
        DL_val = make_dataloader(self.X_val, self.Y_val, with_norm=True, mean=X_mean, std=X_std)
        DL_test = make_dataloader(self.X_test, self.Y_test, with_norm=True, mean=X_mean, std=X_std)

        # create random init model
        if midpoint is None:
            model = model_func().to(device)
        else:
            model = model_func(midpoint).to(device)
        lr = 5e-4
        model.optimizer = torch.optim.Adam([
            {'params': [param for name, param in model.named_parameters() if name[-4:] == 'bias' and param.requires_grad], 'lr': 2 * lr},
            {'params': [param for name, param in model.named_parameters() if name[-4:] != 'bias' and param.requires_grad], 'lr': lr}
        ])
        
        # train model
        model_path = test_folder + 'model.pt'
        criterion = nn.L1Loss() # nn.MSELoss() nn.L1Loss()
        model, train_errs, val_errs, train_times, test_err = train(device, model, criterion, DL_train, DL_val, DL_test, 
                                                                    test_folder, show_lc=False, print_lc=False, model_path=model_path,
                                                                    quantize=quantize, split_point=split_point)

        curves_path = test_folder + 'curves.p'
        run_time = sw.stop()
        curves = {
            'train_errs':train_errs, 
            'val_errs':val_errs, 
            'train_times':train_times, 
            'test_err':test_err,
            'run_time':run_time,
        }
        pickle.dump(curves, open(curves_path, 'wb'))
        params_path = test_folder + 'params.p'
        pickle.dump(params, open(params_path, 'wb'))
        return f'test_mape = {round(test_err, 4)} in time = {sw.stop()} seconds'

    def jpg_job(self, test_folder='dummy_test_folder/', model_name='V1', midpoint=None, quality=95, run_num=0, device_name='cuda', random_seed=42, mem_optim=False):
        sw = Stopwatch()
        params = locals().copy()
        del params['self']
        model_func = getattr(models, model_name)        
        device = torch.device(device_name)
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
        if not os.path.exists(test_folder):
            os.makedirs(test_folder)

        # compress images and evaluate error
        jpg_func = jpg_compress
        if mem_optim:
            jpg_func = jpg_compress2
        X_train_compressed = jpg_func(self.X_train, quality, 'local/temp/jpg_' + str(quality) + '/', self.train_list)
        X_val_compressed = jpg_func(self.X_val, quality, 'local/temp/jpg_' + str(quality) + '/', self.val_list)
        X_test_compressed = jpg_func(self.X_test, quality, 'local/temp/jpg_' + str(quality) + '/', self.test_list)

        # get standardization stats
        X_mean, X_std = self.X_mean, self.X_std
        if self.X_train is None or self.X_std is None:
            X_mean, X_std = get_mean_std(X_train_compressed)

        # convert to dataloader
        DL_train = make_dataloader(X_train_compressed, self.Y_train, with_norm=True, mean=X_mean, std=X_std, shuffle=True, drop_last=True)
        DL_val = make_dataloader(X_val_compressed, self.Y_val, with_norm=True, mean=X_mean, std=X_std)
        DL_test = make_dataloader(X_test_compressed, self.Y_test, with_norm=True, mean=X_mean, std=X_std)

        # create random init model
        if midpoint is None:
            model = model_func().to(device)
        else:
            model = model_func(midpoint).to(device)
        lr = 5e-4
        model.optimizer = torch.optim.Adam([
            {'params': [param for name, param in model.named_parameters() if name[-4:] == 'bias' and param.requires_grad], 'lr': 2 * lr},
            {'params': [param for name, param in model.named_parameters() if name[-4:] != 'bias' and param.requires_grad], 'lr': lr}
        ])

        # train model
        model_path = test_folder + 'model.pt'
        criterion = nn.L1Loss() # nn.MSELoss() nn.L1Loss()
        model, train_errs, val_errs, train_times, test_err = train(device, model, criterion, DL_train, DL_val, DL_test, 
                                                                    test_folder, show_lc=False, print_lc=False, model_path=model_path)

        if mem_optim:
            get_imgs2(self.X_train_list,1,None,self.X_train)
            get_imgs2(self.X_val_list,1,None,self.X_val)
            get_imgs2(self.X_test_list,1,None,self.X_test)
            
        curves_path = test_folder + 'curves.p'
        run_time = sw.stop()
        curves = {
            'train_errs':train_errs, 
            'val_errs':val_errs, 
            'train_times':train_times, 
            'test_err':test_err,
            'run_time':run_time,
        }
        pickle.dump(curves, open(curves_path, 'wb'))
        
        params_path = test_folder + 'params.p'
        pickle.dump(params, open(params_path, 'wb'))
        return f'test_mape = {round(test_err, 4)} in time = {sw.stop()} seconds'