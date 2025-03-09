#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#device_name = 'cuda:0'
model_name = 'ResNet152'

# uncomment this block to grab user input
import sys
args = sys.argv
device_name = args[1]
#model_name = args[2]

from tim import *
#os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# comment out unavailable devices - or edit workload (1 is lab laptop and lowest)
# a value of 8 would apply 8 jobs to every 1 given to laptop
# assigns jobs in order of dict
instances = {
#         'mlserver2021 cuda:0':8,
#         'mlserver2021 cuda:1':8,
#         'apollo cuda:0':1,
#         'torch cuda:0':4,
         'pyro cuda:0':4,
#         'phoenix cuda:0':4,
#         'mlserver2019 cuda:0':6,
#         'mlserver2019 cuda:1':6,
         'mlserver2019 cuda:2':6,
}
local_params = read_local_params()
instance_name = local_params['server_name'] + ' ' + device_name
print('instance_name', instance_name)
print('#GPUs', torch.cuda.device_count())
print('torch available?', torch.cuda.is_available())
print('mp.cpu_count', mp.cpu_count())
print('os.cpu_count', os.cpu_count())
print('version 12/13 11:45')


# In[ ]:


# DATA

# get data paths for train/val/test
data_dir = 'data/depth_airsimnh_RGB_576_1024_depth_map_144_156/'
X_dir = data_dir + 'image/'
Y_dir = data_dir + 'depth/'

train_list = [s.replace('\n', '') for s in list(open(data_dir + 'train.txt'))]
val_list = train_list[:500]
train_list = train_list[500:]
test_list = [s.replace('\n', '') for s in list(open(data_dir + 'test.txt'))]

X_train_list = [X_dir + img_name for img_name in train_list]
Y_train_list = [Y_dir + img_name for img_name in train_list]
X_val_list = [X_dir + img_name for img_name in val_list]
Y_val_list = [Y_dir + img_name for img_name in val_list]
X_test_list = [X_dir + img_name for img_name in test_list]
Y_test_list = [Y_dir + img_name for img_name in test_list]

# read in imgs 
X_shape = [3,576,1024]
#X_shape = (3,144,256)
Y_shape = [1,144,256]
X_train = get_imgs2(X_train_list,1,[len(X_train_list)] + X_shape)
Y_train = get_imgs2(Y_train_list,0,[len(Y_train_list)] + Y_shape)
X_val = get_imgs2(X_val_list,1,[len(X_val_list)] + X_shape)
Y_val = get_imgs2(Y_val_list,0,[len(Y_val_list)] + Y_shape)
X_test = get_imgs2(X_test_list,1,[len(X_test_list)] + X_shape)
Y_test = get_imgs2(Y_test_list,0,[len(Y_test_list)] + Y_shape)
# n = 6
# b = 100
# pool = Pool(processes=n)
# X_train = np.stack(concat_list(pool.starmap(get_imgs, 
#                        [(X_train_list[i*b:min(len(X_train_list), (i+1)*b)],1) for i in range(int(len(X_train_list)/b))])))
# Y_train = np.stack(concat_list(pool.starmap(get_imgs, 
#                        [(Y_train_list[i*b:min(len(Y_train_list), (i+1)*b)],0) for i in range(int(len(Y_train_list)/b))])))
# X_val = np.stack(concat_list(pool.starmap(get_imgs, 
#                        [(X_val_list[i*b:min(len(X_val_list), (i+1)*b)],1) for i in range(int(len(X_val_list)/b))])))
# Y_val = np.stack(concat_list(pool.starmap(get_imgs, 
#                        [(Y_val_list[i*b:min(len(Y_val_list), (i+1)*b)],0) for i in range(int(len(Y_val_list)/b))])))
# X_test = np.stack(concat_list(pool.starmap(get_imgs,
#                        [(X_test_list[i*b:min(len(X_test_list), (i+1)*b)],1) for i in range(int(len(X_test_list)/b))])))
# Y_test = np.stack(concat_list(pool.starmap(get_imgs, 
#                        [(Y_test_list[i*b:min(len(Y_test_list), (i+1)*b)],0) for i in range(int(len(Y_test_list)/b))])))

# calc stats
train_mean, train_std = get_mean_std(X_train)

# assumed states
assumed_mean = np.zeros((3, 1, 1), dtype=np.float32)
assumed_std = np.zeros((3, 1, 1), dtype=np.float32) 
assumed_mean[0, 0, 0] = 0.485
assumed_mean[1, 0, 0] = 0.456
assumed_mean[2, 0, 0] = 0.406
assumed_std[0, 0, 0] = 0.229
assumed_std[1, 0, 0] = 0.224
assumed_std[2, 0, 0] = 0.225

print('X_train.shape', X_train.shape)
print('Y_train.shape', Y_train.shape)
print('X_val.shape', X_val.shape)
print('Y_val.shape', Y_val.shape)
print('X_test.shape', X_test.shape)
print('Y_test.shape', Y_test.shape)


# In[ ]:


# # predictions and evaluations

# DL_test = make_dataloader(X_test, Y_test, with_norm=False)#, mean=assumed_mean, std=assumed_std)

# state_dict = torch.load('models/' + model_name + '.pt')
# model_func = getattr(models, model_name)  
# model = model_func().to(device_name)
# model.load_state_dict(state_dict)
# model = model.to(device_name)
# print('test MAPE:', evaluate(model, DL_test, torch.device(device_name)))


# In[ ]:


# # quantization

# X2 = (X_train/255-train_mean)/train_std

# x_min, x_max = int(np.min(X2))-1, int(np.max(X2)+1)
# scale = (x_max - x_min)/255

# x0 = X2[0].astype(np.float32)
# x1 = ((x0-x_min)/scale+0.5).astype(np.uint8)
# x2 = scale*x1.astype(np.float32) + x_min

# x0t = torch.from_numpy(x0).to('cuda')
# x1t = torch.quantize_per_tensor(x0t, scale, x_min, torch.qint8)
# x2t = torch.dequantize(x1t)

# fig, axs = plt.subplots(2,3)
# axs[0,0].hist(x0.flatten())
# axs[0,1].hist(x1.flatten())
# axs[0,2].hist(x2.flatten())
# axs[1,0].hist(x0t.detach().cpu().numpy().flatten())
# axs[1,2].hist(x2t.detach().cpu().numpy().flatten())
# plt.show()


# In[ ]:


# # predictions and evaluations with quantization

# DL_test = make_dataloader(X_test, Y_test, with_norm=True, mean=assumed_mean, std=assumed_std)

# state_dict = torch.load('models/' + model_name + '.pt')
# model_func = getattr(models, model_name)  
# model = model_func().to(device_name)
# model.load_state_dict(state_dict)
# print('test MAPE:', evaluate(model, DL_test, torch.device(device_name), quantize=True, split_point=2))


# In[ ]:


# make trainer
trainer = Trainer(X_train, Y_train, X_val, Y_val, X_test, Y_test, train_list, val_list, test_list, X_train_list, X_val_list, X_test_list, train_mean, train_std)


# In[ ]:


# create jobs
all_params = []
    
# compression jobs
# experiment_folder = 'results/bottlefit_compression/' + model_name + '/block_2/'
# default_params = {
#     'run_job':'bottlefit_job',
#     'test_folder':'dummy_test_folder/', 
#     'model_name':model_name, 
#     'split_point':4,
#     'quantize':False,
#     'compression':64,
#     'head_block':None,
#     'tail_block':None,
#     'run_num':42,
#     'device_name':device_name, 
#     'random_seed':42,
# }
# for run_num in range(10):
#     for compression in [2, 16, 64]:
#         for quantize in [False]:
#             test_name = str(compression) + '_' + str(quantize) + '_' + str(run_num)
#             test_folder =  experiment_folder + test_name + '/'
#             params = default_params.copy()
#             params['test_folder'] = test_folder
#             params['quantize'] = quantize
#             params['compression'] = compression
#             params['head_block'], params['tail_block'] = models.get_head_tail(model_name, params['split_point'], compression)
#             params['run_num'] = run_num
#             params['random_seed'] = run_num
#             all_params.append([params[key] for key in params])

# jpg jobs
experiment_folder = 'results/jpg_quality/' + model_name + '/'
default_params = {
    'run_job':'jpg_job',
    'test_folder':'dummy_test_folder/', 
    'model_name':model_name, 
    'midpoint':128, 
    'quality':95, 
    'run_num':42,
    'device_name':device_name, 
    'random_seed':42,
    'mem_optim':True,
}
for run_num in range(10):
    for midpoint in [512]:
        for quality in [c*5 for c in range(1, 20, 2)]:
            test_name = str(midpoint) + '_' + str(quality) + '_' + str(run_num)
            test_folder =  experiment_folder + test_name + '/'
            params = default_params.copy()
            params['test_folder'] = test_folder
            params['midpoint'] = midpoint
            params['quality'] = quality
            params['run_num'] = run_num
            params['random_seed'] = run_num
            all_params.append([params[key] for key in params])

# basic jobs
experiment_folder = 'results/basic/' + model_name + '/'
default_params = {
    'run_job':'basic_job',
    'test_folder':'dummy_test_folder/', 
    'model_name':model_name, 
    'split_point':4,
    'quantize':False,
    'midpoint':128, 
    'run_num':42,
    'device_name':device_name, 
    'random_seed':42,
}
for run_num in range(10):
    for midpoint in [512]:
        for quantize in [False, True]:
            test_name = str(midpoint) + '_' + str(quantize) + '_' + str(run_num)
            test_folder =  experiment_folder + test_name + '/'
            params = default_params.copy()
            params['test_folder'] = test_folder
            params['quantize'] = quantize
            params['midpoint'] = midpoint
            params['run_num'] = run_num
            params['random_seed'] = run_num
            all_params.append([params[key] for key in params])
            
        
jobs = create_jobs(all_params, instances, overwrite=False)


# In[ ]:


# run jobs
print('jobs:', jobs[instance_name])
for p_idx in jobs[instance_name]:
    print('running job:', p_idx, all_params[p_idx])
    result = trainer.run_job(*all_params[p_idx])
    print('job complete!', result)


# In[ ]:


def read_results(results_dir, model_func=None, device=None, read_models=False, read_curves=True, read_params=False):
    test_names = os.listdir(results_dir)
    returns = {}
    for test_name in test_names:
        test_returns = []
        if read_models:
            model_path = os.path.join(results_dir, test_name, 'model.pt')
            model = model_func().to(device)
            model.load_state_dict(torch.load(model_path))
            test_returns.append(model)
        if read_curves:
            curves_path = os.path.join(results_dir, test_name, 'curves.p')
            curves = pickle.load(open(curves_path, 'rb'))
            test_returns.append(curves)
        if read_params:
            params_path = os.path.join(results_dir, test_name, 'params.p')
            params = pickle.load(open(params_path, 'rb'))
            test_returns.append(params)
        returns[test_name] = test_returns
    return returns


# In[ ]:


# # calculate memory of various jpg compressions
# jpg_bytes = {}
# for quality in [c*5 for c in range(1, 20, 2)]:
#     file_path = 'local/mem_sizes/jpg_quality/' + model_name + '/' + train_list[0].split('.')[0] + '.jpg'
#     X_compressed = jpg_compress(X_train[:1], quality, 'local/mem_sizes/jpg_quality/' + model_name + '/', [train_list[0]])
#     jpg_bytes[quality] = np.round((0.001*os.path.getsize(file_path)), 2)
# print('JPEG quality : mem [kb]')
# for key in jpg_bytes:
#     print(key, ':', jpg_bytes[key], 'kb')


# In[ ]:


# # calculate memory of various bottlefit compressions
# model_func = getattr(models, model_name)  
# parent = model_func().to(device_name)
# split_point = 2
# sc_bytes = {}
# sc_bytes_quantized = {}
# for compression in [2**c for c in range(1, 6, 1)]:
#     head_block, tail_block = models.get_head_tail(model_name, split_point, compression)
#     student = create_bottleneck(parent, split_point, head_block, tail_block, device_name).to(device_name)
#     x = torch.from_numpy(((X_train[:1]/255-train_mean)/train_std).astype(np.float32)).to(device_name)
#     for m_idx, module in enumerate(student):
#         x = module(x)
#         if m_idx >= split_point:
#             break
#     print(x.shape)
#     mem = 0.001*x.element_size() * x.nelement()
#     sc_bytes[compression] = np.round(mem, 2)
#     x = torch.quantize_per_tensor(x, scale, x_min, torch.qint8)
#     mem = 0.001*x.element_size() * x.nelement()
#     sc_bytes_quantized[compression] = np.round(mem, 2)
# print('SC compression : mem [kb]')
# for key in sc_bytes:
#     print(key, ':', sc_bytes[key], 'kb')
# print('SC compression quantized : mem [kb]')
# for key in sc_bytes:
#     print(key, ':', sc_bytes_quantized[key], 'kb')


# In[ ]:


# # plot errors of jpg vs bottlefit compression
# def plot_test_errs(test_errs, label, bytes_map):
#     sorted_keys = list(test_errs.keys())
#     sorted_keys.sort()
#     mean_errs = [np.mean(test_errs[key]) for key in sorted_keys]
#     std_errs = [np.std(test_errs[key]) for key in sorted_keys]
#     X = np.array([bytes_map[key] for key in sorted_keys])
#     Y = np.array(mean_errs)
#     Sigma = np.array(std_errs)
#     plt.errorbar(X, Y, yerr=Sigma, marker='o', linewidth=1, capsize=8, capthick=1, label=label)

# curves = read_results('results/jpg_quality/V1')
# test_errs = {}
# for test_name in curves:
#     quality = int(test_name.split('_')[0])
#     run = int(test_name.split('_')[1])
#     test_err = curves[test_name][0]['test_err']
#     if quality not in test_errs:
#         test_errs[quality] = []
#     test_errs[quality].append(test_err)
# plot_test_errs(test_errs, 'JPEG', jpg_bytes)

# curves = read_results('results/bottlefit_compression/V1/block_2')
# test_errs_q = {}
# test_errs_nq = {}
# best_runs = {}
# for test_name in curves:
#     compression = int(test_name.split('_')[0])
#     quantize = test_name.split('_')[1] == 'True'
#     run = int(test_name.split('_')[2])
#     test_err = curves[test_name][0]['test_err']
#     test_errs = test_errs_nq
#     if quantize:
#         test_errs = test_errs_q
#     else:
#         if compression not in best_runs:
#             best_runs[compression] = [run, test_err]
#         if test_err < best_runs[compression][1]:
#             best_runs[compression] = [run, test_err]
#     if compression not in test_errs:
#         test_errs[compression] = []
#     test_errs[compression].append(test_err)
# plot_test_errs(test_errs_q, 'CNN_quantized', sc_bytes_quantized)
# plot_test_errs(test_errs_nq, 'CNN_nonquantized', sc_bytes)

# plt.xlabel('Memory[kb]')
# plt.ylabel('Depth Error [MAPE]')
# plt.legend()
# plt.show()

# print(best_runs)

