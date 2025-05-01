import os
root_dir = '/home/tim/Dropbox/experimental/' # your path here where to parent directory where repos are
os.chdir(root_dir)
import sys
sys.path.append(root_dir)
from overseer.overseer_methods import *

file_path = __file__
file_name = os.path.basename(file_path)
manager_name = file_name.split('.')[0]

overseer_dir = f'{root_dir}overseer/'
manager_dir = f'{overseer_dir}managers/'
conda_environment = None

working_directory = f'{root_dir}'
python_file_name = f'{root_dir}experiments/monocular_depth/train_eval.py'

version_dict = {
    8:'scale_8', 
    4:'scale_4', 
    2:'scale_2', 
    1:'scale_1', 
    0.5:'scale_0d5', 
    #0.25:'scale_0d25',
    #0.125:'scale_0d125',
}
all_jobs = []
for scale in version_dict:
    for use_slim_soft in [False, True]:
        for random_seed in range(8):
            if use_slim_soft:
                version = 'V4/' + version_dict[scale]
            else:
                version = 'V3/' + version_dict[scale]
            run_dir = f'models/monocular_depth/{version}/seed_{random_seed}/'
            if os.path.exists(f'{run_dir}r2s.p'):
                continue
            command_line_arguments = {
                'version':version,
                'scale':scale,
                'random_seed':random_seed,
                'use_slim_cnn':True,
                'use_slim_train':True,
                'use_slim_soft':use_slim_soft,
            }
            exclude_servers = ['fox', 'apollo', 'flareon'] ## not enough ram
            if scale < 8:
                exclude_servers.append('magma') # save for big models
                exclude_servers.append('hephaestus') # save for big models
            if scale >= 8:
                exclude_servers.append('ace') # not enough vram
                exclude_servers.append('pyro') # not enough vram
            if scale >= 4:
                exclude_servers.append('phoenix') # not enough vram
                exclude_servers.append('torch') # not enough vram
            job = {
                'working_directory':working_directory,
                'command_line_arguments':command_line_arguments,
                'python_file_name':python_file_name,
                'conda_environment':conda_environment,
                'exlude_servers':exclude_servers,
            }
            all_jobs.append(job)

device_map = {
    'magma':{'cuda:0':0, 'cuda:1':1, 'cpu':0}, # 24 gb VRAM, 125 gb RAM
    #'hephaestus':{'cuda:0':1, 'cpu':0}, # 24 gb VRAM, 126 gb RAM
    #'ace':{'cuda:0':0, 'cuda:1':1, 'cuda:2':0, 'cpu':0}, # 11 gb VRAM, 125 gb RAM
    'pyro':{'cuda:0':1, 'cpu':0}, # 11 gb VRAM, 62 gb RAM
    'phoenix':{'cuda:0':1, 'cpu':0}, # 6 gb VRAM, 62 gb RAM
    'torch':{'cuda:0':1, 'cpu':0}, # 6 gb VRAM, 62 gb RAM
    #'fox':{'cuda:0':1, 'cpu':0}, # 4 gb VRAM, 15 gb RAM
    #'apollo':{'cuda:0':1, 'cpu':0}, # 4 gb VRAM, 15 gb RAM
    #'flareon':{'cuda:0':1, 'cpu':0}, # 4 gb VRAM, 15 gb RAM
    #'ifrit':{'cuda:0':1, 'cpu':0}, # 3 gb VRAM, 31 gb RAM
}

delay = 10
overwrite = True 
if not overwrite and os.path.exists(f'{manager_dir}{manager_name}'):
    manager = Manager.load(manager_name, all_jobs=all_jobs, device_map=device_map)
else:
    manager = Manager(manager_name, all_jobs, device_map,
                 active_jobs=None, completed_jobs=None, job_map=None, job_info=None)

manager.run(delay)
print('all jobs complete!')
