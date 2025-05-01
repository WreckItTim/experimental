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
conda_environment = 'exp'

working_directory = f'{root_dir}'
python_file_name = f'reinforcement_learning/navi_train.py'

version = 'V1'
all_jobs = []
for random_seed in range(8):
    output_dir = f'models/AirSimNH_navi_2d_truths/{version}/seed_{random_seed}/'
    command_line_arguments = {
        'random_seed':random_seed,
        'output_dir':output_dir,
        'continue_training':True,
    }
    exclude_servers = []
    job = {
        'working_directory':working_directory,
        'command_line_arguments':command_line_arguments,
        'python_file_name':python_file_name,
        'conda_environment':conda_environment,
        'exlude_servers':exclude_servers,
    }
    all_jobs.append(job)

device_map = {
    #'hephaestus':{'cuda:0':1, 'cpu':0}, # 24 gb VRAM, 126 gb RAM
    'magma':{'cuda:0':1, 'cuda:1':1, 'cpu':0}, # 24 gb VRAM, 125 gb RAM
    #'ace':{'cuda:0':1, 'cuda:1':1, 'cuda:2':0, 'cpu':0}, # 11 gb VRAM, 125 gb RAM
    #'pyro':{'cuda:0':1, 'cpu':0}, # 11 gb VRAM, 62 gb RAM
    'phoenix':{'cuda:0':1, 'cpu':0}, # 6 gb VRAM, 62 gb RAM
    'torch':{'cuda:0':1, 'cpu':0}, # 6 gb VRAM, 62 gb RAM
    #'fox':{'cuda:0':1, 'cpu':0}, # 4 gb VRAM, 15 gb RAM
    #'apollo':{'cuda:0':1, 'cpu':0}, # 4 gb VRAM, 15 gb RAM
    #'flareon':{'cuda:0':1, 'cpu':0}, # 4 gb VRAM, 15 gb RAM
    #'ifrit':{'cuda:0':1, 'cpu':0}, # 3 gb VRAM, 31 gb RAM
}

delay = 60
overwrite = True 
if not overwrite and os.path.exists(f'{manager_dir}{manager_name}'):
    manager = Manager.load(manager_name, all_jobs=all_jobs, device_map=device_map)
else:
    manager = Manager(manager_name, all_jobs, device_map,
                 active_jobs=None, completed_jobs=None, job_map=None, job_info=None)

manager.run(delay)
print('all jobs complete!')
