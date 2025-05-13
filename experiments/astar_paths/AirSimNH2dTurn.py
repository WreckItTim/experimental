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
python_file_name = f'{root_dir}experiments/astar_paths/find_paths.py'

all_jobs = []
astar_version = 'v1'
map_name = 'AirSimNH'
motion = '2d'
region = 'all'
delta_x = 32
delta_y = 0
for random_seed in range(400, 500):
    out_dir = f'map_data/astar_paths/{astar_version}/{map_name}/{motion}/{region}/{random_seed}/'
    if os.path.exists(f'{out_dir}complete.p'):
        continue
    command_line_arguments = {
        'astar_version':astar_version,
        'map_name':map_name,
        'motion':motion,
        'region':region,
        'random_seed':random_seed,
        'delta_x':delta_x,
        'delta_y':delta_y,
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
    # 'ace':{'cuda:0':0, 'cuda:1':0, 'cuda:2':0, 'cpu':12}, # 11 gb VRAM, 125 gb RAM
    #'hephaestus':{'cuda:0':0, 'cpu':6}, # 24 gb VRAM, 126 gb RAM
    'ifrit':{'cuda:0':0, 'cpu':2}, # 3 gb VRAM, 31 gb RAM
    #'magma':{'cuda:0':0, 'cuda:1':0, 'cpu':12}, # 24 gb VRAM, 125 gb RAM
    'pyro':{'cuda:0':0, 'cpu':6}, # 11 gb VRAM, 62 gb RAM
    #'phoenix':{'cuda:0':0, 'cpu':6}, # 6 gb VRAM, 62 gb RAM
    #'torch':{'cuda:0':0, 'cpu':6}, # 6 gb VRAM, 62 gb RAM
    #'fox':{'cuda:0':0, 'cpu':6}, # 4 gb VRAM, 15 gb RAM
    #'apollo':{'cuda:0':0, 'cpu':6}, # 4 gb VRAM, 15 gb RAM
    #'flareon':{'cuda:0':0, 'cpu':6}, # 4 gb VRAM, 15 gb RAM
}

delay = 10*60
overwrite = True 
if not overwrite and os.path.exists(f'{manager_dir}{manager_name}'):
    manager = Manager.load(manager_name, all_jobs=all_jobs, device_map=device_map)
else:
    manager = Manager(manager_name, all_jobs, device_map,
                 active_jobs=None, completed_jobs=None, job_map=None, job_info=None)

manager.run(delay)
print('all jobs complete!')
