import os
root_dir = '/home/tim/Dropbox/experimental/' # your path here where to parent directory where repos are
os.chdir(root_dir)
import sys
sys.path.append(root_dir)
from overseer.overseer_methods import *
import math

file_path = __file__
file_name = os.path.basename(file_path)
manager_name = file_name.split('.')[0]

overseer_dir = f'{home_dir}overseer/'
manager_dir = f'{overseer_dir}managers/'
conda_environment = 'airsim'

working_directory = f'{root_dir}'
python_file_name = f'reinforcement_learning/_data.py'

split_x = 32
split_y = 32
split_z = 4

map_name = 'AirSimNH'
sensor_name = 'SceneV1'

if map_name in ['AirSimNH']:
    xmin0 = -240
    xmax0 = 242
    ymin0 = -240
    ymax0 = 242
elif map_name in ['Blocks']:
    xmin0 = -140
    xmax0 = 142
    ymin0 = -120
    ymax0 = 102
xint = 2
yint = 2
zmin0 = 4
zmax0 = 8
zint = 4
id_name = 'alpha'

n_x = math.ceil((xmax0 - xmin0) / split_x)
n_y = math.ceil((ymax0 - ymin0) / split_y)
n_z = math.ceil((zmax0 - zmin0) / split_z)
all_jobs = []
for i_x in range(n_x):
    for i_y in range(n_y):
        for i_z in range(n_z):
            xmin = xmin0 + i_x*split_x
            xmax = min(xmax0, xmin + split_x)
            ymin = ymin0 + i_y*split_y
            ymax = min(ymax0, ymin + split_y)
            zmin = zmin0 + i_z*split_z
            zmax = min(zmax0, zmin + split_z)
            output_dir = f'map_data/observations/{sensor_name}/{map_name}/' # write run files to this directory
            part_name = f'{id_name}_{xmin}_{xmax}_{xint}_{ymin}_{ymax}_{yint}_{zmin}_{zmax}_{zint}'
            complete_path = f'{output_dir}completed__{part_name}.p'
            if os.path.exists(complete_path):
                continue
            command_line_arguments = {
                'map_name':map_name,
                'sensor_name':sensor_name,
                'xmin':xmin,
                'xmax':xmax,
                'xint':xint,
                'ymin':ymin,
                'ymax':ymax,
                'yint':yint,
                'zmin':zmin,
                'zmax':zmax,
                'zint':zint,
            }
            job = {
                'working_directory':working_directory,
                'command_line_arguments':command_line_arguments,
                'python_file_name':python_file_name,
                'conda_environment':conda_environment,
            }
            all_jobs.append(job)

device_map = {
    'hephaestus':{'cuda:0':1, 'cpu':0},
    'magma':{'cuda:0':0, 'cuda:1':1, 'cpu':0},
    'ace':{'cuda:0':0, 'cuda:1':0, 'cuda:2':1, 'cpu':0},
    'pyro':{'cuda:0':1, 'cpu':0},
    'phoenix':{'cuda:0':1, 'cpu':0},
    'torch':{'cuda:0':1, 'cpu':0},
    'fox':{'cuda:0':1, 'cpu':0},
    'apollo':{'cuda:0':1, 'cpu':0},
    'flareon':{'cuda:0':1, 'cpu':0},
    #'ifrit':{'cuda:0':1, 'cpu':0},
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
