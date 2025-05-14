import os
root_dir = '/home/tim/Dropbox/experimental/' # your path here where to parent directory where repos are
local_dir = '/home/tim/local/'
os.chdir(root_dir)
import sys
sys.path.append(root_dir)
import map_data.map_methods as mm
import utils.global_methods as gm
import skimage
import numpy as np
initial_locals = locals().copy() # will exclude these parameters from config parameters written to file

# read params from command line
job_name = 'null'
input_dir = 'null'
output_dir = 'null'
part_names = []
transform_name = 'null'
transform_params = 'null'
if len(sys.argv) > 1:
    arguments = gm.parse_arguments(sys.argv[1:])
    locals().update(arguments)

# set directory to write all results to
os.makedirs(output_dir, exist_ok=True)

# save all params to file
all_locals = locals()
new_locals = {k:v for k, v in all_locals.items() if (not k.startswith('__') and k not in initial_locals and k not in ['initial_locals','all_locals'])}
params = new_locals.copy()
gm.pk_write(params, f'{output_dir}params.p')

print('running job with params', params)
gm.set_global('local_dir', local_dir)
gm.progress(job_name, 'started')

for part_idx, part_name in enumerate(part_names):
    gm.progress(job_name, f'{part_idx/len(part_names):0.2f}%')
    in_path = f'{input_dir}data_dict__{part_name}.p'
    out_path = f'{output_dir}data_dict__{part_name}.p'
    if os.path.exists(out_path) or not os.path.exists(in_path):
        continue
    in_data = gm.pk_read(in_path)
    out_data = {}
    for x in in_data:
        out_data[x] = {}
        for y in in_data[x]:
            out_data[x][y] = {}
            for z in in_data[x][y]:
                out_data[x][y][z] = {}
                for d in in_data[x][y][z]:
                    in_X = in_data[x][y][z][d]
    
                    if transform_name == 'crop':
                        out_X = in_X[:, transform_params['u']:transform_params['d'], transform_params['l']:transform_params['r']]
                    if transform_name == 'reduce':
                        out_X = skimage.measure.block_reduce(in_X, (1, transform_params['reduce_magnitude'], transform_params['reduce_magnitude']), np.min)
                        out_X.shape
                    if transform_name == 'flatten':
                        out_X = in_X.flatten()
                    
                    out_data[x][y][z][d] = out_X
    gm.pk_write(out_data, out_path)

#gm.pk_write(True, f'{output_dir}completed__{part_name}.p')

gm.progress(job_name, 'complete')