import os
root_dir = '/home/tim/Dropbox/experimental/' # your path here where to parent directory where repos are
local_dir = '/home/tim/local/'
os.chdir(root_dir)
import sys
sys.path.append(root_dir)
import map_data.map_methods as mm
import shortest_path.shortest_methods as sm
import utils.global_methods as gm
import numpy as np
import random
import matplotlib.pyplot as plt
from IPython.display import clear_output
import psutil
import gc
import pickle
import shutil
import copy

pdir = 'map_data/observations/'
for fname in os.listdir(pdir):
    if 'V' in fname and fname in ['SceneV1']:
        print(fname)
        for fname2 in os.listdir(f'{pdir}{fname}/AirSimNH/'):
            old_path = f'{pdir}{fname}/AirSimNH/{fname2}'
            if fname2 in ['.ipynb_checkpoints']:
                shutil.rmtree(old_path)
                continue
            if 'conflicted' in fname2 or 'complete' in fname2:
                os.remove(old_path)
                continue
            f_pre = fname2.split('__')[0]
            if f_pre in ['log', 'point_list']:
                os.remove(old_path)
                continue
            if '.' not in fname2:
                if fname2[-1] == 'p':
                    f_post = 'p'
                    fname2 = fname2[:-1]
                elif fname2[-4:] == 'json':
                    f_post = 'json'
                    fname2 = fname2[:-4]
            else:
                f_post = fname2.split('.')[1]
            part_name = fname2.split('__')[1].split('.')[0]
            parts = part_name.split('_')
            this_id_name = parts[0]
            xmin, xmax, xint, ymin, ymax, yint, zmin, zmax, zint = [int(part) for part in parts[1:]] 
            zmin, zmax, zint = 0, 4, 4
            parts = [str(component) for component in [xmin, xmax, xint, ymin, ymax, yint, zmin, zmax, zint]]
            new_part_name = f'{this_id_name}_' + '_'.join(parts)
            new_fname2 = f'{f_pre}__{new_part_name}.{f_post}'
            #print(old_path, f'{pdir}{fname}/AirSimNH/{new_fname2}')
            #i = input()
            new_path = f'{pdir}{fname}/AirSimNH/{new_fname2}'
            os.rename(old_path, new_path)

            if f_pre in ['data_dict']:
                new_data_dict = {}
                try:
                    data_dict = gm.pk_read(new_path)
                    # for x in data_dict:
                    #     for y in data_dict[x]:
                    #         if y not in new_data_dict:
                    #             new_data_dict[y] = {}
                    #         if x not in new_data_dict[y]:
                    #             new_data_dict[y][x] = {}
                    #         if -4 in data_dict[x][y]:
                    #             new_data_dict[y][x][4] = data_dict[x][y][-4]
                    #         else:
                    #             new_data_dict[y][x][4] = data_dict[x][y][4]
                    for x in data_dict:
                        if x not in new_data_dict:
                            new_data_dict[x] = {}
                        for y in data_dict[x]:
                            if y not in new_data_dict[x]:
                                new_data_dict[x][y] = {}
                            new_data_dict[y][x][4] = data_dict[x][y][4]()
                    gm.pk_write(new_data_dict, new_path)
                except:
                    os.remove(new_path)