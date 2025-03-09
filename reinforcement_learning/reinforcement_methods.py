import sys
home_dir = '/home/tim/Dropbox/' # home directory of repo
local_dir = '/home/tim/local/' # directory to read/write local files from
sys.path.append(home_dir)
from global_methods import *

# read_directory is the folder with trained model output
# eval_name will create sub_folder within read_directory with all output from evaluations
def evaluate_navi(config_path, model_path, working_directory):
	os.system(f'python _test.py {config_path} {model_path} {working_directory}')
	while not os.path.exists(f'{working_directory}evaluation.json'):
		time.sleep(5)