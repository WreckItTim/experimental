
#import utils.global_methods as gm # sys.path.append('path/to/parent/repo/')
import os
import time
import utils.global_methods as gm
import numpy as np
import json

# read_directory is the folder with trained model output
# eval_name will create sub_folder within read_directory with all output from evaluations
# 'output_dir' must be in command_arguments
def evaluate_navi(command_arguments):
	args_str = gm.args_to_str(command_arguments)
	os.system(f'python3 reinforcement_learning/navi_eval.py {args_str}') # TODO: change back to reinforcement_learning/...
	output_dir = command_arguments['output_dir']
	while not os.path.exists(f'{output_dir}evaluation.json'):
		time.sleep(5)
	evaluation = gm.read_json(f'{output_dir}evaluation.json')
	accuracy = np.mean(evaluation['successes'])
	return accuracy

# json files output with all string key names
# process so that the evaluation dictionary structure is such:
    # episode # - int
        # step # - int
            # state - dictionary of misc key, value pairs for that state
def process_episodes(json_evaluation):
    nEpisodes = len(json_evaluation)
    episodes = [None] * nEpisodes
    episode_idx = 0
    for episode_str in json_evaluation:
        if 'episode_' not in episode_str:
            continue
        json_episode = json_evaluation[episode_str]
        nSteps = len(json_episode)
        states = [None] * nSteps
        for step_str in json_episode:
            step_num = int(step_str.split('_')[1])
            state = json_episode[step_str]
            states[step_num] = state
        episodes[episode_idx] = states
        episode_idx += 1
    return episodes
def read_evaluations(evaluation_folder):
    evaluation_files = [file for file in os.listdir(evaluation_folder) if 'states' in file]
    nEvaluations = len(evaluation_files)
    evaluations = [None] * nEvaluations
    for evaluation_file in evaluation_files:
        if '.json' not in evaluation_file:
            continue
        epoch = int(evaluation_file.split('.')[0].split('_')[-1])
        print(evaluation_file, epoch)
        json_evaluation = json.load(open(evaluation_folder + evaluation_file, 'r'))
        episodes = process_episodes(json_evaluation)
        evaluations[epoch] = episodes
    return evaluations
# architecture for evaluations:
# evaluations - list of episodes (indexed of evaluation number) - 0 idx is first evaluation
    # episodes - list of states (indexed by step number)
        # states - dict of (key, value) pairs for state at all_evaluations[instance][evaluation][episode][step]