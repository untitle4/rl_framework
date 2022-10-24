from learn.learner import Learner
from tools.runner import run
from tools.updater import update
import pandas as pd
from tools.state_generator import get_state_transition
import yaml
import json
from learn.utils import FileHandler, get_file_suffix_map
import os
import shutil
from tools.env_generator import generate_env, generate_init_env
import argparse

total_iter = 2000

parser = argparse.ArgumentParser(description='Experiment Config')
parser.add_argument('--ename', dest='experiment_name', type=str)
parser.add_argument('--lconfig', dest='learning_config_path', type=str)
args = parser.parse_args()

experiment_path = args.experiment_name
learning_config = args.learning_config_path
# Initial configuration is should be in /runs/intersection
runner_config = './configs/config.yaml'
# learning_config = './configs/learning_config.json'

if not os.path.exists(experiment_path):
    os.mkdir(experiment_path)
    shutil.copytree(src='./runs/intersection', dst=f'./{experiment_path}/runs/intersection')

with open(runner_config, "r") as f:
    runner_config = yaml.safe_load(f)

with open(learning_config, "r") as f:
    learning_config = json.load(f)

runner_config['configs']['sumo_loc'] = f'{experiment_path}/runs/intersection'
# Project name won't change
learning_config['transition'], learning_config['states'] = get_state_transition()
# print(learning_config['states'])
# print(learning_config['transition'])
filehandler1 = FileHandler('intersection_0', get_file_suffix_map(learning_config['exploration']), experiment_path)
filehandler2 = FileHandler('intersection_1', get_file_suffix_map(learning_config['exploration']), experiment_path)
filehandler3 = FileHandler('intersection_2', get_file_suffix_map(learning_config['exploration']), experiment_path)
filehandler1.clear_local_file()
filehandler2.clear_local_file()
filehandler3.clear_local_file()
# run(runner_config) # Initial run

# shutil.copytree(src=f'{experiment_path}/runs/intersection',
#                 dst=f'{experiment_path}/runs/intersection_0', dirs_exist_ok=True)

for i in range(total_iter):
    print(f'---------------------------Learning start round {i}-----------------------------------------')
    # runner_config['configs']['sumo_loc'] += '_' + str(i)

    print('Running Simulation....')
    run(runner_config, f'{experiment_path}/runs/intersection/out.xml')
    print('Simulation ended.')

    if i != 0:
        env = generate_env(f'{experiment_path}/runs/intersection', filehandler1.get_local_path('action_hist'))
    else:
        env = generate_init_env()

    print('Fetched environment: ')
    # print(len(env))
    print(env)

    for j in range(3):
        learning_config['task_name'] = f'intersection_{j}'
        print(learning_config)
        learner = Learner(learning_config, experiment_folder=experiment_path)
        learner.run_learning(env)

    param = [{'duration': '33', 'state': 'GGGrrrGGGrrrrrr'},
        {'duration': '6', 'state': 'yyyrrryyyrrrrrr'},
        {'duration': '33', 'state': 'rrrGGGrrrrrrrrr'},
        {'duration': '6', 'state': 'rrryyyrrrrrrrrr'},
        {'duration': '33', 'state': 'rrrrrrrrrGGGGGG'},
        {'duration': '6', 'state': 'rrrrrrrrryyyyyy'}]

    param[0]['duration'] = str(pd.read_csv(filehandler1.get_local_path('update_content')).iloc[-1]['g'])
    param[2]['duration'] = str(pd.read_csv(filehandler2.get_local_path('update_content')).iloc[-1]['g'])
    param[4]['duration'] = str(pd.read_csv(filehandler3.get_local_path('update_content')).iloc[-1]['g'])

    print('Updated parameters:')
    print(param)

    # if not os.path.exists(f'./runs/intersection_{i + 1}'):
    #     os.mkdir(f'./runs/intersection_{i + 1}')
    # shutil.copytree(src=f'{experiment_path}/runs/intersection_{i}', dst=f'{experiment_path}/runs/intersection_{i+1}',
    #                 dirs_exist_ok=True)

    update(param_out=param, in_path=f'{experiment_path}/runs/intersection',
           out_path=f'{experiment_path}/runs/intersection')

    # Revert back
    # runner_config['configs']['sumo_loc'] = runner_config['configs']['sumo_loc'].rsplit('_', maxsplit=1)[0]

    print(f'Finished round {i}')

