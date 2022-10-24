from learn.learner import Learner
from tools.runner import run
from tools.updater import update
import pandas as pd
from tools.state_generator import get_compose_state_transition
import yaml
import json
from learn.utils import FileHandler, get_file_suffix_map
import os
import shutil
from tools.env_generator import generate_env, generate_init_env
import argparse

total_iter = 100000

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
learning_config['transition'], learning_config['states'] = get_compose_state_transition()
learning_config['state_attr'] = ['g1', 'g2', 'g3']

actions = ['K', 'I', 'D']
learning_config['actions'] = []
for a1 in actions:
    for a2 in actions:
        for a3 in actions:
            learning_config['actions'].append(a1+a2+a3)
# print(learning_config['states'])
# print(learning_config['transition'])
filehandler = FileHandler('intersection_compose', get_file_suffix_map(learning_config['exploration']), experiment_path)
filehandler.clear_local_file()
# run(runner_config) # Initial ru

# shutil.copytree(src=f'{experiment_path}/runs/intersection',
#                 dst=f'{experiment_path}/runs/intersection_0', dirs_exist_ok=True)

for i in range(total_iter):
    print(f'---------------------------Learning start round {i}-----------------------------------------')

    print('Running Simulation....')
    run(runner_config, f'{experiment_path}/runs/intersection/out.xml')
    print('Simulation ended.')

    if i != 0:
        env = generate_env(f'{experiment_path}/runs/intersection', filehandler.get_local_path('action_hist'))
    else:
        env = generate_init_env()

    print('Fetched environment: ')
    # print(len(env))
    print(env)

    learning_config['task_name'] = f'intersection_compose'
    # print(learning_config)
    learner = Learner(learning_config, experiment_folder=experiment_path)
    learner.run_learning(env)

    param = [{'duration': '33', 'state': 'GGGrrrGGGrrrrrr'},
        {'duration': '6', 'state': 'yyyrrryyyrrrrrr'},
        {'duration': '33', 'state': 'rrrGGGrrrrrrrrr'},
        {'duration': '6', 'state': 'rrryyyrrrrrrrrr'},
        {'duration': '33', 'state': 'rrrrrrrrrGGGGGG'},
        {'duration': '6', 'state': 'rrrrrrrrryyyyyy'}]
    target = pd.read_csv(filehandler.get_local_path('update_content')).iloc[-1]
    param[0]['duration'] = str(target['g1'])
    param[2]['duration'] = str(target['g2'])
    param[4]['duration'] = str(target['g3'])

    print('Updated parameters:')
    print(param)

    # if not os.path.exists(f'./runs/intersection_{i + 1}'):
    #     os.mkdir(f'./runs/intersection_{i + 1}')
    # shutil.copytree(src=f'{experiment_path}/runs/intersection', dst=f'{experiment_path}/runs/intersection',
    #                 dirs_exist_ok=True)

    update(param_out=param, in_path=f'{experiment_path}/runs/intersection',
           out_path=f'{experiment_path}/runs/intersection')

    # Revert back
    # runner_config['configs']['sumo_loc'] = runner_config['configs']['sumo_loc'].rsplit('_', maxsplit=1)[0]

    print(f'Finished round {i}')

