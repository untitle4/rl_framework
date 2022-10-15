import pandas as pd
from learn.learner import Learner
import json
import os
import time
from learn.test_util import get_algo_config
import shutil

def test_script():
    if os.path.exists('./learning_out'):
        shutil.rmtree('./learning_out')

    task_name = ['test_epsilon_greedy', 'test_thompson_sampling', 'test_upper_confidence_bound']
    task_name = ['test_thompson_sampling']
    for task in task_name[:]:

        test_algo_config = get_algo_config()
        test_algo_config['task_name'] = task
        test_algo_config['exploration'] = task[5:]
        print(task, task[5:])

        res_table_folder = f"./learn_out/{task}"
        action_history_name = "action_hist_parquet.csv"
        knowledge_table_name = "knowledge_parquet.csv"
        action_history = os.path.join(res_table_folder, action_history_name)
        knowledge_table = os.path.join(res_table_folder, knowledge_table_name)

        curr_env = pd.DataFrame({'p1': [1], 'p2': [0], 'p3': [0]})
        for i in range(500):
            print(f'---------------------------------------------Step {i}'
                  f'-----------------------------------------------------')
            learner = Learner(test_algo_config)
            learner.run_learning(curr_env)
            update_df = pd.read_csv(f'./learning_out/{task}/update_content.csv')
            curr_env = update_df.iloc[[-1]]
