import pytest
import pandas as pd
from learn.learner import Learner
import json
import os
from learn.test_util import get_algo_config
import shutil

test_algo_config = get_algo_config()
hue_user = "hk-vulcan-svc"
credential_location = test_algo_config['server_credential']
res_table_folder = f"./learning_out/{test_algo_config['task_name']}"
action_history_name = "action_hist.csv"
knowledge_table_name = "knowledge.json"
action_history = os.path.join(res_table_folder, action_history_name)
knowledge_table = os.path.join(res_table_folder, knowledge_table_name)


class TestAlgorithm:
    def test_algo_loop(self):
        if os.path.exists('./learning_out'):
            shutil.rmtree('./learning_out')

        curr_env = pd.DataFrame({'p1': [1], 'p2': [0], 'p3': [0]})
        for i in range(2):
            learner = Learner(test_algo_config)
            learner.run_learning(curr_env)
            update_df = pd.read_csv('./learning_out/update_content.csv')
            curr_env = update_df.iloc[[-1]]

        assert True
