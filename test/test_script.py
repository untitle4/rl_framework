import pandas as pd
from learn.learner import Learner
import json
import os
from zapp.hadoop import HdfsClient
import time
from learn.test_util import get_algo_config

task_name = ['test_epsilon_greedy', 'test_thompson_sampling', 'test_upper_confidence_bound']

for task in task_name[-1:]:

    test_algo_config = get_algo_config()
    test_algo_config['task_name'] = task
    test_algo_config['exploration'] = task[5:]
    print(task, task[5:])

    hue_user = "hk-vulcan-svc"
    credential_location = test_algo_config['server_credential']
    hadoop_table_folder = f"/user/hk-vulcan-svc/data_science/online_learning_platform/{test_algo_config['task_name']}"
    action_history_name = "action_hist_parquet.parquet"
    knowledge_table_name = "knowledge_parquet.parquet"
    action_history = os.path.join(hadoop_table_folder, action_history_name)
    knowledge_table = os.path.join(hadoop_table_folder, knowledge_table_name)
    hdfs_client = HdfsClient(user_name=hue_user, hadoop_cluster="sgp", path_to_creds=credential_location)

    if hdfs_client.exists_file_dir(hadoop_table_folder):
        hdfs_client.delete(hadoop_table_folder, True)

    curr_env = pd.DataFrame({'p1': [1], 'p2': [0], 'p3': [0]})
    for i in range(50):
        print(f'---------------------------------------------Step {i}'
              f'-----------------------------------------------------')
        learner = Learner(test_algo_config)
        learner.run_learning(curr_env)
        update_content_path = learner.update_content
        hdfs_client.download_file(update_content_path, '/tmp/test/update.parquet')
        update_df = pd.read_parquet('/tmp/test/update.parquet')
        curr_env = update_df.iloc[[-1]]
