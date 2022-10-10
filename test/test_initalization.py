import pytest
import pandas as pd
from learn.learner import Learner
import json
import os
from zapp.hadoop import HdfsClient
from learn.test_util import get_config

test_config = get_config()

hue_user = "hk-vulcan-svc"
credential_location = test_config['server_credential']
hadoop_table_folder = f"/user/hk-vulcan-svc/data_science/online_learning_platform/{test_config['task_name']}"
action_history_name = "action_hist_parquet.parquet"
knowledge_table_name = "knowledge_parquet.parquet"
action_history = os.path.join(hadoop_table_folder, action_history_name)
knowledge_table = os.path.join(hadoop_table_folder, knowledge_table_name)
hdfs_client = HdfsClient(user_name=hue_user, hadoop_cluster="sgp", path_to_creds=credential_location)

sample_action_hist = {"prev_state": [None], "prev_action": [None], "curr_state": ["s1"],
                      "curr_action": ["a"], "e1": [1], "e2": [2], "e3": [3], "reward": [1]}
sample_knowledge = {"s1": {'a': 1}}

sample_action_df = pd.DataFrame.from_dict(sample_action_hist).to_parquet("/tmp/sample_action.parquet")
sample_knowledge_df = pd.DataFrame.from_dict(sample_knowledge) \
    .to_parquet("/tmp/sample_knowledge.parquet")
local_temp_hist_path = f"/tmp/{test_config['task_name']}/action_hist_parquet.parquet"
local_temp_knowledge_path = f"/tmp/{test_config['task_name']}/knowledge_parquet.parquet"


class TestInitialization:
    def test_first_time_init(self):
        """
        The directory should exists in the server
        """
        if hdfs_client.exists_file_dir(hadoop_table_folder):
            hdfs_client.delete(hadoop_table_folder, True)
        learner = Learner(test_config)

        assert hdfs_client.exists_file_dir(hadoop_table_folder)
        hdfs_client.delete(hadoop_table_folder, True)

    def test_init_with_existing_file(self):
        """History should be pulled if it exists in the system"""

        if not hdfs_client.exists_file_dir(hadoop_table_folder):
            hdfs_client.make_dir(hadoop_table_folder)

        if not hdfs_client.exists_file_dir(action_history):
            hdfs_client.upload_file("/tmp/sample_action.parquet", action_history)
        if not hdfs_client.exists_file_dir(knowledge_table):
            hdfs_client.upload_file("/tmp/sample_knowledge.parquet", knowledge_table)

        learner = Learner(test_config)

        assert os.path.exists(local_temp_hist_path) and os.path.exists(local_temp_knowledge_path)
