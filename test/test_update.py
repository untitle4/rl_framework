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
update_content_name = "update_content.parquet"
action_history = os.path.join(hadoop_table_folder, action_history_name)
knowledge_table = os.path.join(hadoop_table_folder, knowledge_table_name)
update_content = os.path.join(hadoop_table_folder, update_content_name)
hdfs_client = HdfsClient(user_name=hue_user, hadoop_cluster="sgp", path_to_creds=credential_location)

sample_env = {"e1": [1], "e2": [2], "e3": [3]}
env_data = pd.DataFrame.from_dict(sample_env)


class TestUpload:
    def test_upload_after_learn_first_run(self):
        if hdfs_client.exists_file_dir(hadoop_table_folder):
            hdfs_client.delete(hadoop_table_folder, True)

        learner = Learner(test_config)
        learner.run_learning(env_data)

        assert hdfs_client.exists_file_dir(action_history) and hdfs_client.exists_file_dir(knowledge_table) and \
               hdfs_client.exists_file_dir(update_content)

        hdfs_client.delete(hadoop_table_folder, True)

    def test_upload_after_learn_existing(self):
        if hdfs_client.exists_file_dir(hadoop_table_folder):
            hdfs_client.delete(hadoop_table_folder, True)

        hdfs_client.make_dir(hadoop_table_folder)
        sample_content_update = {"timestamp": [0],
                                 "curr_state": ["s1"],
                                 "curr_action": ["a"],
                                 "next_state": ["s1"],
                                 "p1": 1,
                                 "p2": 1,
                                 "p3": 1}
        sample_history = {"timestamp": [0],
                          "prev_state": [None],
                          "prev_action": [None],
                          "curr_state": ["s1"],
                          "curr_action": ["a"],
                          "e1": 1,
                          "e2": 1,
                          "e3": 1,
                          "reward": 1}
        sample_knowledge = {"s1": {'a': 1}, "s2": {'b': 1}}

        pd.DataFrame.from_dict(sample_content_update) \
            .to_parquet(os.path.join('/tmp', update_content_name))
        pd.DataFrame.from_dict(sample_history).to_parquet(os.path.join('/tmp', action_history_name))
        pd.DataFrame.from_dict(sample_knowledge) \
            .to_parquet(os.path.join('/tmp', knowledge_table_name))

        hdfs_client.upload_file(os.path.join('/tmp', update_content_name), update_content)
        hdfs_client.upload_file(os.path.join('/tmp', action_history_name), action_history)
        hdfs_client.upload_file(os.path.join('/tmp', knowledge_table_name), knowledge_table)

        learner = Learner(test_config)
        learner.run_learning(env_data)

        hdfs_client.download_file(update_content, os.path.join('/tmp', update_content_name))
        hdfs_client.download_file(action_history, os.path.join('/tmp', action_history_name))
        hdfs_client.download_file(knowledge_table, os.path.join('/tmp', knowledge_table_name))

        update_df = pd.read_parquet(os.path.join('/tmp', update_content_name))
        history_df = pd.read_parquet(os.path.join('/tmp', action_history_name))

        assert len(update_df) == 2 and len(history_df) == 2

        hdfs_client.delete(hadoop_table_folder, True)
