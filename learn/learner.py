import sys

import pandas as pd

from learn.agent import Agent
from learn.exploration import EpsilonGreedyExplorer, UpperConfidenceBoundExplorer, ThompsonSamplingExplorer
from learn.utils import log, get_utc_timestamp, utc_to_datetime_string, pull_file_with_retry, update_file_with_retry
import os
from pandas import read_parquet
import random
from zapp.hadoop.impala_client import ImpalaClient
from zapp.hadoop import HdfsClient


def __get_reward_class__(class_name: str):
    """
    This private method is used to get reward class from the reward class name string.
    :param class_name: Name of the reward class.
    :return: Class type of the associated reward class.
    """
    return getattr(sys.modules['learn.reward'], class_name)


class Learner:
    def __init__(self, config: dict):
        """
        Initialize the learner with configuration file.
        :param config: Dictionary of the configuration.
        """
        hue_user = "hk-vulcan-svc"
        credential_location = config['server_credential']
        self.hadoop_table_folder = f"/user/hk-vulcan-svc/data_science/online_learning_platform/{config['task_name']}"
        self.action_history_name = "action_hist_parquet.parquet"
        self.knowledge_table_name = "knowledge_parquet.parquet"
        self.update_content = "update_content.parquet"
        self.action_history = os.path.join(self.hadoop_table_folder, self.action_history_name)
        self.knowledge_table = os.path.join(self.hadoop_table_folder, self.knowledge_table_name)
        self.update_content = os.path.join(self.hadoop_table_folder, self.update_content)
        self.hdfs_client = HdfsClient(user_name=hue_user, hadoop_cluster="sgp", path_to_creds=credential_location)
        log("Start initializing the learner...")
        self.config = config
        self.task_name = self.config['task_name']
        self.local_temp_hist_path = f'/tmp/{self.task_name}/action_hist_parquet.parquet'
        self.local_temp_knowledge_path = f'/tmp/{self.task_name}/knowledge_parquet.parquet'
        self.local_temp_content_path = f'/tmp/{self.task_name}/update_content.parquet'
        self.local_dir = f'/tmp/{self.task_name}'
        if not os.path.exists(self.local_dir):
            os.mkdir(self.local_dir)

        self.setup_history()

        self.config["initial_state"] = list(config["states"].keys())[0]
        self.agent = Agent(config)
        self.explore_config = {}
        self.explorer = None

        self.reward_name = config['reward']
        self.reward_class = __get_reward_class__(self.reward_name)
        self.reward = self.reward_class(self.agent.get_env_attr())
        self.config['utc'] = get_utc_timestamp()
        self.config['datetime_string'] = utc_to_datetime_string(self.config['utc'])

        self.setup_explorer()
        log("Finished initializing Learner")

    def setup_history(self):

        # Check whether initial run

        if not self.hdfs_client.exists_file_dir(self.hadoop_table_folder):
            log("Folder not found creating one...")
            self.hdfs_client.make_dir(self.hadoop_table_folder)
            self.config['history'] = None
            self.config['knowledge'] = None
            # Initialization handled by exploration module
            log("Created folder for the task")
        else:
            # Download action table to local temp folder
            log("Fetching the history....")
            action_hist_status, knowledge_status = 0, 0
            if self.hdfs_client.exists_file_dir(self.action_history):
                action_hist_status = pull_file_with_retry(self.action_history, self.local_temp_hist_path, 10
                                                          , self.hdfs_client)

            # Download current knowledge information
            if self.hdfs_client.exists_file_dir(self.knowledge_table):
                knowledge_status = pull_file_with_retry(self.knowledge_table, self.local_temp_knowledge_path, 10,
                                                        self.hdfs_client)

            if not action_hist_status or not knowledge_status:
                print("Failed to pull the history, training aborted")
                sys.exit()

            self.config['history'] = read_parquet(self.local_temp_hist_path)
            self.config['knowledge'] = read_parquet(self.local_temp_knowledge_path).to_dict()
            log("Fetched the history.")

    def setup_explorer(self):
        """
        Set up the required explorer according to the explorer name from config file.
        """
        self.explore_config['datetime_string'] = self.config['datetime_string']
        if self.config['exploration'] == 'epsilon_greedy':
            self.explore_config['epsilon'] = self.config['explore_config']['epsilon']
            self.explore_config['learning_rate'] = self.config['explore_config']['learning_rate']
            self.explore_config['discount'] = self.config['explore_config']['discount']
            self.explorer = EpsilonGreedyExplorer(agent=self.agent, explore_config=self.explore_config)

        if self.config['exploration'] == 'upper_confidence_bound':
            self.explore_config['epsilon'] = self.config['explore_config']['epsilon']
            self.explore_config['learning_rate'] = self.config['explore_config']['learning_rate']
            self.explore_config['discount'] = self.config['explore_config']['discount']
            self.explore_config['confidence_bound'] = self.config['explore_config']['confidence_bound']
            self.explorer = UpperConfidenceBoundExplorer(agent=self.agent, explore_config=self.explore_config)

        if self.config['exploration'] == 'thompson_sampling':
            self.explore_config['distribution'] = self.config['explore_config']['distribution']
            self.explorer = ThompsonSamplingExplorer(agent=self.agent, explore_config=self.explore_config)

        if self.explorer is None:
            sys.exit('Exploration method is not initialized.')

    def run_learning(self, env_data):
        """
        Run one step of learning.
        param env_data: Aggregated/Extracted environment representation of a learning problem.
        """
        # TODO: Error Recover machenism
        log("Learning Start....")
        prev_state, prev_action, curr_state, curr_action, knowledge = \
            self.explorer.learn(agent=self.agent, reward=self.reward, env_data=env_data)
        log("Learning Finished.")

        assert len(env_data) == 1
        cols = ["timestamp", "prev_state", "prev_action", "curr_state", "curr_action"] + self.config['env_attr'] \
               + ['reward']
        data = [self.config['utc'], prev_state, prev_action, curr_state, curr_action] + \
               env_data.iloc[0][self.config['env_attr']].to_list() + [self.reward.get_reward(env_data)]
        to_append = pd.DataFrame([data], columns=cols)

        if self.config['history'] is None:
            self.config['history'] = to_append
        else:
            self.config['history'] = pd.concat([self.config['history'], to_append])

        self.config['history'].to_parquet(self.local_temp_hist_path)

        self.config['knowledge'] = pd.DataFrame.from_dict(knowledge)
        self.config['knowledge'].to_parquet(self.local_temp_knowledge_path)

        log("Uploading new content...")
        next_state = self.agent.get_next_state(curr_state, curr_action)
        cols = ["timestamp", "curr_state", "curr_action", "next_state"] + self.config['state_attr']
        data = [self.config['utc'], curr_state, curr_action, next_state] + \
               [self.config['states'][next_state][param] for param in self.config['state_attr']]
        to_append = pd.DataFrame([data], columns=cols)

        if not self.hdfs_client.exists_file_dir(self.update_content):
            new_update_content = to_append
        else:
            self.hdfs_client.download_file(self.update_content, self.local_temp_content_path)
            prev_content_data = pd.read_parquet(self.local_temp_content_path)
            new_update_content = pd.concat([prev_content_data, to_append])

        new_update_content.to_parquet(self.local_temp_content_path)
        res = update_file_with_retry(self.local_temp_content_path, self.update_content, 10, self.hdfs_client)
        assert res == 1
        log("Uploaded new content")

        log("Uploading the new history information....")
        res1 = update_file_with_retry(self.local_temp_hist_path, self.action_history, 10, self.hdfs_client)
        res2 = update_file_with_retry(self.local_temp_knowledge_path, self.knowledge_table, 10, self.hdfs_client)
        assert res1 == 1 and res2 == 1
        log("Update uploaded.")
