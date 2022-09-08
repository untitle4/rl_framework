import sys

import pandas as pd

from learn.agent import Agent
from learn.exploration import EpsilonGreedyExplorer, UpperConfidenceBoundExplorer, ThompsonSamplingExplorer
from learn.utils import log, get_utc_timestamp, utc_to_datetime_string, pull_file_with_retry, update_file_with_retry
import os
from pandas import read_parquet
import random


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
        log("Start initializing the learner...")
        self.config = config
        self.task_name = self.config['task_name']
        self.local_temp_hist_path = f'/tmp/{self.task_name}/action_hist_parquet.parquet'
        self.local_temp_knowledge_path = f'/tmp/{self.task_name}/knowledge_parquet.parquet'
        self.local_temp_content_path = f'/tmp/{self.task_name}/update_content.parquet'
        self.local_dir = f'/tmp/{self.task_name}'
        if not os.path.exists(self.local_dir):
            os.mkdir(self.local_dir)

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

        # TODO: Update the system

