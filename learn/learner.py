import sys

import pandas as pd
# import torch

from learn.agent import Agent
from learn.exploration import EpsilonGreedyExplorer, UpperConfidenceBoundExplorer, ThompsonSamplingExplorer
from learn.utils import log, get_utc_timestamp, utc_to_datetime_string, get_file_suffix_map, \
    fetch_explore_config_requirement, FileHandler, to_json, load_json
import os
from pandas import read_parquet, read_csv


def __get_reward_class__(class_name: str):
    """
    This private method is used to get reward class from the reward class name string.
    :param class_name: Name of the reward class.
    :return: Class type of the associated reward class.
    """
    return getattr(sys.modules['learn.reward'], class_name)


class Learner:
    def __init__(self, config: dict, mode="online", ask_update=True, reset=False):
        """
        Initialize the learner with configuration file.
        :param config: Dictionary of the configuration.
        """
        assert config.keys().__contains__('transition') and config['transition'] is not None
        log("Start initializing the learner...")

        self.mode = mode

        self.config = config
        self.task_name = self.config['task_name']
        self.config['history'] = None
        self.config['knowledge'] = None
        self.agent = None
        self.need_update = True
        if self.mode == "offline":
            log("Running offline mode. Will append result to existing history..")
            if ask_update:
                need_update = input("Do you want to upload the offline learning result? y/n")
                if need_update == "y":
                    self.need_update = True
                else:
                    self.need_update = False
        suffix_map = get_file_suffix_map(self.config['exploration'])
        self.file_handler = FileHandler(self.task_name, suffix_map)
        self.setup_history()

        if not self.config.__contains__("initial_state"):
            self.config["initial_state"] = list(config["states"].keys())[0]

        self.config["initial_state"] = list(config["states"].keys())[0]
        self.agent = Agent(config)
        self.explore_config = {}
        self.explorer = None

        self.reward_name = config['reward']
        self.reward_class = __get_reward_class__(self.reward_name)
        self.reward = self.reward_class(self.config['env_attr'])
        self.config['utc'] = get_utc_timestamp()
        self.config['datetime_string'] = utc_to_datetime_string(self.config['utc'])

        log("Finished initializing Learner")

    def setup_history(self):

        if not os.path.exists(self.file_handler.get_local_path("action_hist")):
            print("Folder not exist, initialize with None")
            self.config['history'] = None
            self.config['Knowledge'] = None
            self.config['update_content'] = None
        else:
            self.config['history'] = read_csv(self.file_handler.get_local_path("action_hist"))
            self.config['knowledge'] = load_json(self.file_handler.get_local_path("knowledge"))
            self.config['update_content'] = read_csv(self.file_handler.get_local_path("update_content"))

    def setup_explorer(self):
        """
        Set up the required explorer according to the explorer name from config file.
        """
        self.explore_config['datetime_string'] = self.config['datetime_string']
        self.explore_config = self.config['explore_config']
        assert fetch_explore_config_requirement(self.config['exploration']).issubset(set(self.explore_config.keys()))
        if self.config['exploration'] == 'epsilon_greedy':
            self.explorer = EpsilonGreedyExplorer(agent=self.agent, explore_config=self.explore_config)

        if self.config['exploration'] == 'upper_confidence_bound':
            self.explorer = UpperConfidenceBoundExplorer(agent=self.agent, explore_config=self.explore_config)

        if self.config['exploration'] == 'thompson_sampling':
            self.explorer = ThompsonSamplingExplorer(agent=self.agent, explore_config=self.explore_config)

        if self.explorer is None:
            sys.exit('Exploration method is not initialized.')

    def step(self, curr_env):
        # One row of env_data
        self.agent = Agent(self.config)  # Reconfigure agent with new information
        self.setup_explorer()
        prev_state, prev_action, curr_state, curr_action, knowledge = \
            self.explorer.learn(agent=self.agent, reward=self.reward, env_data=curr_env)

        self.config['knowledge'] = knowledge
        self.config['utc'] = get_utc_timestamp()

        cols = ["timestamp", "prev_state", "prev_action", "curr_state", "curr_action"] + self.config['env_attr'] \
               + ['reward']
        data = [self.config['utc'], prev_state, prev_action, curr_state, curr_action] + \
               curr_env[self.config['env_attr']].to_list() + [self.reward.get_reward(curr_env)]
        to_append = pd.DataFrame([data], columns=cols)

        if self.config['history'] is None:
            self.config['history'] = to_append
        else:
            self.config['history'] = pd.concat([self.config['history'], to_append])

        next_state = self.agent.get_next_state(curr_state, curr_action)
        cols = ["timestamp", "curr_state", "curr_action", "next_state"] + self.config['state_attr']
        data = [self.config['utc'], curr_state, curr_action, next_state] + \
               [self.config['states'][next_state][param] for param in self.config['state_attr']]
        to_append = pd.DataFrame([data], columns=cols)
        if self.config["update_content"] is None:
            self.config["update_content"] = to_append
        else:
            self.config["update_content"] = pd.concat([self.config["update_content"], to_append])

    # def read_local_model_knowledge(self):
    #     if self.config['exploration'] == 'deep_q':
    #         model = Model(num_embedding=len(self.config['env_attr']), num_action=len(self.config['actions']))
    #         model.load_state_dict(torch.load(self.file_handler.get_local_path("knowledge")))
    #         model.eval()
    #         self.config['knowledge'] = model
    #     else:
    #         knowledge = read_parquet(self.file_handler.get_local_path("knowledge_parquet")).to_dict()
    #         self.config['knowledge'] = knowledge
    #
    # def save_knowledge_to_local(self):
    #     if self.config['exploration'] == 'deep_q':
    #         torch.save(self.config['knowledge'].state_dict(), self.file_handler.get_local_path('knowledge'))
    #     else:
    #         self.config['knowledge'].to_parquet(self.file_handler.get_local_path("knowledge_parquet"))

    def run_learning(self, env_data):
        """
        Run one step of learning.
        param env_data: Aggregated/Extracted environment representation of a learning problem.
        """
        # TODO: Error Recover machenism
        assert len(env_data) == 1 or self.mode != "online"
        log("Learning Start....")
        count = 0
        for i in range(len(env_data)):
            count += 1
            curr_env = env_data.iloc[i]
            self.step(curr_env)
        self.config['knowledge'] = pd.DataFrame.from_dict(self.config['knowledge'])
        log(f"Learning_finish with {count} iterations")

        assert self.config["update_content"] is not None

        self.config['history'].to_csv(self.file_handler.get_local_path("action_hist"), index=False)
        to_json(self.file_handler.get_local_path("knowledge"), self.config['knowledge'].to_dict())
        self.config['update_content'].to_csv(self.file_handler.get_local_path("update_content"), index=False)
