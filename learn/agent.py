import json
import os
import pandas as pd


class Agent:
    def __init__(self, config_dict):
        # TODO: Figure out what does the json file need to define along with the implementation

        self.states = config_dict['states']  # state should be defined as a list
        self.state_attr = config_dict['state_attr']  # Attribute of the state
        self.env_attr = config_dict['env_attr']  # Column name that used to calculate reward
        self.exploration_method = config_dict['exploration']  # Should be a string to define exploration method
        self.actions = config_dict['actions']  # Define the list of actions
        self.transition = config_dict['transition']

        # TODO: How to generalize the Q_Table to all ds
        self.task_name = config_dict['task_name']
        self.hist = config_dict['history'] # pandas dataframe
        self.knowledge = config_dict['knowledge'] # Dictionary table
        self.init_state = config_dict['initial_state']

    def get_states(self):
        return self.states

    def get_env_attr(self):
        return self.env_attr

    def get_exploration_method(self):
        return self.exploration_method

    def get_actions(self):
        return self.actions

    def get_transition(self):
        return self.transition

    def get_hist(self):
        return self.hist

    def get_task_name(self):
        return self.task_name

    def get_curr_state(self):
        if self.hist is None:
            return self.init_state
        return self.transition[self.get_prev_state()][self.get_prev_action()]

    def get_prev_state(self):
        if self.hist is None:
            return None
        return self.hist.iloc[-1]['curr_state']

    def get_knowledge(self):
        return self.knowledge

    def get_prev_action(self):
        if self.hist is None:
            return None

        return self.hist.iloc[-1]['curr_action']

    def get_total_step(self):
        return len(self.hist)

    def get_total_reward(self):
        return self.hist['reward'].sum()

    def get_next_state(self, state, action):
        return self.transition[state][action]
