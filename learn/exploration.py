import collections
import math
import sys
import pandas as pd
# import torch

from learn.agent import Agent
from learn.reward import Reward
import random
from scipy.stats import beta, norm


# import torch.nn as nn
# import torch.nn.functional as F


class Explorer:
    """
    This class is a general representation of the explorer which is used to optimize for online learning.
    """

    def __init__(self, explore_config):
        self.explore_config = explore_config

    def learn(self, agent: Agent, reward: Reward, env_data):
        """
        Agent step forward using current environment data.
        :param agent: Learning agent.
        :param reward: Reward method.
        :param env_data: Data of the environment.
        :return: Possible logging history detail.
        """
        self.explore_config = None
        sys.exit("Explorer not implemented")

    def recommend(self, reward: float):
        """
        Explore next action and update knowledge.
        :param reward: Calculated reward value.
        :return: Recommended action at current step.
        """
        self.explore_config = None
        sys.exit("Explorer not implemented")


class EpsilonGreedyExplorer(Explorer):
    """
    The knowledge representation for EpsilonGreedyExplorer is Q-Table.
    """

    def __init__(self, explore_config: dict, agent: Agent):
        super().__init__(explore_config)
        self.explore_config = explore_config
        self.states = list(agent.get_states().keys())
        self.actions = agent.get_actions()
        self.epsilon = explore_config['epsilon']
        self.alpha = explore_config['learning_rate']
        self.discount = explore_config['discount']

        self.curr_state = agent.get_curr_state()
        self.possible_actions = list(agent.get_transition()[self.curr_state].keys())
        self.agent = agent

        # Initialize Q-table or load history
        if agent.get_hist() is not None:
            self.Q_table = agent.get_knowledge()
        else:
            self.Q_table = collections.defaultdict(dict)
            for state in self.states:
                for action in agent.get_transition()[state]:
                    self.Q_table[str(state)][action] = 0  # TODO: Check how to init table

    def select_action(self):
        if self.agent.get_hist() is None:
            return random.choice(self.possible_actions)
        rand_flag = random.random()
        if rand_flag < self.epsilon:
            return random.choice(self.possible_actions)
        else:
            return self.select_max_action(self.agent.get_curr_state())[1]

    def learn(self, agent: Agent, reward: Reward, env_data: pd.DataFrame):
        env_attr = agent.get_env_attr()
        reward_value = reward.get_reward(env_data[env_attr])
        recommended_action = self.recommend(reward_value)

        return agent.get_prev_state(), agent.get_prev_action(), agent.get_curr_state(), recommended_action, self.Q_table

    def select_max_action(self, state):
        max_val = float('-inf')
        max_action = None
        for action in self.possible_actions:
            if self.Q_table[str(state)][action] >= max_val:
                max_val = self.Q_table[str(state)][action]
                max_action = action
        return max_val, max_action

    def recommend(self, reward_value):
        recommended_action = self.select_action()
        curr_state = self.agent.get_curr_state()
        prev_state = self.agent.get_prev_state()
        prev_action = self.agent.get_prev_action()
        max_val, _ = self.select_max_action(curr_state)
        if prev_state is not None:  # Update only from second run onwards
            # Update Q_value for prev state according to current observation
            self.Q_table[str(prev_state)][prev_action] += self.alpha * (reward_value + self.discount * max_val -
                                                                   self.Q_table[str(prev_state)][prev_action])

        return recommended_action


class ThompsonSamplingExplorer(Explorer):
    """
    The knowledge representation for ThompsonSamplingExplorer is distribution.
    We store a nested dictionary for current distribution for each action
    Currently use a (alpha) and b (beta) to track the change of the distribution
    or a (mean) and b as variance, the update param return sqrt(b) because the library requires std error
    We can reward the agent from the range [0, 1] or provide normalized feature (consider later)
    This could be further update to support different distribution
    """

    def __init__(self, explore_config: dict, agent: Agent):
        super().__init__(explore_config)
        self.explore_config = explore_config
        self.states = agent.get_states()
        self.distribution_name = explore_config['distribution']
        self.agent = agent
        self.rv = None
        self.curr_state = self.agent.get_curr_state()
        self.possible_actions = list(self.agent.get_transition()[self.curr_state].keys())
        self.__init_rv__()

        # Initialize distribution or load history
        if agent.get_hist() is not None:
            self.distribution = agent.get_knowledge()
            self.total_step = agent.get_total_step()
            self.total_reward = agent.get_total_reward()
        else:
            self.distribution = collections.defaultdict(dict)

            if self.distribution_name == "beta":
                for state in self.states.keys():
                    for action in self.agent.get_transition()[state]:
                        self.distribution[state][action] = self.explore_config["initial_distribution"]
            else:
                print(self.states)
                for state in self.states.keys():
                    for action in self.agent.get_transition()[state]:
                        self.distribution[state][action] = self.explore_config["initial_distribution"]
            self.total_step = 0
            self.total_reward = 0

    def __init_rv__(self):
        if self.distribution_name == 'beta':
            self.rv = beta
        else:
            self.rv = norm

    def __update_param__(self, a, b, reward_value):
        if self.distribution_name == 'beta':
            return a + reward_value, b + (1 - reward_value)
        else:  # TODO: Check how to update normal distribution
            b = b * 0.9  # Find more serious update method
            action_step = self.agent.get_total_step_action(self.agent.get_prev_action(), self.agent.get_prev_action())
            if b < 1:
                b = 1
            a = (a * action_step + reward_value) / (action_step + 1)
            return a, b

    def select_action(self):
        # Calculate expectation for each beta distribution
        max_e = float('-inf')
        max_action = self.possible_actions[0]

        for action in self.possible_actions:
            distribution_param = self.distribution[self.curr_state][action]

            # Calculate expectation
            a, b = distribution_param['a'], distribution_param['b']
            action_sample = self.rv(a, b).rvs()

            if action_sample > max_e:
                max_e = action_sample
                max_action = action

        return max_action

    def learn(self, agent: Agent, reward: Reward, env_data: dict):
        env_attr = agent.get_env_attr()
        reward_value = reward.get_reward(env_data[env_attr])
        recommended_action = self.recommend(reward_value)

        return agent.get_prev_state(), agent.get_prev_action(), \
               agent.get_curr_state(), recommended_action, self.distribution

    def recommend(self, reward_value):
        recommended_action = self.select_action()
        prev_state = self.agent.get_prev_state()
        prev_action = self.agent.get_prev_action()
        print(prev_state, prev_action, reward_value)
        if prev_action is not None:
            print(self.distribution)
            prev_a, prev_b = self.distribution[str(prev_state)][prev_action]['a'], \
                             self.distribution[str(prev_state)][prev_action]['b']
            # Update the distribution

            a, b = self.__update_param__(prev_a, prev_b, reward_value)

            self.distribution[str(prev_state)][prev_action]['a'], self.distribution[str(prev_state)][prev_action]['b'] = a, b

        return recommended_action


class UpperConfidenceBoundExplorer(Explorer):
    """
    The knowledge representation for UpperConfidenceBoundExplorer is Q-Table, except the action selection
    counted one more term.
    """

    def __init__(self, explore_config: dict, agent: Agent):
        super().__init__(explore_config)
        self.explore_config = explore_config
        self.states = agent.get_states()
        self.actions = agent.get_actions()
        self.alpha = explore_config['learning_rate']
        self.discount = explore_config['discount']
        self.c = explore_config['confidence_bound']
        self.curr_state = agent.get_curr_state()
        self.possible_actions = list(agent.get_transition()[self.curr_state].keys())
        self.agent = agent
        self.total_step = 1
        self.counter = {}

        for action in self.actions:
            self.counter[action] = 0

        # Initialize Q-table or load history
        if agent.get_hist() is not None:
            self.Q_table = agent.get_knowledge()
            self.__generate_counter_from_hist__()
        else:
            self.Q_table = collections.defaultdict(dict)
            for state in self.states:
                for action in agent.get_transition()[state]:
                    self.Q_table[str(state)][action] = 0
            for action in self.actions:
                self.counter[action] = 0.0001
                self.total_step = 0.0001

    def __generate_counter_from_hist__(self):
        history = self.agent.get_hist()
        target = history['curr_action'].to_list()
        for entry in target:
            self.counter[entry] += 1
            self.total_step += 1

    def select_action(self):
        if self.agent.get_hist() is None:
            return random.choice(self.possible_actions)
        else:
            return self.select_max_action(self.agent.get_curr_state())[1]

    def learn(self, agent: Agent, reward: Reward, env_data: dict):
        env_attr = agent.get_env_attr()
        reward_value = reward.get_reward(env_data[env_attr])
        recommended_action = self.recommend(reward_value)
        return agent.get_prev_state(), agent.get_prev_action(), agent.get_curr_state(), recommended_action, self.Q_table

    def select_max_action(self, state):
        max_val = float('-inf')
        max_action = None
        max_q_val = float('-inf')

        for action in self.possible_actions:
            if self.counter[action] == 0:
                action_count = 0.001
            else:
                action_count = self.counter[action]
            if self.total_step >= 1:
                val = self.Q_table[str(state)][action] + self.c * math.sqrt(math.log2(self.total_step) / action_count)
            else:
                val = self.Q_table[str(state)][action]
            if val >= max_val:
                max_val = val
                max_action = action
                max_q_val = self.Q_table[str(state)][action]
        return max_q_val, max_action

    def recommend(self, reward_value):
        recommended_action = self.select_action()
        curr_state = self.agent.get_curr_state()
        prev_state = self.agent.get_prev_state()
        prev_action = self.agent.get_prev_action()
        max_val, _ = self.select_max_action(curr_state)

        # Update Q_value for prev state according to current observation
        if prev_state is not None:
            self.Q_table[str(prev_state)][prev_action] += self.alpha * (reward_value + self.discount * max_val -
                                                                   self.Q_table[str(prev_state)][prev_action])

        return recommended_action

# class DeepQExplorer(Explorer):
#     """
#     Input of the model should be previous environment attribute. (Given current observation, predict the value of reward
#     by NN inference)
#     """
#
#     def __init__(self, explore_config: dict, agent: Agent):
#         super().__init__(explore_config)
#         assert self.explore_config.__contains__("selection_criteria")  # TODO: Add this to explore config (UCB, EG)
#         self.explore_config = explore_config
#         self.states = agent.get_states()
#         self.actions = agent.get_actions()
#         self.action_index_map = {self.actions[i]: i for i in range(0, len(self.actions))}
#         self.curr_state = agent.get_curr_state()
#         self.possible_actions = list(agent.get_transition()[self.curr_state].keys())
#         self.agent = agent
#         self.optimizer = getattr(sys.modules['torch.optim'], self.explore_config['optimizer'])
#         self.criterion = getattr(sys.modules['torch.nn'], self.explore_config['loss'])
#         self.discount = self.explore_config['discount']
#
#         # TODO: Check this
#         self.embedding_size = len(self.explore_config['env_attr'])
#         self.epsilon = self.explore_config['epsilon']
#         self.c = explore_config['confidence_bound']
#         self.prev_env = agent.get_prev_env()  # dictionary or None
#         self.model = agent.get_knowledge()
#         self.counter = {}
#
#         if agent.get_hist() is not None:
#             self.__generate_counter_from_hist__()
#         else:
#             for action in self.actions:
#                 self.counter[action] = 0.0001
#                 self.total_step = 0.0001
#
#     def __generate_counter_from_hist__(self):
#         history = self.agent.get_hist()
#         target = history['curr_action'].to_list()
#         for entry in target:
#             self.counter[entry] += 1
#             self.total_step += 1
#
#     def learn(self, agent: Agent, reward: Reward, env_data):
#         # Reward is ground truth of taking previous action
#         self.model.eval()
#         env_attr = agent.get_env_attr()
#         reward_value = reward.get_reward(env_data[env_attr])
#
#         if self.agent.get_prev_state() is None:
#             recommended_action = random.choice(self.possible_actions)
#         else:
#             recommended_action = self.recommend_from_model(env_data[env_attr])
#
#         prev_action = agent.get_prev_action()
#         prev_action_index = self.action_index_map[prev_action]
#         max_curr = torch.argmax(self.model(env_data[env_attr]))
#         target_inference_vector = [max_curr * self.discount] * len(self.actions)
#         target_inference_vector[prev_action_index] += reward_value
#
#         # Update model with (prev_state, reward) pair
#         if agent.get_prev_action() is not None:
#             self.model.train()
#             self.optimizer.zero_grad()
#
#             outputs = self.model(self.prev_env)
#             loss = self.criterion(outputs, target_inference_vector)
#             loss.backward()
#             self.optimizer.step()
#
#             print(loss)
#
#         return agent.get_prev_state(), agent.get_prev_action(), agent.get_curr_state(), recommended_action, self.model
#
#     def select_action_with_inference_result(self, inference):
#         if self.agent.get_hist() is None:
#             return random.choice(self.possible_actions)
#
#         # Modify the inference vector according to our chosen selection criteria
#         if self.explore_config["selection_criteria"] == 'epsilon_greedy':
#             rand_flag = random.random()
#             if rand_flag < self.epsilon:
#                 return random.choice(self.possible_actions)
#         if self.explore_config["selection_criteria"] == "upper_confidence_bound":
#             for action in self.possible_actions:
#                 action_count = self.counter[action]
#                 if self.total_step >= 1:
#                     inference[self.action_index_map[action]] += \
#                         self.c * math.sqrt(math.log2(self.total_step) / action_count)
#         return self.get_max_from_inference_result(inference)
#
#     def get_max_from_inference_result(self, inference):
#         max_r = float('-inf')
#         max_a = None
#         for action in self.possible_actions:
#             curr_res = inference[self.action_index_map[action]]
#             if curr_res >= max_r:
#                 max_a = action
#                 max_r = curr_res
#         return max_a
#
#     def recommend_from_model(self, env):
#         self.model.eval()
#         inference_result = self.model.forward(env)
#
#         return self.select_action_with_inference_result(inference_result)


# class Model(nn.Module):
#    def __init__(self, num_embedding, num_action):
#        super().__init__()
#        self.forward = nn.Linear(num_embedding, 10)  #
#        self.output = nn.Linear(10, num_action)
#        self.relu = nn.ReLU()

#    def forward(self, x):
#        x = self.forward(x)
#        x = self.relu(x)
#        return self.output(x)
