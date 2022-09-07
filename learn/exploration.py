import math
import sys


from learn.agent import Agent
from learn.reward import Reward
import random
from scipy.stats import beta, norm


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
        self.actions = list(agent.get_actions().keys())
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
            self.Q_table = {}
            for state in self.states:
                for action in agent.get_transition()[state]:
                    self.Q_table[state] = {action: 0}  # TODO: Check how to init table

    def select_action(self):
        curr_state = self.agent.get_curr_state()
        if self.agent.get_hist() is None:
            return random.choice(self.possible_actions)
        rand_flag = random.random()
        if rand_flag < self.epsilon:
            return random.choice(self.possible_actions)
        else:
            return self.select_max_action(self.agent.get_curr_state())[1]

    def learn(self, agent: Agent, reward: Reward, env_data: dict):
        env_attr = agent.get_env_attr()
        reward_value = reward.get_reward(env_data[env_attr])
        recommended_action = self.recommend(reward_value)

        return agent.get_prev_state(), agent.get_prev_action(), agent.get_curr_state(), recommended_action, self.Q_table

    def select_max_action(self, state):
        max_val = 0
        max_action = None
        for action in self.possible_actions:
            if self.Q_table[state][action] >= max_val:
                max_val = self.Q_table[state][action]
                max_action = action
        return max_val, max_action

    def recommend(self, reward_value):
        recommended_action = self.select_action()
        curr_state = self.agent.get_curr_state()
        prev_state = self.agent.get_prev_state()
        prev_action = self.agent.get_prev_action()
        max_val, _ = self.select_max_action(curr_state)
        if prev_state is not None: # Update only from second run onwards
            # Update Q_value for prev state according to current observation
            self.Q_table[prev_state][prev_action] += self.alpha * (reward_value + self.discount * max_val -
                                                                   self.Q_table[prev_state][prev_action])

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
        self.actions = agent.get_actions()
        self.distribution_name = explore_config['distribution']
        self.agent = agent
        self.rv = None
        self.curr_state = self.agent.get_curr_state()
        self.possible_actions = self.agent.get_transition()[self.curr_state].keys()
        self.__init_rv__()

        # Initialize distribution or load history
        if agent.get_hist() is not None:
            self.distribution = agent.get_knowledge()
            self.total_step = agent.get_total_step()
            self.total_reward = agent.get_total_reward()
        else:
            self.distribution = {}
            for action in self.actions:
                self.distribution[action] = {'a': 1, 'b': 1}
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
        else: # TODO: Check how to update normal distribution
            b = 1.0 / ((1 / 100 ** 2) + self.total_step)
            a = b * self.total_reward
            return a, math.sqrt(b)

    def select_action(self):
        # Calculate expectation for each beta distribution
        max_e = 0
        max_action = self.possible_actions[0]

        for action in self.possible_actions:
            distribution_param = self.distribution[action]

            # Calculate expectation
            a, b = distribution_param['a'], distribution_param['b']
            action_sample = self.rv(a, b)

            if action > max_e:
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
        prev_action = self.agent.get_prev_action()

        if prev_action is not None:
            prev_a, prev_b = self.distribution[prev_action]['a'], self.distribution[prev_action]['b']
            # Update the distribution

            a, b = self.__update_param__(prev_a, prev_b, reward_value)

            self.distribution[prev_action]['a'], self.distribution[prev_action]['b'] = a, b

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
        self.epsilon = explore_config['epsilon']
        self.alpha = explore_config['learning_rate']
        self.discount = explore_config['discount']
        self.c = explore_config['confidence_bound']
        self.curr_state = agent.get_curr_state()
        self.possible_actions = agent.get_transition()[self.curr_state].keys()
        self.agent = agent
        self.total_step = 1
        self.counter = {}

        for action in self.actions:
            self.counter[action] = 0

        # Initialize Q-table or load history
        if agent.get_hist() is not None:
            self.Q_table = agent.get_knowledge()
        else:
            self.Q_table = {}
            for (state, action) in enumerate(self.states, self.actions):
                self.Q_table[state] = {action: 0}  # TODO: Check how to init table
            self.__generate_counter_from_hist__()

    def __generate_counter_from_hist__(self):
        history = self.agent.get_hist()

        for entry in history:
            prev_action = entry['prev_action']
            self.counter[prev_action] += 1
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
        # TODO: Update agent's policy here
        return agent.get_prev_state(), agent.get_prev_action(), agent.get_curr_state(), recommended_action, self.Q_table

    def select_max_action(self, state):
        max_val = 0
        max_action = None

        for action in self.possible_actions:
            if self.counter[action] == 0:
                action_count = 0.001
            else:
                action_count = self.counter[action]

            val = self.Q_table[state][action] + self.c * math.sqrt(math.log2(self.total_step) / action_count)
            if val >= max_val:
                max_val = self.Q_table[state][action]
                max_action = action
        return max_val, max_action

    def recommend(self, reward_value):
        recommended_action = self.select_action()
        curr_state = self.agent.get_curr_state()
        prev_state = self.agent.get_prev_state()
        prev_action = self.agent.get_prev_action()
        max_val, _ = self.select_max_action(curr_state)

        # Update Q_value for prev state according to current observation
        if prev_state is not None:
            self.Q_table[prev_state][prev_action] += self.alpha * (reward_value + self.discount * max_val -
                                                                    self.Q_table[prev_state][prev_action])

        return recommended_action
