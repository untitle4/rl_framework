import sys

import pandas as pd


class Reward:
    def __init__(self, env_attr):
        """
        A general interface for reward class.
        :param env_attr: Environment attributes which are used to calculate the reward given an environment.
        This should match the attribute provided by env_data
        """
        self.env_attr = env_attr

    def get_reward(self, env):
        environment = env[self.env_attr]
        sys.exit('Reward function not implemented')


class SampleReward(Reward):
    def __init__(self, env_attr):
        super().__init__(env_attr)

    def get_reward(self, env):
        return 1


class TestReward(Reward):
    def __init__(self, env_attr):
        super().__init__(env_attr)

    def get_reward(self, env: pd.DataFrame):
        target = env
        p1 = int(target['p1'])
        p2 = int(target['p2'])
        p3 = int(target['p3'])
        assert p1 + p2 + p3 == 1

        if p1 == 1:
            return -5
        elif p2 == 1:
            return 0
        else:
            return 5


