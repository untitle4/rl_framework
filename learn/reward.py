import sys


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
