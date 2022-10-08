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
        assert set(self.env_attr).issubset(env.columns)
        target = env.iloc[-1]
        p1 = int(target['p1'])
        p2 = int(target['p2'])
        p3 = int(target['p3'])
        assert p1 + p2 + p3 == 1

        if p1 == 1:
            return 0
        elif p2 == 1:
            return 0.5
        else:
            return 1


class TTLReward(Reward):
    def __init__(self, env_attr):
        super().__init__(env_attr)

    def get_reward(self, env: pd.DataFrame):
        assert set(self.env_attr).issubset(env.keys())

        today_booking_counts = env["today_booking_counts"]
        daily_uptime_pct = env["daily_uptime_pct"]
        supplier_avg_bookings_before_today = env["supplier_avg_bookings_before_today"]

        reward = 0
        if today_booking_counts < supplier_avg_bookings_before_today:
            reward = reward - 5
        else:
            reward = reward + 5

        if daily_uptime_pct >= 0.90:
            reward = reward + 1

        if daily_uptime_pct < 0.90:
            reward = reward - 1

        if daily_uptime_pct < 0.80:
            reward = reward - 2

        if daily_uptime_pct < 0.60:
            reward = reward - 4

        return reward


class TTLRewardTS(Reward):
    def __init__(self, env_attr):
        super().__init__(env_attr)

    def get_reward(self, env: pd.Series):
        assert set(self.env_attr).issubset(env.keys())

        today_booking_counts = env["today_booking_counts"]
        daily_uptime_pct = env["daily_uptime_pct"]
        supplier_avg_bookings_before_today = env["supplier_avg_bookings_before_today"]

        reward = 0.5
        if today_booking_counts < supplier_avg_bookings_before_today:
            reward = reward - 0.2
        else:
            reward = reward + 0.2

        if daily_uptime_pct >= 0.90:
            reward = reward + 0.1

        if daily_uptime_pct < 0.90:
            reward = reward - 0.1

        if daily_uptime_pct < 0.80:
            reward = reward - 0.05

        if daily_uptime_pct < 0.60:
            reward = reward - 0.1

        return reward
