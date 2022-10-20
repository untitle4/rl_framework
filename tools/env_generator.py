import os.path

import pandas as pd


def generate_env(path, history_path):
    res = pd.read_csv(os.path.join(path, 'out.csv'))
    hist = pd.read_csv(history_path)

    env = {'avg_lane_queueing_length': res['lane_queueing_length'].mean(numeric_only=True, skipna=True),
           'avg_lane_queueing_time': res['lane_queueing_time'].mean(numeric_only=True, skipna=True),
           'prev_avg_lane_queueing_length': hist.iloc[-1]['avg_lane_queueing_length'],
           'prev_avg_lane_queueing_time': hist.iloc[-1]['avg_lane_queueing_length']}

    return pd.DataFrame.from_dict(env)


def generate_init_env():
    # Random init env to start training
    return {'avg_lane_queueing_length': 100,
            'avg_lane_queueing_time': 100,
            'prev_avg_lane_queueing_length': 101,
            'prev_avg_lane_queueing_time': 101}
