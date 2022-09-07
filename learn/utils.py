import pandas as pd
import time
from datetime import datetime
import os


def generate_log_hist(prev_state, prev_action, curr_state, curr_action):
    return {'prev_state': prev_state, 'prev_action': prev_action, 'curr_state': curr_state, 'curr_action': curr_action}


def table_to_dictionary(data_frame: pd.DataFrame):
    return data_frame.to_dict()


def dictionary_to_table(data_dict: dict):
    return pd.DataFrame.from_dict(data_dict)


def get_utc_timestamp() -> int:
    return int(time.time())


def utc_to_datetime_string(utc_timestamp: int) -> str:
    return str(datetime.fromtimestamp(utc_timestamp))

def log(msg):
    print(msg)
