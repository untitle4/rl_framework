import pandas as pd
import time
from datetime import datetime
from zapp.hadoop import HdfsClient
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


def pull_file_with_retry(src: os.path, des: os.path, retry_num: int, client: HdfsClient):
    retry_iter = 1

    while retry_iter < retry_num:
        try:
            client.download_file(src, des)
            retry_iter = retry_num
        except:
            print("Connection error. retry: ", retry_iter)
            retry_iter += 1

    if os.path.exists(des):
        return 1
    else:
        return 0


def log(msg):
    print(msg)
