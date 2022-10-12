import pandas as pd
import time
from datetime import datetime
import os
import shutil
import json


def to_json(path, knowledge):
    json_dir = json.dumps(knowledge)
    f = open(path, "w")
    f.write(json_dir)
    f.close()


def load_json(path):
    with open(path) as json_file:
        data = json.load(json_file)
        return data


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


def fetch_explore_config_requirement(exploration: str):
    if exploration == 'epsilon_greedy':
        return {'epsilon', 'learning_rate', 'discount'}

    if exploration == 'upper_confidence_bound':
        return {'epsilon', 'learning_rate', 'discount', 'confidence_bound'}

    if exploration == 'thompson_sampling':
        return {'distribution', 'initial_distribution'}
    return set()


def log(msg):
    print(msg)


def get_file_suffix_map(learning_method):
    if learning_method == 'epsilon_greedy' or learning_method == 'thompson_sampling' or \
            learning_method == 'upper_confidence_bound':
        return {
            "action_hist": ".csv",
            "knowledge": ".json",
            "update_content": ".csv"
        }
    # if learning_method == 'deep_q':
    #     return {
    #         "action_hist_parquet": ".parquet",
    #         "knowledge": ".pt",
    #         "update_content": ".parquet"
    #     }


# TODO: Adapt this to the framework
class FileHandler:
    def __init__(self, task_name, suffix_map):
        """
        Initialize the file handler
        :param tracking_files: The file list that needs to be tracked.
        :param client_verbose: Verbose of the Hdfs client
        :param credential: Credential file path
        """
        self.tracking_files = [tracking_file + suffix_map[tracking_file] for tracking_file in list(suffix_map.keys())]
        self.suffix_map = suffix_map
        curr_dir = os.getcwd()
        self.local_dir = os.path.join(curr_dir, f'learning_out/{task_name}')
        if not os.path.exists(self.local_dir):
            os.mkdir(self.local_dir)

        # Initialize local directory list
        self.local_file_map = {track_file: os.path.join(self.local_dir, track_file)
                               for track_file in self.tracking_files}

    def clear_local_file(self):
        if os.path.exists(self.local_dir):
            shutil.rmtree(self.local_dir)
        log("Cleared local file")

    def get_local_path(self, target_name):
        target_file_name = target_name + self.suffix_map[target_name]
        return self.local_file_map[target_file_name]
