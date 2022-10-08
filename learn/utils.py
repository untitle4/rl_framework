import hdfs
import pandas as pd
import time
from datetime import datetime
from zapp.hadoop import HdfsClient
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


def update_file_with_retry(src: os.path, des: os.path, retry_num: int, client: HdfsClient):
    retry_iter = 1

    while retry_iter < retry_num:
        try:
            client.upload_file(src, des)
            retry_iter = retry_num
        except:
            print("Connection error. retry: ", retry_iter)
            retry_iter += 1

    if client.exists_file_dir(des):
        return 1
    else:
        return 0


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
            "action_hist_parquet": ".parquet",
            "knowledge_parquet": ".json",
            "update_content": ".parquet"
        }
    # if learning_method == 'deep_q':
    #     return {
    #         "action_hist_parquet": ".parquet",
    #         "knowledge": ".pt",
    #         "update_content": ".parquet"
    #     }


# TODO: Adapt this to the framework
class FileHandler:
    def __init__(self, task_name, client_verbose, user, credential, suffix_map):
        """
        Initialize the file handler
        :param tracking_files: The file list that needs to be tracked.
        :param client_verbose: Verbose of the Hdfs client
        :param credential: Credential file path
        """
        self.tracking_files = [tracking_file + suffix_map[tracking_file] for tracking_file in list(suffix_map.keys())]
        self.suffix_map = suffix_map
        self.client_verbose = client_verbose
        self.user = user
        self.credential = credential
        self.local_dir = f'/tmp/{task_name}'
        if not os.path.exists(self.local_dir):
            os.mkdir(self.local_dir)
        self.hadoop_table_folder = f"/user/hk-vulcan-svc/data_science/online_learning_platform/{task_name}"

        # Initialize local directory list
        self.local_file_map = {track_file: os.path.join(self.local_dir, track_file)
                               for track_file in self.tracking_files}

        # Initialize remove directory list
        self.remote_file_map = {track_file: os.path.join(self.hadoop_table_folder, track_file)
                                for track_file in self.tracking_files}
        self.hdfs_client = HdfsClient(user_name=self.user, hadoop_cluster="sgp",
                                      path_to_creds=self.credential, verbose=self.client_verbose)

    def upload_tracking_file(self):
        for track_file in self.tracking_files:
            log(f"Uploading {self.local_file_map[track_file]}")
            res = update_file_with_retry(self.local_file_map[track_file], self.remote_file_map[track_file],
                                         10, self.hdfs_client)
            assert res == 1
            log(f"Uploading {self.local_file_map[track_file]} successfully.")

    def reset_tracking_file(self):
        log("Clearing the tracking file...")
        if self.hdfs_client.exists_file_dir(self.hadoop_table_folder):
            self.hdfs_client.delete(self.hadoop_table_folder, True)
        log("Tracking file has been reset.")

    def pull_tracking_file(self):
        # Download action table to local temp folder
        pull_res = []
        for file in self.tracking_files:
            curr_res = 0
            log(f"Fetching {file}")
            if self.hdfs_client.exists_file_dir(self.remote_file_map[file]):
                curr_res = pull_file_with_retry(self.remote_file_map[file], self.local_file_map[file], 10,
                                                self.hdfs_client)
            if curr_res == 1:
                log("Fetched successfully.")
            else:
                log("Failed to fetch file.")
            pull_res.append(curr_res)
        return pull_res

    def check_or_create_remote(self):
        if not self.hdfs_client.exists_file_dir(self.hadoop_table_folder):
            log(f"{self.hadoop_table_folder} Not exist, Creating one...")
            self.hdfs_client.make_dir(self.hadoop_table_folder)
            log("Made directory successfully.")
        else:
            log(f"{self.hadoop_table_folder} has been created.")

    def update_with_local(self, target_name):
        target_file_name = target_name + self.suffix_map[target_name]
        local_file = self.local_file_map[target_file_name]
        remote_file = self.remote_file_map[target_file_name]
        if not os.path.exists(local_file):
            log(f"{target_name} does not exists on local. Please save on local first.")
        else:
            res = update_file_with_retry(local_file, remote_file, 10, self.hdfs_client)
            if res == 1:
                log(f"{target_name} Fetched Successfully.")
            else:
                log(f"Error occurs when fetching {target_name}.")

    def download_to_local(self, target_name):
        target_file_name = target_name + self.suffix_map[target_name]
        local_file = self.local_file_map[target_file_name]
        remote_file = self.remote_file_map[target_file_name]
        if not self.hdfs_client.exists_file_dir(remote_file):
            log(f"{target_name} does not exists on remote. Please upload to remote first.")
        else:
            try:
                pull_file_with_retry(remote_file, local_file, 10, self.hdfs_client)
                log(f"{target_name} Downloaded Successfully.")
            except:
                log(f"Error when downloading{target_name}")

    def clear_local_file(self):
        if os.path.exists(self.local_dir):
            shutil.rmtree(self.local_dir)
        log("Cleared local file")

    def get_local_path(self, target_name):
        target_file_name = target_name + self.suffix_map[target_name]
        return self.local_file_map[target_file_name]

    def get_remote_path(self, target_name):
        target_file_name = target_name + self.suffix_map[target_name]
        return self.remote_file_map[target_file_name]
