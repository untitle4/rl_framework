import subprocess
import sys

import yaml
import os

# assume your intersection.sumocfg is under SUMO_sample directory
# the output file will also be generated under that directory


def checkFileExistence(proj_name):
    files = []
    for f in os.listdir(f"./{config['configs']['sumo_loc']}"):
        if os.path.isfile(f"./{config['configs']['sumo_loc']}/{f}"):
            files.append(f)
    check_list = [".nod.xml", ".con.xml", ".edg.xml", ".netccfg", ".rou.xml", ".add.xml"]
    for c in check_list:
        if proj_name + c not in files:
            print(f"{proj_name + c} not exists")
            sys.exit(1)


def run(runner_config):
    with open(runner_config, "r") as f:
        config = yaml.safe_load(f)

    checkFileExistence(config["configs"]["proj_name"])

    subprocess.check_output(f"sumo -c {config['configs']['sumo_loc']}/intersection.sumocfg --queue-output {config['configs']['sumo_loc']}/out.xml", shell=True)


