import subprocess
import yaml

# assume your intersection.sumocfg is under SUMO_sample directory
# the output file will also be generated under that directory

with open("./configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

subprocess.check_output(f"sumo -c {config['configs']['sumo_loc']}/intersection.sumocfg --queue-output {config['configs']['sumo_loc']}/out.xml", shell=True)