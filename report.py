REPORT_REPO = "/home/l/projects/Morpheus/Tutorial/BayesFlowTutorial"
REPORT_NAME = "report.tex"

################################################################################################

import argparse, os, importlib, sys
from subprocess import Popen

################################################################################################

parser = argparse.ArgumentParser(
    prog="ExperimentRunner",
    description="Starts individual experiments and logs results",
)
parser.add_argument(
    "-w",
    "--workdir",
    type=str,
    help="path to the working directory for input and output",
    default="/home/l/projects/Morpheus/Tutorial/Experiments/experiments/offline_50_cv_data_split/",
)
parser.add_argument(
    "-c", "--configfile", type=str, help="path to the config file", default="config.py"
)
args = parser.parse_args()
workdir = os.path.abspath(os.path.abspath(os.path.join(os.getcwd(), args.workdir)))
configfile = os.path.abspath(os.path.abspath(os.path.join(workdir, args.configfile)))

spec = importlib.util.spec_from_file_location("config", configfile)
config = importlib.util.module_from_spec(spec)
sys.modules["config"] = config
spec.loader.exec_module(config)

################################################################################################

experiment_name = workdir.split("/")[-1]
report_path = os.path.join(REPORT_REPO, experiment_name)
report = os.path.join(report_path, REPORT_NAME)
checkpoint_path = os.path.join(workdir, "checkpoints", "checkpoint")
experiment_timestamp = Popen('sed "$!d" %s | grep -E -o "[0-9]{10}"' % checkpoint_path)

################################################################################################

if not os.path.exists(REPORT_REPO):
    raise Exception("Path %s does not exist.\n" % REPORT_REPO)
if not os.path.exists(report_path):
    print("Report folder for experiment does not yet exist.\n")
    os.mkdir(report_path)
    print("Folder %s created." % report_path)
else:
    print("Folder %s exists." % report_path)

os.mkdir(os.path.join(report_path, "results_%s" % experiment_timestamp))
os.mkdir(os.path.join(report_path, "results_%s" % experiment_timestamp, "plots"))

if not os.path.exists(report):
    print("Report file for experiment does not yet exist.\n")
    Popen("touch %s" % report)
    print("Report file %s created." % report)


# create repository
# copy plots
# if not exists, create tex file
# add figure with config values and date
