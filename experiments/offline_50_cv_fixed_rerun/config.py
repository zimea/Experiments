# config uses a python file, which is questionable in production use. Instead e.g. yaml would be a better choice

import sys, os


SourcePath = "/home/l/projects/Morpheus/Tutorial/Experiments/src"
BayesFlowPath = "/home/l/projects/Morpheus/Tutorial/Experiments/BayesFlow"
sys.path.append(os.path.abspath(SourcePath))
import PriorFunctions
import SummaryNetworks

# set BayesFlow modules
prior_names = [r"bcf", r"pi"]
prior_func = PriorFunctions.simple_prior
summary_parm = 16
summary_network = SummaryNetworks.ConvLSTM(n_summary=summary_parm)

# where to put the results
resultsPath = "experiments"
checkpoints = "checkpoints"
plots = "plots"

# data config
data_path = "/home/l/projects/Morpheus/Modelle/cell_free_50"
folder = "output_fixed_cv_wm"
cell_nr = 51
grid_size = 10
timesteps = 50
cut_off_start = 9
cut_off_end = 10

# training hyperparameter
param_nr = len(prior_func())
inn_layer = 4
batch_size = 32
iter_per_epoch = 200
epochs = 100
retrain = False
model_name = "morpheus"
training_mode = "offline"
amortizer_name = "emune_amortizer"
optional_stopping = True

# which plots and diagnostics
losses = True
latent2d = True
sbc_histograms = True
sbc_ecdf = True
posterior_scores = True
recovery = True
correlation = True
slope = True
run_resimulations = sbc_ecdf or posterior_scores or recovery or correlation or slope
resimulation_param = {"simulations": 100, "post_samples": 500}
