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
data_path = "/home/l/projects/Morpheus/Modelle/cell_free_1000"
folder = "output_fixed_cv_wm"
cell_nr = 1001
grid_size = 45
timesteps = 50
cut_off_start = 9
cut_off_end = 10
# data cache
processed_data_path = "/home/l/projects/Morpheus/Tutorial/Experiments/data"
test_ratio = 0.3
validation_ratio = 0.3
validation_nr = 100
reread_data = False
test_online = False

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
sbc_test_data = True
sbc_ecdf = True
posterior_scores = True
recovery = True

plot_individual_sims = True
plot_posterior_2d = True
plot_ppc = True
correlation = True
slope = True
nr_individual_sims = 10
nr_of_resimulations = 10

run_resimulations = (
    sbc_ecdf or posterior_scores or recovery or correlation or slope or sbc_test_data
)
resimulation_param = {"simulations": 100, "post_samples": 100}
