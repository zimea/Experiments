import argparse
import sys, os
import importlib.util

parser = argparse.ArgumentParser(
    prog="ExperimentRunner",
    description="Starts individual experiments and logs results",
)
parser.add_argument(
    "-w",
    "--workdir",
    type=str,
    help="path to the working directory for input and output",
    default="/home/l/projects/Morpheus/Tutorial/Experiments/experiments/trial_1000",
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

sys.path.append(os.path.abspath(os.path.join(config.SourcePath)))
sys.path.append(os.path.abspath(os.path.join(config.BayesFlowPath)))
from DataReader import DataReader
from SimulationRunner import SimulationRunner
from ResultLogger import ResultLogger
from functools import partial
import tensorflow as tf
from contextlib import redirect_stdout, redirect_stderr
from bayesflow.forward_inference import GenerativeModel, Prior, Simulator
from bayesflow.networks import InvertibleNetwork
from bayesflow.amortized_inference import AmortizedPosterior
from bayesflow.trainers import Trainer
import bayesflow.diagnostics as diag

# Allow memory growth for the GPU
# physical_devices = tf.config.experimental.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
# tf.config.experimental.set_memory_growth(physical_devices[1], True)


if __name__ == "__main__":
    with open(os.path.join(workdir, "log_training.txt"), "w") as logfile:
        with redirect_stdout(logfile), redirect_stderr(logfile):
            prior = Prior(prior_fun=config.prior_func, param_names=config.prior_names)
            prior_means, prior_stds = prior.estimate_means_and_stds()
            dataReader = DataReader(
                config, prior_means=prior_means, prior_stds=prior_stds
            )
            simulationRunner = SimulationRunner(config, workdir, dataReader)

            simulator = Simulator(simulator_fun=partial(simulationRunner.run_morpheus))
            model = GenerativeModel(prior, simulator, name=config.model_name)

            summary_net = config.summary_network
            inference_net = InvertibleNetwork(
                {
                    "n_params": config.param_nr,
                    "n_coupling_layers": config.inn_layer,
                }
            )
            amortizer = AmortizedPosterior(
                inference_net, summary_net, name=config.amortizer_name
            )
            trainer = Trainer(
                amortizer=amortizer,
                generative_model=model,
                configurator=dataReader.prepare_input,
                checkpoint_path=os.path.join(workdir, config.checkpoints),
                optional_stopping=config.optional_stopping,
            )

            match config.training_mode:
                case "offline":
                    data, params = dataReader.read_offline_data(
                        os.path.join(config.data_path, config.folder + "/*")
                    )
                    h = trainer.train_offline(
                        epochs=config.epochs,
                        batch_size=config.batch_size,
                        simulations_dict={"prior_draws": params, "sim_data": data},
                    )
                case "online":
                    h = trainer.train_online(
                        epochs=config.epochs,
                        iterations_per_epoch=config.iter_per_epoch,
                        batch_size=config.batch_size,
                    )
                case _:
                    logfile.write("Unbekannter Trainingsmodus")
                    sys.exit()

            results = ResultLogger(
                workdir=workdir,
                trainer=trainer,
                losses=h,
                config=config,
                prior=prior,
                diag=diag,
            )
            results.create_plots()
