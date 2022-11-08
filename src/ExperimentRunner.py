import config
import SimulationRunner
import DataReader
import ResultLogger

import warnings
import os, sys
import numpy as np
import pandas as pd
import datetime
from functools import partial
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
import glob
from fnmatch import fnmatch
import plotnine as p9
import patchworklib as pw
from sklearn.linear_model import LinearRegression

sys.path.append(os.path.abspath(os.path.join(config.BayesFlowPath)))
from bayesflow.forward_inference import GenerativeModel, Prior, Simulator
from bayesflow.networks import InvertibleNetwork, InvariantNetwork
from bayesflow.amortized_inference import AmortizedPosterior
from bayesflow.trainers import Trainer

import time
import glob
import subprocess
from contextlib import redirect_stdout, redirect_stderr


# Allow memory growth for the GPU
physical_devices = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.experimental.set_memory_growth(physical_devices[1], True)


if __name__ == "__main__":
    workdir = os.environ["WORKDIR"]
    with open(os.path.join(workdir, "log_training.txt"), "w") as logfile:
        with redirect_stdout(logfile) and redirect_stderr(logfile):
            prior = Prior(prior_fun=config.prior_func, param_names=config.prior_names)

            simulator = Simulator(
                simulator_fun=partial(SimulationRunner.run_morpheus), workdir=workdir
            )
            model = GenerativeModel(prior, simulator, name=config.model_name)

            prior_means, prior_stds = prior.estimate_means_and_stds()
            prepare_input_fun = partial(
                DataReader.prepare_input, prior_means=prior_means, prior_stds=prior_stds
            )
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
                configurator=prepare_input_fun,
                checkpoint_path=os.path.join(workdir, config.checkpoints),
                optional_stopping=config.optional_stopping,
            )

            match config.training_mode:
                case "offline":
                    data, params = DataReader.read_offline_data(
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

            results = ResultLogger(trainer=trainer, losses=h, workdir=workdir)
            results.create_plots()
