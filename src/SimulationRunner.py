import config
import glob
import os, sys
from subprocess import Popen, PIPE, STDOUT
from DataReader import calculate_V, calculate_volume
import numpy as np
import time
from pandas import read_csv
from contextlib import redirect_stdout, redirect_stderr


def run_morpheus(params, workdir):
    with open(os.path.join(workdir, "log_morpheus.txt"), "w") as logfile:
        with redirect_stderr(logfile) and redirect_stdout(logfile):
            bcf, pi = params
            cv = 0.33
            DV_str = "wm"
            DV = 999

            model_dir = "model"
            model_pattern = os.path.join(
                config.data_path, model_dir, "*DV-%s*.xml" % DV_str
            )
            models = glob.glob(model_pattern)
            model = models[0]

            OUT = (
                os.path.join(
                    config.data_path,
                    config.folder,
                    "DV-" + str(DV) + "_bcf-" + str(bcf) + "_" + "pi-" + str(pi),
                )
                + "_"
                + "cv-"
                + str(cv)
            )
            create_dir = Popen("mkdir " + OUT)
            create_dir.wait()

            run_sim = Popen(
                "morpheus"
                + " -f "
                + model
                + " -o "
                + OUT
                + " -b_cf "
                + bcf
                + " -c_V "
                + cv
                + " -p_V "
                + pi
            )
            run_sim.wait()

            final_plot = os.path.join(
                OUT, "plot_" + str(config.timesteps).zfill(5) + ".png"
            )
            while not os.path.exists(final_plot):
                time.sleep(1)

            population_file = os.path.join(OUT, "logger_2.csv")
            df = read_csv(population_file, sep="\t")
            df_tar = df["celltype.target.size"].values[:, np.newaxis][
                config.cut_off_start + 1 : config.timesteps - config.cut_off_end
            ]
            df_inf = df["celltype.infected.size"].values[:, np.newaxis][
                config.cut_off_start + 1 : config.timesteps - config.cut_off_end
            ]
            df_cells = np.append(df_tar, df_inf, axis=1)

            v_path = os.path.join(OUT, "logger_6_Ve.csv")
            v = calculate_V(v_path)
            sim = np.append(df_cells, v, axis=1)

            I_volume = calculate_volume(OUT)[
                config.cut_off_start + 1 : config.timesteps - config.cut_off_end
            ]
            sim = np.append(sim, I_volume, axis=1)

            return sim
