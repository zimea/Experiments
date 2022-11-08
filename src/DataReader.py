import os
import config
import pandas as pd
import numpy as np
import glob


def calculate_V(path: str):
    df = pd.read_csv(path, sep="\t")
    df = df[df[str(config.grid_size)] != config.grid_size]
    morpheus_ts = np.repeat(range(0, config.timesteps + 1), config.grid_size)
    times = np.tile(morpheus_ts, int(len(df.index) / morpheus_ts.shape[0]))
    if len(times) != len(df.index):
        print(path)
    df["time"] = np.tile(morpheus_ts, int(len(df.index) / morpheus_ts.shape[0]))
    df["sum_V"] = df.iloc[:, 1 : config.grid_size + 1].sum(axis=1)
    df = df.drop(columns=[str(x) for x in range(0, config.grid_size + 1)])
    df = df.groupby(list(df.columns[:-1])).agg({"sum_V": "sum"}).reset_index()
    return np.expand_dims(df["sum_V"], axis=1)[
        config.cut_off_start + 1 : config.timesteps - config.cut_off_end
    ]


def calculate_volume(path: str):
    logger_state = "logger_1.csv"
    logger_volume = "logger_4_cell.id.csv"
    state = pd.read_csv(os.path.join(path, logger_state), sep="\t").rename(
        columns={"cell.id": "id"}
    )
    volume = np.genfromtxt(os.path.join(path, logger_volume), delimiter="\t")
    lines_per_timepoint = (config.timesteps + 1) * (config.grid_size + 1)
    volume = np.delete(
        volume, list(range(0, int(lines_per_timepoint), config.grid_size + 1)), axis=0
    )

    first_ts_after_freeze = config.grid_size * (config.cut_off_start + 1)
    cell_id, counts = np.array(
        np.unique(
            np.array(volume)[
                first_ts_after_freeze : first_ts_after_freeze + config.grid_size, 1:
            ],
            return_counts=True,
        )
    )
    counts = np.append(
        np.expand_dims(cell_id, axis=1), np.expand_dims(counts, axis=1), axis=1
    )
    if counts[0, 0] != 0:
        counts = np.insert(counts, 0, [[0, 0]], axis=0)
    counts = pd.DataFrame(counts, columns=["id", "count"])
    df = pd.merge(left=state, right=counts, how="left", on=["id"])
    df = df.groupby(["time", "V"]).agg({"count": "sum"}).reset_index()
    return np.expand_dims(df.query("V == 1")["count"], axis=1)


def read_offline_data(path: str):
    path_list = glob.glob(path)
    nr_of_params = config.param_nr

    n_sim = len(path_list)
    dfs = np.empty(
        (n_sim, (config.timesteps - 1 - config.cut_off_start - config.cut_off_end), 4),
        dtype=np.float32,
    )
    params = np.empty((n_sim, nr_of_params), dtype=np.float32)
    invalidIndices = []

    for path in range(n_sim):
        pathname = path_list[path]
        filename = os.path.join(pathname, "logger_2.csv")
        filename_V = os.path.join(pathname, "logger_6_Ve.csv")
        df = pd.read_csv(filename, index_col=None, header=0, delimiter="\t")
        path_split = filename.split("/")[len(filename.split("/")) - 2]
        if "e" in path_split:
            invalidIndices.append(path)
            continue
        if path_split.startswith("sweep") or path_split.startswith("DV"):
            start_nr = 1
        else:
            start_nr = 0
        params_split = path_split.split("_")[start_nr : nr_of_params + 1]
        param_file = list(map(lambda x: round(float(x.split("-")[1]), 3), params_split))

        df_tar = df["celltype.target.size"].values[:, np.newaxis][
            config.cut_off_start + 1 : config.timesteps - config.cut_off_end
        ]
        df_inf = df["celltype.infected.size"].values[:, np.newaxis][
            config.cut_off_start + 1 : config.timesteps - config.cut_off_end
        ]
        df_cells = np.append(df_tar, df_inf, axis=1)
        df_V = calculate_V(filename_V)
        I_volume = calculate_volume(pathname)[
            config.cut_off_start + 1 : config.timesteps - config.cut_off_end
        ]

        if (
            np.any(df_inf < 1)
            or np.any(np.asarray(param_file) > 1)
            or len(df_inf)
            != (config.timesteps - 1 - config.cut_off_start - config.cut_off_end)
        ):
            invalidIndices.append(path)
            continue
        params[path] = param_file
        dfs[path] = np.append(np.append(df_cells, df_V, axis=1), I_volume, axis=1)

    dfs = np.delete(dfs, invalidIndices, axis=0)
    params = np.delete(params, invalidIndices, axis=0)


def prepare_input(forward_dict, prior_means, prior_stds):
    """Function to configure the simulated quantities (i.e., simulator outputs)
    into a neural network-friendly (BayesFlow) format.
    """

    # Prepare placeholder dict
    out_dict = {}

    # Convert data to logscale

    logdata = np.log1p(forward_dict["sim_data"]).astype(np.float64)

    # Extract prior draws and z-standardize with previously computed means
    params = forward_dict["prior_draws"].astype(np.float64)
    params = (params - prior_means) / prior_stds

    # Remove a batch if it contains nan, inf or -inf
    # idx_keep = np.all(np.isfinite(logdata), axis=(1, 2))

    # Add to keys
    out_dict["summary_conditions"] = logdata
    out_dict["parameters"] = params
    return out_dict
