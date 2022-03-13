import csv
import pickle
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Union

import numpy as np
import pandas as pd


def average_experiment_runs(base_experiment_dir):
    experiment_rounds = base_experiment_dir.iterdir()
    initial_experiment = next(experiment_rounds)
    experiment_rounds = list(base_experiment_dir.iterdir())
    experiment_rounds.sort()

    pkl_files = list(base_experiment_dir.glob("*/metrics/*.pkl"))

    def load_pkl(path):
        with path.open("rb") as f:
            data = pickle.load(f)
        return data

    loaded_pkl_files = defaultdict(list)
    for file_path in pkl_files:
        loaded_pkl_files[file_path.stem].append(load_pkl(file_path))

    def avg_l_dicts(l_dict: List[Dict[str, Union[float, int]]]):
        epoch_result_dict = defaultdict(lambda: 0.0)

        # Sum all dict results from that epoch
        num_dicts = 0
        for dict_ in l_dict:
            if not any(pd.isna(list(dict_.values()))):
                for key, val in dict_.items():
                    epoch_result_dict[key] += val
                num_dicts += 1
            else:
                pass

        return {key: val / num_dicts for key, val in epoch_result_dict.items()}

    avg_pkl_data = {filename: avg_l_dicts(same_epoch_files) for filename, same_epoch_files in
                    loaded_pkl_files.items()}

    experiment_name = base_experiment_dir.name
    avg_eval_metrics = base_experiment_dir / f"avg_evaluation_metrics_{experiment_name}.pkl"
    with avg_eval_metrics.open("wb") as f:
        pickle.dump(avg_pkl_data, f)

    # Avg CSV Files
    csv_files = list(base_experiment_dir.glob("*/metrics/*.csv"))

    def load_csv(csv_file: Path):
        with csv_file.open(newline="") as f:
            data = list(map(float, *csv.reader(f)))
        return data

    loaded_csv_data_arrays = []
    for file_path in csv_files:
        loaded_csv_data_arrays.append(load_csv(file_path))  # array of arrays

    round_datapoints_map = defaultdict(list)
    for data_array in loaded_csv_data_arrays:
        for i, data_point in enumerate(data_array):
            round_datapoints_map[i].append(data_array[i])

    # Note that the dict is ordered by insertion order = index order
    avg_accuracy_data = np.array(list(map(lambda ls: np.average(np.array(ls)), round_datapoints_map.values())))

    avg_accuracy_file = base_experiment_dir / "avg_accuracy_metrics.csv"
    with avg_accuracy_file.open("w") as f:
        writer = csv.writer(f)
        writer.writerow(avg_accuracy_data)

    (base_experiment_dir / "experiment_metadata_file.json").write_text(
        (initial_experiment / "experiment_metadata_file.json").read_text())