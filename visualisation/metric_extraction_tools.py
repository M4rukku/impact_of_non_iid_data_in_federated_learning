import pickle
import json
from enum import IntEnum, auto
from pathlib import Path
from typing import List, Dict, Literal, Optional
import matplotlib.pyplot as plt


def load_metrics(filepath: Path):
    with filepath.open("rb") as f:
        data = pickle.load(f)
    return data


def get_metrics_for_individual_run(content_root, path_to_run, multiplicator_centralised=10):
    data_folder = content_root / path_to_run / "metrics"

    centralised_evaluation_metrics_filter = \
        lambda w: w.stem.startswith("centralised_evaluation_metrics")
    aggregated_evaluation_metrics_filter = \
        lambda w: w.stem.startswith("evaluation_metrics")

    files = list(filter(lambda f: f.suffix == ".pkl", data_folder.iterdir()))
    files.sort(key=lambda m: int(m.stem.split("_")[-1]))

    centralised_evaluation_metric_files = list(filter(centralised_evaluation_metrics_filter, files))
    aggregated_evaluation_metric_files = list(filter(aggregated_evaluation_metrics_filter, files))

    centralised_evaluation_metrics = list(map(load_metrics, centralised_evaluation_metric_files))
    aggregated_evaluation_metrics = list(map(load_metrics, aggregated_evaluation_metric_files))

    def get_metric_by_id(id: str, metrics: List[Dict], centralised=False):
        y = list(map(lambda d: d[id], metrics))
        multiplicator = multiplicator_centralised if centralised else 1
        upper_lim = multiplicator * (len(y) + 1)
        return list(range(1 * multiplicator, upper_lim, multiplicator)), y

    centralised_metrics = {id: get_metric_by_id(id, centralised_evaluation_metrics, True)
                           for id in centralised_evaluation_metrics[0].keys()}

    aggregated_metrics = {id: get_metric_by_id(id, aggregated_evaluation_metrics)
                          for id in aggregated_evaluation_metrics[0].keys()}

    return centralised_metrics, aggregated_metrics


def get_averaged_metrics_of_run(path_to_experiment: Path, multiplicator_centralised=5):
    files = path_to_experiment.iterdir()
    evaluation_metric_file = \
        list(filter(lambda file: str(file.stem).startswith("avg_evaluation_metrics"), files))[0]

    data = None
    with evaluation_metric_file.open("rb") as f:
        data = pickle.load(f)

    metrics = list(data.items())
    centralised_evaluation_metrics_filter = \
        lambda w: w[0].startswith("centralised_evaluation_metrics")
    aggregated_evaluation_metrics_filter = \
        lambda w: w[0].startswith("evaluation_metrics")

    metrics.sort(key=lambda m: int(m[0].split("_")[-1]))
    snd = lambda p: p[1]

    centralised_evaluation_metrics = list(map(snd,
                                              filter(centralised_evaluation_metrics_filter,
                                                     metrics)))
    aggregated_evaluation_metrics = list(map(snd,
                                             filter(aggregated_evaluation_metrics_filter, metrics)))

    def get_metric_by_id(id: str, metrics: List[Dict], centralised=False):
        y = list(map(lambda d: d[id], metrics))
        multiplicator = multiplicator_centralised if centralised else 1
        upper_lim = multiplicator * (len(y) + 1)
        return list(range(1 * multiplicator, upper_lim, multiplicator)), y

    centralised_metrics = {id: get_metric_by_id(id, centralised_evaluation_metrics, True)
                           for id in centralised_evaluation_metrics[0].keys()}

    aggregated_metrics = {id: get_metric_by_id(id, aggregated_evaluation_metrics)
                          for id in aggregated_evaluation_metrics[0].keys()}

    return centralised_metrics, aggregated_metrics


def get_metadata_of_run(relative_path: Path):
    with (relative_path / "experiment_metadata_file.json").open("r") as f:
        data = json.load(f)
    return data


def get_all_experiments_in_folder(content_dir: Path, multiplicator_centralised=5):
    subfolders = list(content_dir.iterdir())

    centralised_metrics = {}
    aggregated_metrics = {}
    metadata = {}

    for subfolder in subfolders:
        cm, am = get_averaged_metrics_of_run(subfolder, multiplicator_centralised)
        centralised_metrics[subfolder.stem] = cm
        aggregated_metrics[subfolder.stem] = am
        metadata[subfolder.stem] = get_metadata_of_run(subfolder)

    return metadata, centralised_metrics, aggregated_metrics


class EvaluationType(IntEnum):
    CENTRALISED = auto()
    AGGREGATED = auto()


METADATA_KEY = Literal['strategy_name',
                       'clients_per_round',
                       'num_clients',
                       'num_rounds',
                       'batch_size',
                       'local_epochs',
                       'val_steps',
                       "optimizer_name",
                       "optimizer_lr",
                       'local_learning_rate']

metdata_keys_abbreviation_map = {'strategy_name': "sn",
                                 'clients_per_round': "cpr",
                                 'num_clients': "nc",
                                 'num_rounds': "nr",
                                 'batch_size': "bs",
                                 'local_epochs': "le",
                                 'val_steps': "vs",
                                 "optimizer_name": "on",
                                 "optimizer_lr": "lr"}


def _create_plot_name(file_name, metadata_dict, naming_keys):
    if naming_keys is None:
        return file_name
    else:
        result = ""
        for key in naming_keys:
            if key == "optimizer_name":
                opt_config = metadata_dict["optimizer_config"]
                result += f"_{opt_config['name']}"
            elif key == "optimizer_lr":
                opt_config = metadata_dict["optimizer_lr"]
                result += f"_{metdata_keys_abbreviation_map['optimizer_lr']}" \
                          f"{opt_config['learning_rate']}"
            elif key == "strategy_name":
                strategy_basename = metadata_dict["strategy_name"].split("_")[-1]
                result += f"_{strategy_basename}"
            else:
                val = metadata_dict[key]
                abbr = metdata_keys_abbreviation_map[key]
                result += f"_{val}{abbr}"

        result = result.strip(" _")
        return result


def plot_all_experiments_in_folder(content_dir: Path,
                                   multiplicator_centralised=5,
                                   metric_name="accuracy",
                                   evaluation_type=EvaluationType.CENTRALISED,
                                   naming_keys: Optional[List[METADATA_KEY]] = None):
    metadata, centralised_metrics, aggregated_metrics = \
        get_all_experiments_in_folder(content_dir, multiplicator_centralised)

    metrics_dict = centralised_metrics if evaluation_type == EvaluationType.CENTRALISED else \
        aggregated_metrics

    fig, ax = plt.subplots(1, 1, figsize=(6, 2))

    for key, sub_metrics_dict in metrics_dict.items():
        ax.plot(*sub_metrics_dict[metric_name],
                label=_create_plot_name(key, metadata[key], naming_keys))
    ax.legend()
