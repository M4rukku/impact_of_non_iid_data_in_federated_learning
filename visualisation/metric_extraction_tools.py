import pickle
import json
import os
from enum import IntEnum, auto
from pathlib import Path
from typing import List, Dict, Literal, Optional
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def in_debug_environment():
    return "DEBUG" in os.environ and str(True) == os.environ["DEBUG"]


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

    try:
        evaluation_metric_file = \
            list(filter(lambda file: str(file.stem).startswith("avg_evaluation_metrics"), files))[0]
    except IndexError as e:
        if in_debug_environment():
            print(f"Was not able to load averaged data from {str(path_to_experiment)}. Maybe the "
                  f"experiment did not complete successfully? Please REDO!")
        return [], []

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

    try:
        centralised_metrics = {id: get_metric_by_id(id, centralised_evaluation_metrics, True)
                               for id in centralised_evaluation_metrics[0].keys()}

        aggregated_metrics = {id: get_metric_by_id(id, aggregated_evaluation_metrics)
                              for id in aggregated_evaluation_metrics[0].keys()}
    except KeyError as e:
        if in_debug_environment():
            print(f"Data is corrupted for path at {str(path_to_experiment)}. The loss probably "
                  f"diverged to Nan")
        return [], []
    except IndexError as e:
        if in_debug_environment():
            print(e)
            print(f"There is no data (centralised/aggregated) for path at"
                  f" {str(path_to_experiment)}. Please find out what went wrong")
        return [], []
    return centralised_metrics, aggregated_metrics


def get_metadata_of_run(relative_path: Path):
    with (relative_path / "experiment_metadata_file.json").open("r") as f:
        data = json.load(f)
    return data


def get_all_experiments_in_folder(content_dir: Path, multiplicator_centralised=10):
    subfolders = list(content_dir.iterdir())

    centralised_metrics = {}
    aggregated_metrics = {}
    metadata = {}

    for subfolder in subfolders:
        cm, am = get_averaged_metrics_of_run(subfolder, multiplicator_centralised)
        if len(cm) == 0 and len(am) == 0:
            continue
        centralised_metrics[subfolder.name] = cm
        aggregated_metrics[subfolder.name] = am
        metadata[subfolder.name] = get_metadata_of_run(subfolder)

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
                                   multiplicator_centralised=10,
                                   metric_name="accuracy",
                                   evaluation_type=EvaluationType.CENTRALISED,
                                   naming_keys: Optional[List[METADATA_KEY]] = None,
                                   ax=None,
                                   gen_name_from_subfolder_name=None,
                                   title=None,
                                   format=None,
                                   format_list=None,
                                   args=None,
                                   args_list=None,
                                   exclude_from_run_predicate_fun_by_name=None,
                                   win=1,
                                   legend_pos="lower right",
                                   title_fontdict = {"fontsize": 14}
                                   ):
    metadata, centralised_metrics, aggregated_metrics = \
        get_all_experiments_in_folder(content_dir, multiplicator_centralised)

    metrics_dict = centralised_metrics if evaluation_type == EvaluationType.CENTRALISED else \
        aggregated_metrics

    if ax is None:
        fig, ax = get_default_plot()

    for i, (key, sub_metrics_dict) in enumerate(metrics_dict.items()):

        if exclude_from_run_predicate_fun_by_name is not None:
            if exclude_from_run_predicate_fun_by_name(key):
                continue

        if gen_name_from_subfolder_name is not None:
            label = gen_name_from_subfolder_name(key)
        else:
            label = _create_plot_name(key, metadata[key], naming_keys)

        x, y = sub_metrics_dict[metric_name]

        def movingaverage(interval, window_size):
            interval = np.pad(interval, (0, window_size - 1), "edge")
            return np.average(sliding_window_view(interval, window_shape=window_size), axis=1)

        y = movingaverage(y, win)

        added_args = {}
        if args is not None:
            added_args = args
        elif args_list is not None:
            added_args = args_list[i]

        if format_list is not None:
            ax.plot(x, y, format_list[i], **added_args, label=label)
        elif format is not None:
            ax.plot(x, y, format, **added_args, label=label)
        else:
            ax.plot(x, y, label=label, **added_args)

    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.2,
                     box.width, box.height * 0.8])

    # Put a legend below current axis
    ax.legend(loc=legend_pos)

    if title is not None:
        ax.set_title(title, fontdict=title_fontdict)


def get_default_plot():
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    return fig, ax
