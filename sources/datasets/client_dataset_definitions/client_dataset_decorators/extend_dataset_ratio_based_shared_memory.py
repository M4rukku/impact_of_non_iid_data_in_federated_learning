import logging
import math
from typing import Tuple

import numpy as np

from sources.datasets.client_dataset_definitions.client_dataset_loaders.client_dataset_loader import \
    DatasetComponents
from sources.utils.dataset import Dataset
from sources.datasets.client_dataset_definitions.client_dataset_decorators.base_client_dataset_decorator import \
    BaseClientDatasetDecorator
from sources.datasets.client_dataset_definitions.client_dataset import ClientDataset

DEFAULT_CLIENT_DS_GSD_PART_RATIOS = (1.0, 0.0, 0.0)  # train test validate


class ExtendDatasetRatioBasedSharedMemory(BaseClientDatasetDecorator):
    # Since this class creates deterministic datasets (as in global data will always be before client data,
    # need to ensure that we shuffle the data during fitting (see BaseClient)

    def __init__(self,
                 client_dataset: ClientDataset,
                 shared_dataset: Dataset,
                 client_gsd_partition_sz_ratios: Tuple[float, float, float]
                 = DEFAULT_CLIENT_DS_GSD_PART_RATIOS
                 ):
        super().__init__(client_dataset)

        self.client_gsd_partition_sz_ratios = client_gsd_partition_sz_ratios
        self.client_gsd_part_sz_ratio_train = client_gsd_partition_sz_ratios[0]
        self.client_gsd_part_sz_ratio_test = client_gsd_partition_sz_ratios[1]
        self.client_gsd_part_sz_ratio_val = client_gsd_partition_sz_ratios[2]

        self.client_gsd_part_sz_ratios_map = {
            DatasetComponents.TRAIN: self.client_gsd_part_sz_ratio_train,
            DatasetComponents.TEST: self.client_gsd_part_sz_ratio_test,
            DatasetComponents.VALIDATION: self.client_gsd_part_sz_ratio_val
        }

        self.shared_dataset = shared_dataset
        self.shared_dataset_sizes = {
            DatasetComponents.TRAIN: len(shared_dataset.train["x"]),
            DatasetComponents.TEST: len(shared_dataset.test["x"]),
            DatasetComponents.VALIDATION: len(shared_dataset.validation["x"])
        }

        self.component_selection_map = {
            DatasetComponents.TRAIN: None,
            DatasetComponents.TEST: None,
            DatasetComponents.VALIDATION: None
        }

    def get_selection_for_dataset_component(self,
                                            decorated_dataset_component_length: int,
                                            component: DatasetComponents):
        if self.component_selection_map[component] is not None:
            return self.component_selection_map[component]

        shared_dataset_component_size = self.shared_dataset_sizes[component]
        client_ds_gsd_partition_ratio = self.client_gsd_part_sz_ratios_map[component]
        target_size = math.ceil(decorated_dataset_component_length * client_ds_gsd_partition_ratio)

        if target_size > shared_dataset_component_size:
            logging.warning(f"Attempted to enrich local dataset with globally shared dataset, "
                            f"size ratio between client and gsd partition is supposed to be "
                            f" {client_ds_gsd_partition_ratio}. The client dataset has length "
                            f"{decorated_dataset_component_length}, so the amount of data to add is"
                            f" {target_size}. Component {component.name} for the gsd only has "
                            f"size {shared_dataset_component_size}. Enriching the local dataset "
                            f"with all data available...")
            target_size = shared_dataset_component_size

        rng = np.random.default_rng()
        component_selection = rng.choice(shared_dataset_component_size,
                                         target_size,
                                         replace=False)
        self.component_selection_map[component] = component_selection
        return component_selection

    @property
    def training_data_x(self):
        """Returns the Training Data as an array of samples"""
        if self.client_gsd_part_sz_ratio_train > 0:
            client_training_data_x = self.client_dataset.training_data_x
            selection = self.get_selection_for_dataset_component(len(client_training_data_x),
                                                                 DatasetComponents.TRAIN)
            data_x = self.shared_dataset.train["x"][selection]
            return np.concatenate((
                data_x,
                client_training_data_x
            ))
        else:
            return self.client_dataset.training_data_x

    @property
    def training_data_y(self):
        """Returns the Classifications for the Training Data as array"""
        if self.client_gsd_part_sz_ratio_train > 0:
            client_training_data_y = self.client_dataset.training_data_y
            selection = self.get_selection_for_dataset_component(len(client_training_data_y),
                                                                 DatasetComponents.TRAIN)
            data_y = self.shared_dataset.train["y"][selection]
            return np.concatenate((
                data_y,
                client_training_data_y
            ))
        else:
            return self.client_dataset.training_data_y

    @property
    def test_data(self):
        """Returns the Test Data as pair of arrays containing the samples x,
         and classification y"""
        return [self.test_data_x, self.test_data_y]

    @property
    def test_data_x(self):
        """Returns the Test Data as an array of samples"""

        if self.client_gsd_part_sz_ratio_test > 0:
            client_test_data_x = self.client_dataset.test_data_x
            selection = self.get_selection_for_dataset_component(len(client_test_data_x),
                                                                 DatasetComponents.TEST)
            data_x = self.shared_dataset.test["x"][selection]

            return np.concatenate((
                data_x,
                client_test_data_x
            ))
        else:
            return self.client_dataset.test_data_x

    @property
    def test_data_y(self):
        """Returns the Classifications for the Test Data as array"""
        if self.client_gsd_part_sz_ratio_test > 0:
            client_test_data_y = self.client_dataset.test_data_y
            selection = self.get_selection_for_dataset_component(len(client_test_data_y),
                                                                 DatasetComponents.TEST)
            data_y = self.shared_dataset.test["y"][selection]

            return np.concatenate((
                data_y,
                client_test_data_y
            ))
        else:
            return self.client_dataset.test_data_y

    @property
    def validation_data(self):
        """Returns the Validation Data as pair of arrays containing the
        samples x,
         and classification y"""
        return [self.validation_data_x, self.validation_data_y]

    @property
    def validation_data_x(self):
        """Returns the Validation Data as an array of samples"""
        if self.client_gsd_part_sz_ratio_val > 0:
            client_val_data_x = self.client_dataset.validation_data_x
            selection = self.get_selection_for_dataset_component(len(client_val_data_x),
                                                                 DatasetComponents.VALIDATION)
            data_x = self.shared_dataset.validation["x"][selection]

            return np.concatenate((
                data_x,
                client_val_data_x
            ))
        else:
            return self.client_dataset.validation_data_x

    @property
    def validation_data_y(self):
        """Returns the Classifications for the Validation Data as array"""
        if self.client_gsd_part_sz_ratio_val > 0:
            client_val_data_y = self.client_dataset.validation_data_y
            selection = self.get_selection_for_dataset_component(len(client_val_data_y),
                                                                 DatasetComponents.VALIDATION)
            data_y = self.shared_dataset.validation["y"][selection]

            return np.concatenate((
                data_y,
                client_val_data_y
            ))
        else:
            return self.client_dataset.validation_data_y
