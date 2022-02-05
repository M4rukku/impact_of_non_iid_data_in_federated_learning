from typing import Literal, get_args, Dict, Optional
import numpy.typing as npt

from sources.dataset_utils.dataset import Dataset
from sources.datasets.client_dataset_definitions.client_dataset_processors.client_dataset_processor import ClientDatasetProcessor
from sources.ray_tooling.ray_store_manager import RayObjectStoreAccessor

ClientDatasetComponents = Literal["training_data_x",
                                  "training_data_y",
                                  "test_data_x",
                                  "test_data_y",
                                  "validation_data_x",
                                  "validation_data_y"]

dataset_component_names = list(get_args(ClientDatasetComponents))


def generate_dataset_component_identifier(dataset_identifier: str,
                                          component_identifier: ClientDatasetComponents):
    return f"{dataset_identifier}_{component_identifier}"


# noinspection PyTypeChecker
def get_all_components_from_dataset(
        dataset: Dataset,
        client_dataset_processor: Optional[ClientDatasetProcessor] = None
) -> Dict[ClientDatasetComponents, npt.ArrayLike]:
    if client_dataset_processor is None:
        return dict(training_data_x=dataset.train["x"],
                    training_data_y=dataset.train["y"],
                    test_data_x=dataset.test["x"],
                    test_data_y=dataset.test["y"],
                    validation_data_x=dataset.validation["x"],
                    validation_data_y=dataset.validation["y"])
    else:
        return dict(training_data_x=client_dataset_processor.process_x(dataset.train["x"]),
                    training_data_y=client_dataset_processor.process_y(dataset.train["y"]),
                    test_data_x=client_dataset_processor.process_x(dataset.test["x"]),
                    test_data_y=client_dataset_processor.process_y(dataset.test["y"]),
                    validation_data_x=client_dataset_processor.process_x(dataset.validation["x"]),
                    validation_data_y=client_dataset_processor.process_y(dataset.validation["y"]))


def load_dataset_into_ray(
        dataset_identifier: str,
        dataset: Dataset,
        client_dataset_processor: Optional[ClientDatasetProcessor] = None
) -> None:
    '''
    Args:
        client_dataset_processor: Should the dataset not yet have been preprocessed, it may be
        handled at this stage
        dataset_identifier: Identifier of the dataset i.e. celeba (used to create name)
        dataset: PREPROCESSED! IID Dataset to load into ray, it will be split into differnet
        components which you can then fetch with fetch components from ray
    '''

    dataset_components = get_all_components_from_dataset(dataset, client_dataset_processor)

    for component_name, component_data in dataset_components.items():
        RayObjectStoreAccessor.save_object_in_manager(
            generate_dataset_component_identifier(dataset_identifier, component_name),
            component_data
        )


def fetch_dataset_component_from_ray(
        dataset_identifier: str, dataset_component: ClientDatasetComponents
):
    return RayObjectStoreAccessor.get_object_in_manager(
        generate_dataset_component_identifier(dataset_identifier, dataset_component)
    )


def delete_dataset_components_from_ray(dataset_identifier: str):
    for component_name in dataset_component_names:
        RayObjectStoreAccessor.delete_object_in_manager(
            generate_dataset_component_identifier(dataset_identifier, component_name)
        )
