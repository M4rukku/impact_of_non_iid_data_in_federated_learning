from pathlib import Path

from sources.datasets.shakespeare.shakespeare_client_dataset_factory import \
    ShakespeareClientDatasetFactory


class ShakespeareIIDClientDatasetFactory(ShakespeareClientDatasetFactory):
    def __init__(self, root_data_dir: Path, path_from_data_dir_to_client_dataset="shakespeare_iid"):
        super().__init__(root_data_dir,
                         path_from_data_dir_to_client_dataset=path_from_data_dir_to_client_dataset)
