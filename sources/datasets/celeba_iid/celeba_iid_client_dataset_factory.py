from pathlib import Path

from sources.datasets.celeba.celeba_client_dataset_factory import CelebaClientDatasetFactory


class CelebaIIDClientDatasetFactory(CelebaClientDatasetFactory):
    def __init__(self, root_data_dir: Path, path_from_data_dir_to_client_dataset="celeba_iid"):
        super().__init__(root_data_dir,
                         path_from_data_dir_to_client_dataset=path_from_data_dir_to_client_dataset)
