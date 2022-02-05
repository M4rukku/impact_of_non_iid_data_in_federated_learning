from pathlib import Path
from sources.datasets.femnist.femnist_client_dataset_factory import FemnistClientDatasetFactory


class FemnistIIDClientDatasetFactory(FemnistClientDatasetFactory):
    def __init__(self, root_data_dir: Path, path_from_data_dir_to_client_dataset="femnist_iid"):
        super().__init__(root_data_dir,
                         path_from_data_dir_to_client_dataset=path_from_data_dir_to_client_dataset)
