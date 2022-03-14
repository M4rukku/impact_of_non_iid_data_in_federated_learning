from sources.datasets.client_dataset_definitions.client_dataset_processors.client_dataset_processor import \
    ClientDatasetProcessor


class Cifar10LdaClientDatasetProcessor(ClientDatasetProcessor):

    def process_x(self, raw_x_batch):
        return raw_x_batch / 255.0 - 0.5

    def process_y(self, raw_y_batch):
        return raw_y_batch
