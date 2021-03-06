# CELEBA
from sources.dataset_creation_utils.create_lda_dataset_utils import get_lda_cifar10_dataset_name

CELEBA_IMAGE_SIZE = 84
CELEBA_CLASSES = 2

# FEMNIST
FEMNIST_IMAGE_SIZE = 28
FEMNIST_CLASSES = 62

# SHAKESPEARE

LEAF_CHARACTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"

# CIFAR10

DEFAULT_CONCENTRATIONS_CIFAR10 = [0.001, 0.5, 100.0]
TOTAL_IMAGES_CIFAR10 = 60000
DEFAULT_PARTITIONS_CIFAR10 = 100
DEFAULT_IID_DATASET_SIZE_CIFAR10 = 0.005  # 300 images

CIFAR_10_CLASSES = 10
CIFAR_10_IMAGE_SIZE = 32
CIFAR_10_IMAGE_DIMENSIONS = 3

# DATASET INDEPENDENT CONSTANTS
GLOBALLY_SHARED_DATASET_FRACTION = 0.05
DEFAULT_RATIOS_DATASET_SIZE_GD_PARTITION = [0.5, 1.0, 2.0]


NUM_DATA_SAMPLES_USED_PER_CLIENT_FROM_GLOBAL_DATASET_CIFAR10 = 250
NUM_DATA_SAMPLES_USED_PER_CLIENT_FROM_GLOBAL_DATASET_FEMNIST = 250
NUM_DATA_SAMPLES_USED_PER_CLIENT_FROM_GLOBAL_DATASET_CELEBA = 30
NUM_DATA_SAMPLES_USED_PER_CLIENT_FROM_GLOBAL_DATASET_SHAKESPEARE = 3000

DEFAULT_TRAIN_SPLIT = 0.8
DEFAULT_TEST_SPLIT = 0.15
DEFAULT_VALIDATION_SPLIT = 0.05

# Dataset Names

LEAF_DATASET_NAME_LIST = ["femnist", "celeba", "shakespeare"]

CIFAR10_CLIENT_DATASET_NAMES = \
    [get_lda_cifar10_dataset_name(concentration, partition)
     for concentration, partition in zip(DEFAULT_CONCENTRATIONS_CIFAR10,
                                         [DEFAULT_PARTITIONS_CIFAR10] * len(DEFAULT_CONCENTRATIONS_CIFAR10))]

CLIENT_DATASETS_NAME_LIST = LEAF_DATASET_NAME_LIST + CIFAR10_CLIENT_DATASET_NAMES
IID_DATASET_NAME_LIST = LEAF_DATASET_NAME_LIST + CIFAR10_CLIENT_DATASET_NAMES

# Number of Clients we consider each round (if it is less than the total number)
FEMNIST_CLIENTS_TO_CONSIDER = 1000
CELEBA_CLIENTS_TO_CONSIDER = 5000
SHAKESPEARE_CLIENTS_TO_CONSIDER = 798

CLIENT_SUBSET_TO_CONSIDER = {
    "femnist" : FEMNIST_CLIENTS_TO_CONSIDER,
    "celeba": CELEBA_CLIENTS_TO_CONSIDER,
    "shakespeare": SHAKESPEARE_CLIENTS_TO_CONSIDER,
    "default": None
}

# There is too much data in Shakespeare
MAX_DATA_TRAIN_SHAKESPEARE = 2000
MAX_DATA_TEST_SHAKESPEARE = int(MAX_DATA_TRAIN_SHAKESPEARE / DEFAULT_TRAIN_SPLIT *
                                DEFAULT_TEST_SPLIT)
MAX_DATA_VAL_SHAKESPEARE = int(MAX_DATA_TRAIN_SHAKESPEARE / DEFAULT_TRAIN_SPLIT *
                                DEFAULT_VALIDATION_SPLIT)

# global dataset fractions
DEFAULT_FEMNIST_DATASET_FRACTION: float = 0.05
DEFAULT_CELEBA_DATASET_FRACTION: float = 0.05
DEFAULT_SHAKESPEARE_DATASET_FRACTION: float = 0.01

DATASET_NAME_DEFAULT_FRACTION_DICT = {
    "celeba": DEFAULT_CELEBA_DATASET_FRACTION,
    "femnist": DEFAULT_FEMNIST_DATASET_FRACTION,
    "shakespeare": DEFAULT_SHAKESPEARE_DATASET_FRACTION,
    "default": 1.0
}

DATASET_NAME_GLOBAL_SHARED_FRACTION_DICT = {
    "celeba": GLOBALLY_SHARED_DATASET_FRACTION,
    "femnist": GLOBALLY_SHARED_DATASET_FRACTION,
    "shakespeare": GLOBALLY_SHARED_DATASET_FRACTION,
    "default": 0.05
}
