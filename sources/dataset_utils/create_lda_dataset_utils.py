import numpy as np


def get_random_id_splits(total: int,
                         test_ratio: float,
                         val_ratio: float,
                         shuffle: bool = True
                         ):
    if isinstance(total, int):
        if total < 4:
            raise RuntimeError("Datasets must have at least size 3!")
        indices = list(range(total))
    else:
        indices = total

    split_test = int(np.ceil(test_ratio * len(indices)))
    split_val = int(np.ceil(val_ratio * len(indices)))

    if shuffle:
        np.random.shuffle(indices)
    return (indices[(split_test + split_val):],
            indices[split_val:(split_test + split_val)],
            indices[:split_val])


def get_lda_dataset_name(concentration_: float, num_partitions_: int):
    return f"cifar10_{str(num_partitions_)}p_{format(concentration_, '.3f')}c"
