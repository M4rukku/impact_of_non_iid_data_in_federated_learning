DEFAULT_SEED: int = 0

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
import tensorflow as tf


def set_seeds(seed=DEFAULT_SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.keras.utils.set_random_seed(seed)


def set_global_determinism(seed=DEFAULT_SEED):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
