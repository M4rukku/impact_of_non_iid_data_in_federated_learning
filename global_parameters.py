CELEBA_IMAGE_SIZE = 84
FEMNIST_IMAGE_SIZE = 28

CELEBA_CLASSES = 2
FEMNIST_CLASSES = 62

seq_len: int = 80,
hidden_size: int = 256,
embedding_dim: int = 8,

MODEL_PARAMS = {
    'femnist': (0.0003, 62), # lr, num_classes
    'shakespeare': (0.0003, 80, 80, 256), # lr, seq_len, num_classes, num_hidden
    'celeba': (0.1, 2), # lr, num_classes
}