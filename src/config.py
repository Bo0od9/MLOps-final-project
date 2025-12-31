import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


HOUR_SECONDS = 60 * 60
DAY_SECONDS = 24 * HOUR_SECONDS

VAL_SIZE = 1 * DAY_SECONDS
TEST_SIZE = 1 * DAY_SECONDS

LAST_TIMESTAMP = 26000000
TEST_TIMESTAMP = LAST_TIMESTAMP - TEST_SIZE

HASH_DIM = 64
LEARNING_RATE = 1e-3
TRAIN_BATCH_SIZE = 256
VALID_BATCH_SIZE = 1024
NUM_EPOCHS = 10

EMBEDDING_DIM = 64
NUM_HEADS = 2
MAX_SEQ_LEN = 100
MIN_SEQ_LEN = 2
DROPOUT_RATE = 0.1
NUM_TRANSFORMER_LAYERS = 2

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
