# Project-specific settings live here.
import pytorch_lightning as pl
import numpy as np

# Ensure reproducibility
pl.seed_everything(0)

# data subset names
TRAIN_SUBSET = "training"
VAL_SUBSET = "validation"
TEST_SUBSET = "test"

# General dataset parameters
# TOTAL_SAMPLES = 10000
# TRAIN_RATIO = 0.8
# VAL_RATIO = 0.1
TOTAL_SAMPLES = 180
TRAIN_RATIO = 0.9
VAL_RATIO = 0.05
TRAIN_SET_SIZE = int(TOTAL_SAMPLES * TRAIN_RATIO)
VAL_SET_SIZE = int(TOTAL_SAMPLES * VAL_RATIO)
TEST_SET_SIZE = TOTAL_SAMPLES - (TRAIN_SET_SIZE + VAL_SET_SIZE)
LANDMARK_MEAN = 0
LANDMARK_STD_DEV = 1

# Models
ARCHITECTURE_TYPE = "cmnet"

# Training parameters
NUM_CPUS = 8
MAX_EPOCHS = 5
LEARNING_RATE = 1e-4
BATCH_SIZE = 3

# Paths
IS_RUNNING_ON_SERVER = False

if IS_RUNNING_ON_SERVER is True:
    ROOT_DIR = "/Volumes/scratchdata/roberto/place-to-store-things/"
    MODEL_DIR = ROOT_DIR + "models/"
    RESULTS_DIR = ROOT_DIR + "evaluation/"
else:
    ROOT_DIR = "/workspace/data/landmark-dewarping/"
    DATA_DIR = ROOT_DIR + "tmp_data_store/"
    MODEL_DIR = ROOT_DIR + "models/"
    RESULTS_DIR = ROOT_DIR + "evaluation/"
