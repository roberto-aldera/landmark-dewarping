# Project-specific settings live here.
import pytorch_lightning as pl
import numpy as np

# Ensure reproducibility
pl.seed_everything(0)

K_MAX_MATCHES = 600
# Setting this range to 1 effectively turns off normalisation
MAX_LANDMARK_RANGE_METRES = 165  # 165  # 3768 bins at a resolution of 0.0438 cm per bin

# General dataset parameters
TOTAL_SAMPLES = 150
TRAIN_RATIO = 0.99
TRAIN_SET_SIZE = int(TOTAL_SAMPLES * TRAIN_RATIO)
VAL_SET_SIZE = TOTAL_SAMPLES - TRAIN_SET_SIZE
# TEST_SET_SIZE = TOTAL_SAMPLES - (TRAIN_SET_SIZE + VAL_SET_SIZE)

# Models
ARCHITECTURE_TYPE = "pointnet"

# Training parameters
NUM_CPUS = 8
MAX_EPOCHS = 5
LEARNING_RATE = 1e-6
BATCH_SIZE = 128

# Metrics
K_RADAR_INDEX_OFFSET = 0
AUX0_NAME = "CME-means"
AUX1_NAME = "Corrected-landmarks"

# Bools
IS_RUNNING_ON_SERVER = False

if IS_RUNNING_ON_SERVER:
    DO_PLOTS_IN_LOSS = False
    DO_PLOTS_IN_FORWARD_PASS = False
    DO_CORRECTION_MAGNITUDE_PLOTS = False
    DO_GRADIENT_PLOTS = False
else:
    DO_PLOTS_IN_LOSS = False
    DO_PLOTS_IN_FORWARD_PASS = False
    DO_CORRECTION_MAGNITUDE_PLOTS = False
    DO_GRADIENT_PLOTS = False

if IS_RUNNING_ON_SERVER is True:
    ROOT_DIR = "/Volumes/scratchdata/roberto/landmark-dewarping/"
    DATA_DIR = "/workspace/landmark-data/"
    MODEL_DIR = ROOT_DIR + "models/"
    RESULTS_DIR = ROOT_DIR + "evaluation/"
else:
    ROOT_DIR = "/workspace/data/landmark-dewarping/"
    DATA_DIR = ROOT_DIR + "landmark-data/"
    MODEL_DIR = ROOT_DIR + "models/"
    RESULTS_DIR = ROOT_DIR + "evaluation/"

PLOTTING_ITR = 0
CORRECTION_PLOTTING_ITR = 0
GRAD_PLOTTING_ITR = 0
