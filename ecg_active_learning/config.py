import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SRC_DIR = BASE_DIR / "src"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
MODELS_DIR = SRC_DIR / "models"
AL_DIR = SRC_DIR / "active_learning"

# Ensure directories exist
for path in [DATA_DIR, SRC_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR, NOTEBOOKS_DIR, MODELS_DIR, AL_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Experiment Configurations
RANDOM_SEED = 42

# Preprocessing & Windowing
SAMPLING_RATE = 360 # MIT-BIH is 360 Hz
WINDOW_SECONDS = 0.8
WINDOW_SIZE = int(SAMPLING_RATE * WINDOW_SECONDS)

# Data Splitting
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15

# Active Learning Configuration
INITIAL_LABELED_RATIO = 0.05
BATCH_SIZE = 100
MAX_LABEL_RATIO = 1.0  # Go up to 100% of the training pool

# Subsampling (to make the simulation run reasonably fast on a single machine)
MAX_TRAIN_POOL_SIZE = 10000

# Model Configuration
CLASSIFIER = "RandomForest"
N_ESTIMATORS = 100
