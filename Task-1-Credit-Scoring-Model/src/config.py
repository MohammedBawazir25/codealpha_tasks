from pathlib import Path

# ==================================================
# PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = BASE_DIR / "models"

REPORT_DIR = BASE_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

LOG_DIR = BASE_DIR / "logs"

# ==================================================
# DATA FILES
# ==================================================

TRAIN_DATA = RAW_DATA_DIR / "cs-training.csv"
TEST_DATA = RAW_DATA_DIR / "cs-test.csv"

PROCESSED_DATA = PROCESSED_DATA_DIR / "processed.csv"

MODEL_FILE = MODEL_DIR / "best_model.pkl"

SCALER_FILE = MODEL_DIR / "scaler.pkl"

# ==================================================
# ML SETTINGS
# ==================================================

TARGET = "SeriousDlqin2yrs"

RANDOM_STATE = 42

TEST_SIZE = 0.20

CV = 5

# ==================================================
# CREATE DIRECTORIES
# ==================================================

for folder in [

    DATA_DIR,

    RAW_DATA_DIR,

    PROCESSED_DATA_DIR,

    MODEL_DIR,

    REPORT_DIR,

    FIGURE_DIR,

    LOG_DIR

]:

    folder.mkdir(parents=True, exist_ok=True)
# Backward compatibility

TRAIN_DATA_PATH = TRAIN_DATA
TEST_DATA_PATH = TEST_DATA
PROCESSED_DATA_PATH = PROCESSED_DATA

MODEL_PATH = MODEL_FILE