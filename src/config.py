"""
Global Configuration & Constants for Heart Disease Diagnosis Project.
Centralized repository parameters to ensure reproducibility across all experiments.
"""

import os
from pathlib import Path

# ==========================================
# 1. REPRODUCIBILITY & RANDOM SEEDS
# ==========================================
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)

# ==========================================
# 2. PATHS CONFIGURATION
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "cleveland.csv"
SPLITS_DIR = DATA_DIR / "splits"

# Cloud dataset URLs
UCI_CLEVELAND_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
)
GDOWN_DRIVE_ID_PART1 = "16HPyuXWXPptt5g3xvS_kR_wXAfjpR1Ju"
GDOWN_DRIVE_FOLDER_ID_PART2 = "1cMoqIDEgGYDVzv8B7cKp3csxujQ4OFp7"

# ==========================================
# 3. DATA SCHEMA & COLUMN SPECIFICATIONS
# ==========================================
TARGET = "target"

# Full 14-column header for Cleveland Dataset
ALL_COLUMNS = [
    "age",       # Age in years
    "sex",       # Sex (1 = male; 0 = female)
    "cp",        # Chest pain type (1: typical, 2: atypical, 3: non-anginal, 4: asymptomatic)
    "trestbps",  # Resting blood pressure (in mm Hg on admission to the hospital)
    "chol",      # Serum cholesterol in mg/dl
    "fbs",       # Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
    "restecg",   # Resting electrocardiographic results (0: normal, 1: ST-T wave abnormality, 2: LV hypertrophy)
    "thalach",   # Maximum heart rate achieved
    "exang",     # Exercise induced angina (1 = yes; 0 = no)
    "oldpeak",   # ST depression induced by exercise relative to rest
    "slope",     # Slope of the peak exercise ST segment (1: upsloping, 2: flat, 3: downsloping)
    "ca",        # Number of major vessels (0-3) colored by fluoroscopy
    "thal",      # Thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect)
    "target"     # Diagnosis of heart disease (0: < 50% diameter narrowing, 1-4: > 50% diameter narrowing)
]

# Feature columns (excluding target)
FEATURE_COLUMNS = [c for c in ALL_COLUMNS if c != TARGET]

# Continuous / Numerical columns
NUMERIC_COLUMNS = ["age", "trestbps", "chol", "thalach", "oldpeak"]

# Discrete / Categorical columns
CATEGORICAL_COLUMNS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

# Columns that may contain string missing values ('?') requiring numeric coercion
COERCE_NUMERIC_COLUMNS = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca", "thal"]

# ==========================================
# 4. SPLIT RATIOS & PARAMETERS
# ==========================================
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# Number of features to select in Feature Selection step
DEFAULT_TOP_K_FEATURES = 10
RAW_NUM_FEATURES = 13

# ==========================================
# 5. HYPERPARAMETER SEARCH GRIDS (BASELINE DEFAULTS)
# ==========================================
KNN_K_RANGE = range(1, 21)
DT_DEPTH_RANGE = range(2, 11)
RF_N_ESTIMATORS_RANGE = range(50, 501, 50)
ADA_N_ESTIMATORS_RANGE = range(50, 501, 50)
GB_N_ESTIMATORS_RANGE = range(50, 501, 50)
XGB_N_ESTIMATORS_RANGE = range(50, 501, 50)
CV_SPLITS = 5
CV_SPLITS_ENSEMBLE = 3
