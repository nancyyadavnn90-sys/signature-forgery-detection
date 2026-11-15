"""
Central configuration for the Signature Forgery Detection project.

Every path, random seed and experiment setting lives here so that the
notebook, the training script and the Streamlit app all use the same values.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"                 # downloaded archive lives here
# extracted images: full_org/ and full_forg/ (can be overridden with the CEDAR_DIR env variable)
CEDAR_DIR = Path(os.environ.get("CEDAR_DIR", DATA_DIR / "CEDAR"))
GENUINE_DIR = CEDAR_DIR / "full_org"       # original_<writer>_<n>.png
FORGED_DIR = CEDAR_DIR / "full_forg"       # forgeries_<writer>_<n>.png

OUTPUT_DIR = ROOT_DIR / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
MODEL_DIR = ROOT_DIR / "models"

FEATURES_CSV = OUTPUT_DIR / "features_raw.csv"    # one row per image (cache)

for _d in (OUTPUT_DIR, FIG_DIR, TABLE_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- experiment
RANDOM_STATE = 42          # one seed for every random operation -> reproducible
TEST_WRITER_FRACTION = 0.20  # 20 % of the 55 writers (11 writers) are held out
CV_FOLDS = 5               # GroupKFold folds (grouped by writer)
N_REFERENCES = 5           # genuine signatures per writer kept "on file" as references

# Labels: the positive class is FORGED, so recall(forged) = 1 - FAR
LABEL_GENUINE = 0
LABEL_FORGED = 1
CLASS_NAMES = ["Genuine", "Forged"]

# Pre-processing
MEDIAN_FILTER_SIZE = 3     # removes salt-and-pepper scanner noise
MIN_COMPONENT_SIZE = 30    # connected components smaller than this (pixels) are noise
NORM_WIDTH = 400           # canvas used for projection-profile features
NORM_HEIGHT = 160

# Correlation filter for feature selection
CORR_THRESHOLD = 0.95
