import os

# Project Root (healtheat_vision)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Data Directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, "train_images")
TRAIN_ANNOTATIONS_DIR = os.path.join(DATA_DIR, "train_annotations")
TEST_IMAGES_DIR = os.path.join(DATA_DIR, "test_images")

# YOLO Conversion Output
YOLO_DIR = os.path.join(DATA_DIR, "yolo")
YOLO_IMAGES_DIR = os.path.join(YOLO_DIR, "images")
YOLO_LABELS_DIR = os.path.join(YOLO_DIR, "labels")

# Splits
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
SPLIT_TRAIN = os.path.join(SPLITS_DIR, "train.txt")
SPLIT_VAL = os.path.join(SPLITS_DIR, "val.txt")

# Artifacts
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
CLASS_MAP_PATH = os.path.join(ARTIFACTS_DIR, "class_map.csv")
MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")
VIZ_DIR = os.path.join(ARTIFACTS_DIR, "viz_predictions")

# Configs
CONFIGS_DIR = os.path.join(PROJECT_ROOT, "configs")
YOLO_DATA_YAML = os.path.join(CONFIGS_DIR, "yolo_data.yaml")

# Scripts
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")

# Submissions
SUBMISSIONS_DIR = os.path.join(PROJECT_ROOT, "submissions")

def ensure_dirs():
    """Ensure all critical directories exist."""
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(VIZ_DIR, exist_ok=True)
    os.makedirs(SPLITS_DIR, exist_ok=True)
    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
