import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
HAM_IMAGES_DIR = RAW_DATA_DIR / "HAM10000_images"
HAM_METADATA_CSV = HAM_IMAGES_DIR / "HAM10000_metadata.csv"

# Images are split into two parts
HAM_IMAGES_PART_1 = HAM_IMAGES_DIR / "HAM10000_images_part_1"
HAM_IMAGES_PART_2 = HAM_IMAGES_DIR / "HAM10000_images_part_2"

MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"

# Model specific output paths
MODEL_SAVE_PATH = MODELS_DIR / "efficientnetb0_best.keras"
CLASS_MAP_PATH = MODELS_DIR / "class_names.json"

# Model Configuration
MODEL_NAME = "EfficientNetB0"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.5

# Classes (HAM10000 specific exact ordering optional, but keeping consistency)
CLASSES = ['mel', 'bcc', 'akiec', 'nv', 'bkl', 'df', 'vasc']
NUM_CLASSES = len(CLASSES)

CLASS_NAMES_MAP = {
    'mel': 'Melanoma',
    'bcc': 'Basal Cell Carcinoma',
    'akiec': 'Actinic Keratoses',
    'nv': 'Melanocytic Nevus',
    'bkl': 'Benign Keratosis',
    'df': 'Dermatofibroma',
    'vasc': 'Vascular Lesion'
}

# Random Seed for Reproducibility
RANDOM_SEED = 42
