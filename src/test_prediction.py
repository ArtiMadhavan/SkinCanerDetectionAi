import sys
from pathlib import Path

import tensorflow as tf

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import predict_image
from utils.config import MODELS_DIR


# Load final trained model
model_path = MODELS_DIR / "efficientnetb0_best.keras"

print("Loading model...")
model = tf.keras.models.load_model(model_path)

# Test image
image_path = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "HAM10000_images"
    / "HAM10000_images_part_1"
    / "ISIC_0025451.jpg"
)

print("\nRunning prediction...\n")

results, _ = predict_image(
    model=model,
    img_path=str(image_path)
)

print("========================================")
print("TOP 3 PREDICTIONS")
print("========================================")

for result in results:
    print(
        f"{result['class_short']} - "
        f"{result['class_full']} - "
        f"{result['probability']:.2f}%"
    )