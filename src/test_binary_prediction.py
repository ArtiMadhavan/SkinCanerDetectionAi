import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Binary model
MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "efficientnetb0_binary_cancer.keras"
)

# Image to test
IMAGE_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "HAM10000_images",
    "HAM10000_images_part_1",
    "ISIC_0025451.jpg"
)


IMG_SIZE = (224, 224)

print("=" * 55)
print("          SKIN CANCER SCREENING")
print("=" * 55)

# Load model
print("\nLoading binary model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

# Load image
print("\nLoading image...")
img = image.load_img(
    IMAGE_PATH,
    target_size=IMG_SIZE
)

img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# Prediction
print("Running prediction...")

# Do NOT divide by 255
probability = float(
    model.predict(img_array, verbose=0)[0][0]
)

# Binary decision
if probability >= 0.5:
    result = "CANCER"
    confidence = probability * 100
else:
    result = "NOT CANCER"
    confidence = (1 - probability) * 100

# Display result
print("\n" + "=" * 55)
print("                    RESULT")
print("=" * 55)

print(f"\nPrediction  : {result}")
print(f"Confidence  : {confidence:.2f}%")
print(f"Raw score   : {probability:.4f}")

print("\n" + "=" * 55)