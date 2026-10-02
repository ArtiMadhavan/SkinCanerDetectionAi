import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ============================================================
# PATHS
# ============================================================

THREE_CLASS_MODEL = "models/efficientnetb0_three_class.keras"
BINARY_MODEL = "models/efficientnetb0_binary_cancer.keras"

IMAGE_PATH = r"data/raw/HAM10000_images/HAM10000_images_part_1/ISIC_0025964.jpg"


# ============================================================
# LOAD MODELS
# ============================================================

print("\nLoading models...")

three_model = tf.keras.models.load_model(
    THREE_CLASS_MODEL
)

binary_model = tf.keras.models.load_model(
    BINARY_MODEL
)

print("Models loaded successfully.")


# ============================================================
# LOAD IMAGE
# ============================================================

image = Image.open(IMAGE_PATH).convert("RGB")

image = image.resize((224, 224))

image_array = np.array(image)

image_array = np.expand_dims(
    image_array,
    axis=0
)

print("\nImage loaded:")
print("Shape:", image_array.shape)


# ============================================================
# 3-CLASS PREDICTION
# ============================================================

three_prediction = three_model.predict(
    image_array,
    verbose=0
)[0]

print("\n" + "=" * 60)
print("3-CLASS MODEL")
print("=" * 60)

print(
    "NOT A CANCER CELL : "
    f"{three_prediction[0] * 100:.2f}%"
)

print(
    "CANCER             : "
    f"{three_prediction[1] * 100:.2f}%"
)

print(
    "NEEDS MEDICAL CHECK: "
    f"{three_prediction[2] * 100:.2f}%"
)

three_index = np.argmax(three_prediction)

three_names = [
    "NOT A CANCER CELL",
    "CANCER",
    "NEEDS MEDICAL CHECK"
]

print(
    "\n3-Class Prediction:",
    three_names[three_index]
)


# ============================================================
# BINARY PREDICTION
# ============================================================

binary_prediction = binary_model.predict(
    image_array,
    verbose=0
)[0]

print("\n" + "=" * 60)
print("BINARY MODEL")
print("=" * 60)

print(
    "NOT CANCER : "
    f"{binary_prediction[0] * 100:.2f}%"
)

print(
    "CANCER     : "
    f"{binary_prediction[1] * 100:.2f}%"
)

binary_index = np.argmax(binary_prediction)

binary_names = [
    "NOT CANCER",
    "CANCER"
]

print(
    "\nBinary Prediction:",
    binary_names[binary_index]
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(
    "3-Class:",
    three_names[three_index]
)

print(
    "Binary:",
    binary_names[binary_index]
)

print("=" * 60)