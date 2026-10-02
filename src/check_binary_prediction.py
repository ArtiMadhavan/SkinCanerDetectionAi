import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import tensorflow as tf
from PIL import Image


MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "efficientnetb0_binary_cancer.keras"
)

IMAGE_PATH = os.path.join(
    PROJECT_ROOT,
    "cancer.jpg"
)


print("=" * 60)
print("BINARY MODEL DIAGNOSTIC")
print("=" * 60)

print("\nModel:")
print(MODEL_PATH)

print("\nImage:")
print(IMAGE_PATH)


# ------------------------------------------------------------
# CHECK FILES
# ------------------------------------------------------------

if not os.path.exists(MODEL_PATH):
    print("\nERROR: Model not found.")
    sys.exit()

if not os.path.exists(IMAGE_PATH):
    print("\nERROR: cancer.jpg not found.")
    print("Put your test cancer image in the project folder.")
    sys.exit()


# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

print("\nLoading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")
print("Output shape:", model.output_shape)


# ------------------------------------------------------------
# LOAD IMAGE
# ------------------------------------------------------------

image = Image.open(
    IMAGE_PATH
).convert("RGB")

image = image.resize(
    (224, 224)
)

image_array = np.asarray(
    image,
    dtype=np.float32
)


# ------------------------------------------------------------
# TEST 1: RAW 0-255
# ------------------------------------------------------------

raw_input = np.expand_dims(
    image_array,
    axis=0
)

raw_prediction = model.predict(
    raw_input,
    verbose=0
)[0]


print("\n" + "=" * 60)
print("TEST 1: RAW IMAGE 0-255")
print("=" * 60)

print(
    "Non-Cancer:",
    f"{raw_prediction[0] * 100:.2f}%"
)

print(
    "Cancer:",
    f"{raw_prediction[1] * 100:.2f}%"
)


# ------------------------------------------------------------
# TEST 2: NORMALIZED 0-1
# ------------------------------------------------------------

normalized_input = image_array / 255.0

normalized_input = np.expand_dims(
    normalized_input,
    axis=0
)

normalized_prediction = model.predict(
    normalized_input,
    verbose=0
)[0]


print("\n" + "=" * 60)
print("TEST 2: NORMALIZED IMAGE 0-1")
print("=" * 60)

print(
    "Non-Cancer:",
    f"{normalized_prediction[0] * 100:.2f}%"
)

print(
    "Cancer:",
    f"{normalized_prediction[1] * 100:.2f}%"
)


# ------------------------------------------------------------
# MODEL LAYERS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("FIRST MODEL LAYERS")
print("=" * 60)

for layer in model.layers[:8]:
    print(
        layer.name,
        "->",
        layer.__class__.__name__
    )


# ------------------------------------------------------------
# FINAL
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)