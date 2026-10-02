import os
import numpy as np
import tensorflow as tf
from PIL import Image


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "efficientnetb0_best.keras"
)

IMAGE_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "HAM10000_images",
    "HAM10000_images_part_2",
    "ISIC_0029417.jpg"
)


# ============================================================
# CLASS MAPPING
# ============================================================

CLASS_CODES = [
    "mel",
    "bcc",
    "akiec",
    "nv",
    "bkl",
    "df",
    "vasc"
]

CLASS_NAMES = {
    "mel": "Melanoma",
    "bcc": "Basal Cell Carcinoma",
    "akiec": "Actinic Keratoses",
    "nv": "Melanocytic Nevus",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "vasc": "Vascular Lesion"
}


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("LOADING 7-CLASS MODEL")
print("=" * 60)

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# CHECK IMAGE
# ============================================================

if not os.path.exists(IMAGE_PATH):

    print("\nERROR: Image not found:")
    print(IMAGE_PATH)
    raise SystemExit


print("\nImage:")
print(IMAGE_PATH)


# ============================================================
# LOAD IMAGE
# ============================================================

image = Image.open(
    IMAGE_PATH
).convert("RGB")

image = image.resize(
    (224, 224)
)

img_array = np.asarray(
    image,
    dtype=np.float32
)

img_array = np.expand_dims(
    img_array,
    axis=0
)


# ============================================================
# PREDICTION
# ============================================================

predictions = model.predict(
    img_array,
    verbose=0
)[0]


# ============================================================
# DISPLAY ALL PREDICTIONS
# ============================================================

print("\n" + "=" * 60)
print("7-CLASS PREDICTIONS")
print("=" * 60)

for i in range(len(CLASS_CODES)):

    code = CLASS_CODES[i]

    name = CLASS_NAMES[code]

    confidence = predictions[i] * 100

    print(
        f"{name:<30} {confidence:>7.2f}%"
    )


# ============================================================
# TOP PREDICTION
# ============================================================

predicted_index = int(
    np.argmax(predictions)
)

predicted_code = CLASS_CODES[
    predicted_index
]

predicted_name = CLASS_NAMES[
    predicted_code
]

confidence = (
    predictions[predicted_index] * 100
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("FINAL PREDICTION")
print("=" * 60)

print(
    f"Predicted Class: {predicted_name}"
)

print(
    f"Class Code: {predicted_code}"
)

print(
    f"Confidence: {confidence:.2f}%"
)


# ============================================================
# CANCER CATEGORY
# ============================================================

if predicted_code in ["mel", "bcc"]:

    category = "CANCER"

elif predicted_code == "akiec":

    category = "NEEDS MEDICAL CHECK"

else:

    category = "NOT CANCER"


print(
    f"Category: {category}"
)

print("=" * 60)