import tensorflow as tf
import numpy as np
import pandas as pd
import glob
from PIL import Image

# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "models/efficientnetb0_binary_cancer.keras"

METADATA_PATH = (
    "data/raw/HAM10000_images/HAM10000_metadata.csv"
)

# Confirmed melanoma images from your HAM10000 dataset
MELANOMA_IDS = [
    "ISIC_0025964",
    "ISIC_0030623",
    "ISIC_0027190",
    "ISIC_0031023",
    "ISIC_0028086",
    "ISIC_0031177",
    "ISIC_0026993",
    "ISIC_0026120",
    "ISIC_0028412",
    "ISIC_0030417",
]

THRESHOLDS = [0.50, 0.45, 0.40]


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading binary model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.\n")


# ============================================================
# LOAD METADATA
# ============================================================

df = pd.read_csv(METADATA_PATH)


# ============================================================
# TEST EACH MELANOMA IMAGE
# ============================================================

results = []

for image_id in MELANOMA_IDS:

    print("-" * 60)
    print("Testing:", image_id)

    # Find image
    matches = glob.glob(
        f"data/raw/HAM10000_images/**/{image_id}.jpg",
        recursive=True
    )

    if not matches:
        print("IMAGE NOT FOUND")
        continue

    image_path = matches[0]

    # Load image
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))

    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)

    # Predict
    prediction = model.predict(
        image_array,
        verbose=0
    )[0]

    non_cancer_probability = float(prediction[0])
    cancer_probability = float(prediction[1])

    print(
        f"Non-Cancer: {non_cancer_probability * 100:.2f}%"
    )

    print(
        f"Cancer:     {cancer_probability * 100:.2f}%"
    )

    result = {
        "image_id": image_id,
        "cancer_probability": cancer_probability
    }

    results.append(result)


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

print("\n")
print("=" * 70)
print("CANCER THRESHOLD ANALYSIS")
print("=" * 70)

for threshold in THRESHOLDS:

    correct = 0

    print(
        f"\nThreshold: {threshold * 100:.0f}%"
    )

    print("-" * 70)

    for result in results:

        cancer_probability = result[
            "cancer_probability"
        ]

        if cancer_probability >= threshold:

            prediction = "CANCER"

            # All these images are confirmed melanoma
            correct += 1

        else:

            prediction = "NOT CANCER"

        print(
            f"{result['image_id']} | "
            f"Cancer: {cancer_probability * 100:6.2f}% | "
            f"{prediction}"
        )

    total = len(results)

    recall = (
        correct / total * 100
        if total > 0
        else 0
    )

    print("-" * 70)

    print(
        f"Detected {correct}/{total} confirmed melanoma images"
    )

    print(
        f"Cancer detection rate: {recall:.2f}%"
    )


# ============================================================
# FINISHED
# ============================================================

print("\n")
print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)