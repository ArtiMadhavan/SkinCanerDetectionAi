import os
import sys
import numpy as np
import tensorflow as tf

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORTS
# ============================================================

from src.dataset import load_metadata
from src.preprocessing import add_binary_label, split_data


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "efficientnetb0_binary_cancer.keras"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 60)
print("TESTING BINARY MODEL")
print("=" * 60)

print("\nLoading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")
print("Output shape:", model.output_shape)


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading HAM10000 metadata...")

df = load_metadata()

df = add_binary_label(df)

train_df, val_df, test_df = split_data(df)


# ============================================================
# SELECT TEST IMAGE
# ============================================================

# Take the first cancer image from the REAL test set
cancer_test = test_df[
    test_df["binary_label"] == "cancer"
]

# Take the first non-cancer image from the REAL test set
non_cancer_test = test_df[
    test_df["binary_label"] == "non_cancer"
]


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def test_image(row, expected):

    image_path = row["image_path"]

    print("\n" + "-" * 60)

    print("Image:")
    print(image_path)

    print("Expected:")
    print(expected)

    if not os.path.exists(image_path):
        print("ERROR: Image file not found.")
        return

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = tf.keras.utils.load_img(
        image_path,
        target_size=(224, 224)
    )

    image_array = tf.keras.utils.img_to_array(
        image
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(
        image_array,
        verbose=0
    )[0]

    non_cancer = float(prediction[0])
    cancer = float(prediction[1])

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    if cancer >= non_cancer:
        predicted = "cancer"
        confidence = cancer * 100
    else:
        predicted = "non_cancer"
        confidence = non_cancer * 100

    print("\nModel probabilities:")

    print(
        f"Non-Cancer: {non_cancer * 100:.2f}%"
    )

    print(
        f"Cancer:     {cancer * 100:.2f}%"
    )

    print("\nPredicted:")
    print(predicted)

    print(
        f"Confidence: {confidence:.2f}%"
    )

    print("\nExpected:")
    print(expected)

    if predicted == expected:
        print("RESULT: CORRECT")
    else:
        print("RESULT: WRONG")


# ============================================================
# TEST CANCER IMAGE
# ============================================================

print("\n" + "=" * 60)
print("TEST 1: CANCER IMAGE")
print("=" * 60)

if len(cancer_test) > 0:

    test_image(
        cancer_test.iloc[0],
        "cancer"
    )

else:

    print("No cancer image found.")


# ============================================================
# TEST NON-CANCER IMAGE
# ============================================================

print("\n" + "=" * 60)
print("TEST 2: NON-CANCER IMAGE")
print("=" * 60)

if len(non_cancer_test) > 0:

    test_image(
        non_cancer_test.iloc[0],
        "non_cancer"
    )

else:

    print("No non-cancer image found.")


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)