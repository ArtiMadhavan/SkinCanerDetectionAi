import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from utils.config import IMG_SIZE, CLASSES, CLASS_NAMES_MAP


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

# Put your AK test image here.
# Change this filename if your image has a different name.
IMAGE_PATH = PROJECT_ROOT / "actinic_keratosis.jpg"

MODEL_PATHS = [
    PROJECT_ROOT / "models" / "efficientnetb0_best.keras",
    PROJECT_ROOT / "models" / "efficientnetb0_balanced_best.keras",
    PROJECT_ROOT / "models" / "efficientnetb0_finetuned_best.keras",
    PROJECT_ROOT / "models" / "skin_cancer_model.keras",
]


# ---------------------------------------------------------
# Load image
# ---------------------------------------------------------

def load_test_image(image_path):
    if not image_path.exists():
        raise FileNotFoundError(
            f"\n❌ Test image not found:\n{image_path}\n\n"
            "Put the image in the project folder and make sure "
            "the filename matches IMAGE_PATH."
        )

    img = Image.open(image_path).convert("RGB")

    print(f"Image: {image_path.name}")
    print(f"Original size: {img.size}")

    img = img.resize(IMG_SIZE)

    img_array = np.array(
        img,
        dtype=np.float32
    )

    # IMPORTANT:
    # Do NOT divide by 255 here.
    # EfficientNetB0 performs its own input rescaling.
    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    return img_array


# ---------------------------------------------------------
# Predict with one model
# ---------------------------------------------------------

def test_model(model_path, img_array):

    print("\n" + "=" * 75)
    print(f"MODEL: {model_path.name}")
    print("=" * 75)

    if not model_path.exists():
        print("❌ Model file not found.")
        return None

    try:
        model = tf.keras.models.load_model(
            model_path,
            compile=False
        )

        print(f"Model loaded successfully.")
        print(f"Output shape: {model.output_shape}")

        predictions = model.predict(
            img_array,
            verbose=0
        )[0]

        predictions = np.asarray(
            predictions,
            dtype=np.float64
        )

        # Check model output
        if len(predictions) != len(CLASSES):
            print(
                f"❌ ERROR: Model has {len(predictions)} outputs "
                f"but config has {len(CLASSES)} classes."
            )
            return None

        # Make sure values are probabilities.
        # Normally EfficientNet classifier outputs softmax probabilities.
        if (
            np.min(predictions) < 0
            or np.max(predictions) > 1
            or not np.isclose(
                predictions.sum(),
                1.0,
                atol=0.05
            )
        ):
            exp_predictions = np.exp(
                predictions - np.max(predictions)
            )
            predictions = (
                exp_predictions /
                exp_predictions.sum()
            )

        # -------------------------------------------------
        # Top 3
        # -------------------------------------------------

        top_indices = np.argsort(
            predictions
        )[::-1][:3]

        print("\nTop 3 predictions:")
        print("-" * 75)

        for rank, index in enumerate(
            top_indices,
            start=1
        ):
            class_code = CLASSES[index]
            class_name = CLASS_NAMES_MAP[class_code]
            confidence = predictions[index] * 100

            print(
                f"{rank}. "
                f"{class_code:6s} | "
                f"{class_name:25s} | "
                f"{confidence:7.2f}%"
            )

        # -------------------------------------------------
        # Final prediction
        # -------------------------------------------------

        predicted_index = int(
            np.argmax(predictions)
        )

        predicted_code = CLASSES[
            predicted_index
        ]

        predicted_class = CLASS_NAMES_MAP[
            predicted_code
        ]

        confidence = (
            predictions[predicted_index] * 100
        )

        print("\nFINAL PREDICTION")
        print("-" * 75)
        print(f"Class code : {predicted_code}")
        print(f"Class name : {predicted_class}")
        print(f"Confidence : {confidence:.2f}%")

        return {
            "model": model_path.name,
            "class_code": predicted_code,
            "class_name": predicted_class,
            "confidence": confidence,
        }

    except Exception as e:
        print(f"\n❌ Error loading/testing model:")
        print(e)

        return None


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("\n")
    print("=" * 75)
    print("        SKIN CANCER MODEL COMPARISON TEST")
    print("=" * 75)

    print("\nConfigured classes:")

    for index, code in enumerate(CLASSES):
        print(
            f"{index} → {code} → "
            f"{CLASS_NAMES_MAP[code]}"
        )

    # Load image once
    try:
        img_array = load_test_image(
            IMAGE_PATH
        )
    except Exception as e:
        print(e)
        return

    results = []

    # Test every model
    for model_path in MODEL_PATHS:

        result = test_model(
            model_path,
            img_array
        )

        if result is not None:
            results.append(result)

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print("\n\n")
    print("=" * 75)
    print("                         SUMMARY")
    print("=" * 75)

    if not results:
        print("❌ No models were successfully tested.")
        return

    print(
        f"{'MODEL':35s} "
        f"{'PREDICTION':25s} "
        f"{'CONFIDENCE':>12s}"
    )

    print("-" * 75)

    for result in results:

        print(
            f"{result['model']:35s} "
            f"{result['class_name']:25s} "
            f"{result['confidence']:10.2f}%"
        )

    print("=" * 75)

    print("\nTest completed.")
    print(
        "⚠️ This comparison does NOT prove that a prediction "
        "is medically correct."
    )


if __name__ == "__main__":
    main()