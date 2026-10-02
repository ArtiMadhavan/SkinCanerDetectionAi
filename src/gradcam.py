import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image


# --------------------------------------------------
# Project path
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------
# Grad-CAM function
# --------------------------------------------------

def generate_gradcam(
    model,
    image_path,
    image_id,
    true_class,
    last_conv_layer,
    output_path,
    IMG_SIZE,
    CLASSES
):

    # Load image
    img = image.load_img(
        image_path,
        target_size=IMG_SIZE
    )

    img_array = image.img_to_array(img)

    # EfficientNet expects pixel values in [0, 255]
    input_tensor = np.expand_dims(
        img_array,
        axis=0
    )

    # Grad-CAM model
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            last_conv_layer.output,
            model.output
        ]
    )

    # Calculate gradients
    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            input_tensor
        )

        predicted_class = tf.argmax(
            predictions[0]
        )

        class_score = predictions[
            0,
            predicted_class
        ]

    gradients = tape.gradient(
        class_score,
        conv_outputs
    )

    # Remove batch dimension
    conv_outputs = conv_outputs[0]
    gradients = gradients[0]

    # Calculate importance weights
    weights = tf.reduce_mean(
        gradients,
        axis=(0, 1)
    )

    # Generate heatmap
    heatmap = tf.reduce_sum(
        conv_outputs * weights,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    max_value = tf.reduce_max(
        heatmap
    )

    heatmap = heatmap / (
        max_value +
        tf.keras.backend.epsilon()
    )

    heatmap = heatmap.numpy()

    # Prediction information
    predicted_class_index = int(
        predicted_class.numpy()
    )

    predicted_class_name = CLASSES[
        predicted_class_index
    ]

    confidence = float(
        predictions[
            0,
            predicted_class_index
        ]
    )

    # Load original image
    original = np.array(
        image.load_img(image_path)
    )

    # Plot
    plt.figure(
        figsize=(12, 5)
    )

    # Original
    plt.subplot(
        1,
        2,
        1
    )

    plt.imshow(original)

    plt.title(
        f"Original\n"
        f"True: {true_class}"
    )

    plt.axis("off")

    # Grad-CAM
    plt.subplot(
        1,
        2,
        2
    )

    plt.imshow(original)

    plt.imshow(
        heatmap,
        cmap="jet",
        alpha=0.45
    )

    plt.title(
        f"Grad-CAM\n"
        f"Predicted: {predicted_class_name}\n"
        f"Confidence: {confidence:.1%}"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"✓ {true_class}: "
        f"{image_id} → "
        f"{predicted_class_name} "
        f"({confidence:.2%})"
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    from src.dataset import load_metadata
    from src.preprocessing import split_data
    from utils.config import (
        MODELS_DIR,
        RESULTS_DIR,
        IMG_SIZE,
        CLASSES
    )

    print("\n========================================")
    print("GRAD-CAM - ALL 7 CLASSES")
    print("========================================")

    # --------------------------------------------------
    # 1. Load metadata
    # --------------------------------------------------

    print("\nLoading dataset...")

    df = load_metadata()

    _, _, test_df = split_data(df)

    print(
        f"Test images: {len(test_df)}"
    )

    # --------------------------------------------------
    # 2. Load model
    # --------------------------------------------------

    model_path = os.path.join(
    MODELS_DIR,
    "efficientnetb0_best.keras"
    )
    print("\nLoading model...")

    model = tf.keras.models.load_model(
        model_path
    )

    print(
        "Model loaded successfully."
    )

    # --------------------------------------------------
    # 3. Find last Conv2D layer
    # --------------------------------------------------

    conv_layers = [
        layer
        for layer in model.layers
        if isinstance(
            layer,
            tf.keras.layers.Conv2D
        )
    ]

    if not conv_layers:

        raise ValueError(
            "No Conv2D layer found."
        )

    last_conv_layer = conv_layers[-1]

    print(
        f"Grad-CAM layer: "
        f"{last_conv_layer.name}"
    )

    # --------------------------------------------------
    # 4. Output directory
    # --------------------------------------------------

    output_dir = os.path.join(
        RESULTS_DIR,
        "gradcam"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # --------------------------------------------------
    # 5. Image root
    # --------------------------------------------------

    image_root = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "HAM10000_images"
    )

    # --------------------------------------------------
    # 6. Generate one image per class
    # --------------------------------------------------

    print("\nGenerating Grad-CAM examples...\n")

    for class_name in CLASSES:

        # Find first test image of this class
        class_df = test_df[
            test_df["dx"] == class_name
        ]

        if len(class_df) == 0:

            print(
                f"WARNING: No test image "
                f"found for {class_name}"
            )

            continue

        sample = class_df.iloc[0]

        image_id = sample["image_id"]

        true_class = sample["dx"]

        # Find image
        image_path = None

        for folder in [
            image_root / "HAM10000_images_part_1",
            image_root / "HAM10000_images_part_2"
        ]:

            candidate = (
                folder /
                f"{image_id}.jpg"
            )

            if candidate.exists():

                image_path = candidate

                break

        if image_path is None:

            print(
                f"WARNING: Image not found "
                f"for {image_id}"
            )

            continue

        # Output filename
        output_path = os.path.join(
            output_dir,
            f"{class_name}_gradcam.png"
        )

        # Generate Grad-CAM
        generate_gradcam(
            model=model,
            image_path=image_path,
            image_id=image_id,
            true_class=true_class,
            last_conv_layer=last_conv_layer,
            output_path=output_path,
            IMG_SIZE=IMG_SIZE,
            CLASSES=CLASSES
        )

    # --------------------------------------------------
    # 7. Finished
    # --------------------------------------------------

    print("\n========================================")
    print("ALL GRAD-CAM EXAMPLES COMPLETED")
    print("========================================")

    print(
        f"\nSaved in:\n{output_dir}"
    )


# --------------------------------------------------
# Run
# --------------------------------------------------

if __name__ == "__main__":
    main()