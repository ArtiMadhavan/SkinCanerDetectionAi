import os
import sys
from pathlib import Path

import tensorflow as tf
import numpy as np
import random

# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================
# MAIN
# =========================================================

def main():

    print("\n========================================")
    print("FINE-TUNING EXISTING EFFICIENTNETB0")
    print("========================================")

    # -----------------------------------------------------
    # Import project modules
    # -----------------------------------------------------

    from src.dataset import (
        load_metadata,
        get_class_weights
    )

    from src.preprocessing import (
        split_data,
        create_generators
    )

    from utils.config import MODELS_DIR

    # -----------------------------------------------------
    # Paths
    # -----------------------------------------------------

    base_model_path = os.path.join(
        MODELS_DIR,
        "efficientnetb0_best.keras"
    )

    finetuned_model_path = os.path.join(
        MODELS_DIR,
        "efficientnetb0_finetuned_best.keras"
    )

    print("\nExisting model:")
    print(base_model_path)

    # -----------------------------------------------------
    # Check model
    # -----------------------------------------------------

    if not os.path.exists(base_model_path):

        raise FileNotFoundError(
            f"Model not found:\n{base_model_path}"
        )

    print("\nExisting model found.")

    # -----------------------------------------------------
    # Load metadata
    # -----------------------------------------------------

    print("\nLoading metadata...")

    df = load_metadata()

    # -----------------------------------------------------
    # Split dataset
    # -----------------------------------------------------

    print("Splitting data securely by lesion_id...")

    train_df, val_df, test_df = split_data(df)

    # -----------------------------------------------------
    # Generators
    # -----------------------------------------------------

    print("Creating data generators...")

    train_gen, val_gen, test_gen = create_generators(
        train_df,
        val_df,
        test_df
    )

    # -----------------------------------------------------
    # Class weights
    # -----------------------------------------------------

    print("Calculating class weights...")

    class_weights = get_class_weights(train_df)

    print("\nClass weights:")
    print(class_weights)

    # -----------------------------------------------------
    # Load existing trained model
    # -----------------------------------------------------

    print("\nLoading existing trained model...")

    model = tf.keras.models.load_model(
        base_model_path
    )

    print("Model loaded successfully.")

    print(
        "\nModel output shape:",
        model.output_shape
    )

    # =====================================================
    # FINE-TUNING
    # =====================================================

    print("\n========================================")
    print("CONFIGURING FINE-TUNING")
    print("========================================")

    # -----------------------------------------------------
    # Freeze everything first
    # -----------------------------------------------------

    for layer in model.layers:

        layer.trainable = False

    # -----------------------------------------------------
    # Fine-tune upper EfficientNet layers
    #
    # Layers 0-190 remain frozen.
    # Layers 191 onward are fine-tuned.
    #
    # This includes the upper EfficientNet blocks and
    # classification head.
    # -----------------------------------------------------

    fine_tune_from = 191

    print(
        f"Fine-tuning layers {fine_tune_from} onward."
    )

    for layer in model.layers[fine_tune_from:]:

        layer.trainable = True

    # -----------------------------------------------------
    # Keep BatchNormalization frozen
    # -----------------------------------------------------

    for layer in model.layers:

        if isinstance(
            layer,
            tf.keras.layers.BatchNormalization
        ):

            layer.trainable = False

    # -----------------------------------------------------
    # Count trainable parameters
    # -----------------------------------------------------

    trainable_params = np.sum([
        np.prod(v.shape)
        for v in model.trainable_weights
    ])

    print(
        "\nTrainable parameters:",
        trainable_params
    )

    # -----------------------------------------------------
    # Compile
    # -----------------------------------------------------

    print("\nCompiling model...")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-5
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # =====================================================
    # CALLBACKS
    # =====================================================

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=finetuned_model_path,
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )

    # =====================================================
    # START FINE-TUNING
    # =====================================================

    print("\n========================================")
    print("STARTING FINE-TUNING")
    print("========================================")

    print("\nLearning rate: 0.00001")
    print("Maximum epochs: 10")
    print("Test dataset is NOT used for training.")

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10,
        class_weight=class_weights,
        callbacks=[
            checkpoint,
            early_stopping,
            reduce_lr
        ],
        verbose=1
    )

    # =====================================================
    # FINISHED
    # =====================================================

    print("\n========================================")
    print("FINE-TUNING COMPLETED")
    print("========================================")

    print(
        "\nBest fine-tuned model:"
    )

    print(finetuned_model_path)


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()