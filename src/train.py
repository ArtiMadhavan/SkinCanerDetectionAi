import os
import sys
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.dataset import load_metadata, get_class_weights
from src.preprocessing import split_data, create_generators
from src.model import build_model

from utils.config import (
    MODELS_DIR,
    EPOCHS,
    LEARNING_RATE,
    RESULTS_DIR,
    RANDOM_SEED,
    MODEL_SAVE_PATH,
    CLASS_MAP_PATH,
    CLASS_NAMES_MAP,
    CLASSES
)


# ==========================================================
# REPRODUCIBILITY
# ==========================================================

def set_seeds():

    os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)

    np.random.seed(RANDOM_SEED)

    tf.random.set_seed(RANDOM_SEED)

    tf.keras.utils.set_random_seed(RANDOM_SEED)

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


# ==========================================================
# SAVE CLASS MAPPING
# ==========================================================

def save_class_mapping():

    os.makedirs(MODELS_DIR, exist_ok=True)

    with open(CLASS_MAP_PATH, "w") as f:
        json.dump(CLASS_NAMES_MAP, f, indent=4)

    print(f"Class names saved to {CLASS_MAP_PATH}")


# ==========================================================
# PLOT TRAINING HISTORY
# ==========================================================

def plot_history(history, save_path):

    os.makedirs(save_path, exist_ok=True)

    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["accuracy"],
        label="Train Accuracy"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.title("Model Accuracy")

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            save_path,
            "training_accuracy.png"
        )
    )

    plt.close()


    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["loss"],
        label="Train Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.title("Model Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            save_path,
            "training_loss.png"
        )
    )

    plt.close()


# ==========================================================
# CONFUSION MATRIX
# ==========================================================

def evaluate_detailed(model, test_gen):

    print("\n========================================")
    print("DETAILED TEST EVALUATION")
    print("========================================")

    # Important: don't shuffle test generator
    test_gen.reset()

    predictions = model.predict(
        test_gen,
        verbose=1
    )

    predicted_indices = np.argmax(
        predictions,
        axis=1
    )

    true_indices = test_gen.classes

    class_indices = list(range(len(CLASSES)))

    print("\nCLASSIFICATION REPORT")
    print("----------------------------------------")

    print(
        classification_report(
            true_indices,
            predicted_indices,
            labels=class_indices,
            target_names=[
                CLASS_NAMES_MAP[c]
                for c in CLASSES
            ],
            digits=4,
            zero_division=0
        )
    )

    # ------------------------------------------------------
    # Confusion Matrix
    # ------------------------------------------------------

    cm = confusion_matrix(
        true_indices,
        predicted_indices,
        labels=class_indices
    )

    print("\nCONFUSION MATRIX")
    print("----------------------------------------")

    print(cm)

    results_dir = os.path.join(
        RESULTS_DIR,
        "figures"
    )

    os.makedirs(
        results_dir,
        exist_ok=True
    )

    plt.figure(figsize=(10, 8))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            CLASS_NAMES_MAP[c]
            for c in CLASSES
        ]
    )

    disp.plot(
        xticks_rotation=45
    )

    plt.title(
        "Skin Cancer Classification Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            results_dir,
            "confusion_matrix.png"
        )
    )

    plt.close()

    # ------------------------------------------------------
    # Melanoma vs Nevus
    # ------------------------------------------------------

    mel_index = CLASSES.index("mel")
    nv_index = CLASSES.index("nv")

    melanoma_total = cm[mel_index].sum()

    melanoma_correct = cm[
        mel_index,
        mel_index
    ]

    nevus_total = cm[nv_index].sum()

    nevus_correct = cm[
        nv_index,
        nv_index
    ]

    print("\nMELANOMA vs NEVUS")
    print("----------------------------------------")

    if melanoma_total > 0:

        print(
            f"Melanoma accuracy: "
            f"{melanoma_correct / melanoma_total * 100:.2f}%"
        )

    if nevus_total > 0:

        print(
            f"Nevus accuracy: "
            f"{nevus_correct / nevus_total * 100:.2f}%"
        )

    print(
        "\nMelanoma incorrectly predicted as Nevus:",
        cm[mel_index, nv_index]
    )

    print(
        "Nevus incorrectly predicted as Melanoma:",
        cm[nv_index, mel_index]
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    set_seeds()

    # ======================================================
    # DATA
    # ======================================================

    print("\n========================================")
    print("LOADING DATA")
    print("========================================")

    print("Loading metadata...")

    df = load_metadata()

    print("Splitting data securely by lesion_id...")

    train_df, val_df, test_df = split_data(df)

    print("Creating data generators...")

    train_gen, val_gen, test_gen = create_generators(
        train_df,
        val_df,
        test_df
    )

    print(
        "Calculating class weights from training set ONLY..."
    )

    class_weights = get_class_weights(
        train_df
    )

    print("\nClass weights:")

    for index, weight in class_weights.items():

        print(
            f"{index} - "
            f"{CLASSES[index]}: "
            f"{weight:.4f}"
        )

    print("\nSaving class mapping...")

    save_class_mapping()

    # ======================================================
    # STAGE 1
    # ======================================================

    print("\n========================================")
    print("STAGE 1: TRAINING CLASSIFICATION HEAD")
    print("========================================")

    model = build_model()

    print(
        f"Model output shape: "
        f"{model.output_shape}"
    )

    assert model.output_shape == (
        None,
        len(CLASSES)
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    stage1_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_SAVE_PATH),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )

    stage1_early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=6,
        restore_best_weights=True,
        verbose=1
    )

    stage1_reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )

    print(
        f"\nStarting Stage 1 for "
        f"{EPOCHS} epochs..."
    )

    history_stage1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=[
            stage1_checkpoint,
            stage1_early_stop,
            stage1_reduce_lr
        ],
        verbose=1
    )

    # ======================================================
    # LOAD BEST STAGE 1 MODEL
    # ======================================================

    print("\n========================================")
    print("LOADING BEST STAGE 1 MODEL")
    print("========================================")

    model = tf.keras.models.load_model(
        str(MODEL_SAVE_PATH)
    )

    # ======================================================
    # STAGE 2
    # ======================================================

    print("\n========================================")
    print("STAGE 2: FINE-TUNING EFFICIENTNETB0")
    print("========================================")

    total_layers = len(model.layers)

    print(
        f"Total model layers: {total_layers}"
    )

    # Freeze everything
    for layer in model.layers:
        layer.trainable = False

    # ------------------------------------------------------
    # Unfreeze last 50%
    # ------------------------------------------------------

    fine_tune_from = int(
        total_layers * 0.50
    )

    print(
        f"Fine-tuning from layer "
        f"{fine_tune_from} "
        f"of {total_layers}"
    )

    for layer in model.layers[fine_tune_from:]:

        if isinstance(
            layer,
            tf.keras.layers.BatchNormalization
        ):

            layer.trainable = False

        else:

            layer.trainable = True

    trainable_count = sum(
        layer.trainable
        for layer in model.layers
    )

    print(
        f"Trainable model layers: "
        f"{trainable_count}"
    )

    # ======================================================
    # COMPILE FINE-TUNING MODEL
    # ======================================================

    fine_tune_lr = 5e-6

    print(
        f"Fine-tuning learning rate: "
        f"{fine_tune_lr}"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=fine_tune_lr
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # ======================================================
    # CALLBACKS
    # ======================================================

    fine_tune_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_SAVE_PATH),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )

    fine_tune_early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=6,
        restore_best_weights=True,
        verbose=1
    )

    fine_tune_reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )

    # ======================================================
    # FINE-TUNING
    # ======================================================

    fine_tune_epochs = 20

    print(
        f"\nStarting Stage 2 for "
        f"{fine_tune_epochs} epochs..."
    )

    history_stage2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=fine_tune_epochs,
        class_weight=class_weights,
        callbacks=[
            fine_tune_checkpoint,
            fine_tune_early_stop,
            fine_tune_reduce_lr
        ],
        verbose=1
    )

    # ======================================================
    # LOAD BEST FINAL MODEL
    # ======================================================

    print("\n========================================")
    print("LOADING BEST FINAL MODEL")
    print("========================================")

    model = tf.keras.models.load_model(
        str(MODEL_SAVE_PATH)
    )

    # ======================================================
    # FINAL VALIDATION
    # ======================================================

    print("\n========================================")
    print("FINAL VALIDATION RESULTS")
    print("========================================")

    val_loss, val_accuracy = model.evaluate(
        val_gen,
        verbose=1
    )

    print(
        f"\nValidation Loss: "
        f"{val_loss:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{val_accuracy * 100:.2f}%"
    )

    # ======================================================
    # FINAL TEST
    # ======================================================

    print("\n========================================")
    print("FINAL TEST RESULTS")
    print("========================================")

    test_loss, test_accuracy = model.evaluate(
        test_gen,
        verbose=1
    )

    print(
        f"\nTest Loss: "
        f"{test_loss:.4f}"
    )

    print(
        f"Test Accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )

    # ======================================================
    # DETAILED EVALUATION
    # ======================================================

    evaluate_detailed(
        model,
        test_gen
    )

    # ======================================================
    # SAVE HISTORY
    # ======================================================

    print(
        "\nSaving training history..."
    )

    plot_history(
        history_stage2,
        os.path.join(
            RESULTS_DIR,
            "figures"
        )
    )

    # ======================================================
    # COMPLETED
    # ======================================================

    print("\n========================================")
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("========================================")

    print(
        "\nBest model saved to:"
    )

    print(
        MODEL_SAVE_PATH
    )

    print(
        f"\nFinal validation accuracy: "
        f"{val_accuracy * 100:.2f}%"
    )

    print(
        f"Final test accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )

    print(
        "\nYou can now use this model in Streamlit."
    )


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":
    main()