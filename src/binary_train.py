import os
import sys

# ============================================================
# PROJECT ROOT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from src.binary_model import build_binary_model
from src.dataset import load_metadata

from src.preprocessing import (
    add_binary_label,
    split_data,
    create_generators
)

from utils.config import (
    IMG_SIZE,
    DROPOUT_RATE
)


# ============================================================
# SETTINGS
# ============================================================

EPOCHS_STAGE1 = 10
EPOCHS_STAGE2 = 20

LEARNING_RATE_STAGE1 = 1e-4
LEARNING_RATE_STAGE2 = 5e-6

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "efficientnetb0_binary_cancer_best.keras"
)

FINAL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "efficientnetb0_binary_cancer.keras"
)

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)

np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.keras.utils.set_random_seed(SEED)
except Exception:
    pass


# ============================================================
# START
# ============================================================

print("\n" + "=" * 60)
print("BINARY SKIN CANCER DETECTION")
print("=" * 60)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading HAM10000 metadata...")

df = load_metadata()

print(f"Original dataset size: {len(df)}")


# ============================================================
# CREATE BINARY LABELS
# ============================================================

print("\nCreating binary labels...")

df = add_binary_label(df)

print(f"Binary dataset size: {len(df)}")

print("\nBinary class distribution:")
print(df["binary_label"].value_counts())


# ============================================================
# SHOW ORIGINAL CLASSES
# ============================================================

print("\nOriginal classes used in binary training:")
print(df["dx"].value_counts())


# ============================================================
# SPLIT DATA
# ============================================================

print("\nSplitting dataset by lesion_id...")

train_df, val_df, test_df = split_data(df)

print("\nDataset split:")
print(f"Train:      {len(train_df)}")
print(f"Validation: {len(val_df)}")
print(f"Test:       {len(test_df)}")


# ============================================================
# CREATE GENERATORS
# ============================================================

print("\nCreating image generators...")

train_gen, val_gen, test_gen = create_generators(
    train_df,
    val_df,
    test_df
)

print("\nGenerator class mapping:")
print(train_gen.class_indices)


# ============================================================
# CLASS WEIGHTS
# ============================================================

print("\nCalculating class weights...")

classes = np.unique(train_gen.classes)

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train_gen.classes
)

class_weights = {
    int(class_id): float(weight)
    for class_id, weight in zip(
        classes,
        class_weights_array
    )
}

print("\nClass weights:")
print(class_weights)


# ============================================================
# BUILD BINARY MODEL
# ============================================================

print("\n" + "=" * 60)
print("BUILDING BINARY EFFICIENTNETB0")
print("=" * 60)

model = build_binary_model()

print("\nModel output shape:")
print(model.output_shape)

assert model.output_shape == (
    None,
    2
), "Binary model must have output shape (None, 2)"


# ============================================================
# STAGE 1
# FEATURE EXTRACTION
# ============================================================

print("\n" + "=" * 60)
print("STAGE 1 - FEATURE EXTRACTION")
print("=" * 60)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE_STAGE1
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

stage1_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    BEST_MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

stage1_early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=5,
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
    f"{EPOCHS_STAGE1} epochs..."
)

history_stage1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_STAGE1,
    class_weight=class_weights,
    callbacks=[
        stage1_checkpoint,
        stage1_early_stop,
        stage1_reduce_lr
    ],
    verbose=1
)


# ============================================================
# LOAD BEST STAGE 1 MODEL
# ============================================================

print("\n" + "=" * 60)
print("LOADING BEST STAGE 1 MODEL")
print("=" * 60)

model = tf.keras.models.load_model(
    BEST_MODEL_PATH
)

print(
    f"Loaded model output shape: "
    f"{model.output_shape}"
)


# ============================================================
# STAGE 2
# FINE-TUNING
# ============================================================

print("\n" + "=" * 60)
print("STAGE 2 - FINE-TUNING EFFICIENTNETB0")
print("=" * 60)

print(
    f"Total model layers: "
    f"{len(model.layers)}"
)


# Freeze everything first

for layer in model.layers:
    layer.trainable = False


# Unfreeze upper 40%

fine_tune_from = int(
    len(model.layers) * 0.60
)

print(
    f"Fine-tuning from layer "
    f"{fine_tune_from} "
    f"of {len(model.layers)}"
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
    1
    for layer in model.layers
    if layer.trainable
)

print(
    f"Trainable layers: "
    f"{trainable_count}"
)


# ============================================================
# COMPILE STAGE 2
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE_STAGE2
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# STAGE 2 CALLBACKS
# ============================================================

stage2_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    BEST_MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

stage2_early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=7,
    restore_best_weights=True,
    verbose=1
)

stage2_reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=3,
    min_lr=1e-7,
    verbose=1
)


# ============================================================
# TRAIN STAGE 2
# ============================================================

print(
    f"\nStarting Stage 2 for "
    f"{EPOCHS_STAGE2} epochs..."
)

history_stage2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_STAGE2,
    class_weight=class_weights,
    callbacks=[
        stage2_checkpoint,
        stage2_early_stop,
        stage2_reduce_lr
    ],
    verbose=1
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\n" + "=" * 60)
print("LOADING BEST MODEL")
print("=" * 60)

model = tf.keras.models.load_model(
    BEST_MODEL_PATH
)


# ============================================================
# VALIDATION
# ============================================================

print("\nEvaluating validation set...")

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


# ============================================================
# TEST
# ============================================================

print("\nEvaluating test set...")

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


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

test_gen.reset()

predictions = model.predict(
    test_gen,
    verbose=1
)

predicted_classes = np.argmax(
    predictions,
    axis=1
)

true_classes = test_gen.classes

class_names = [
    "non_cancer",
    "cancer"
]

print(
    classification_report(
        true_classes,
        predicted_classes,
        target_names=class_names,
        digits=4
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")

cm = confusion_matrix(
    true_classes,
    predicted_classes
)

print(cm)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

print("\n" + "=" * 60)
print("SAVING FINAL MODEL")
print("=" * 60)

model.save(
    FINAL_MODEL_PATH
)

print(
    f"\nFinal model saved to:\n"
    f"{FINAL_MODEL_PATH}"
)


# ============================================================
# FINAL INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print("\nBinary class mapping:")

print("0 -> non_cancer")
print("1 -> cancer")

print(
    f"\nValidation Accuracy: "
    f"{val_accuracy * 100:.2f}%"
)

print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    "\nModel ready for Streamlit application."
)