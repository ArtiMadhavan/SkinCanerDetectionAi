import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

METADATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "HAM10000_images",
    "HAM10000_metadata.csv"
)

IMAGE_DIR_1 = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "HAM10000_images",
    "HAM10000_images_part_1"
)

IMAGE_DIR_2 = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "HAM10000_images",
    "HAM10000_images_part_2"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "efficientnetb0_three_class.keras"
)

CLASS_NAMES_PATH = os.path.join(
    MODEL_DIR,
    "three_class_names.json"
)

# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

STAGE1_EPOCHS = 10
STAGE2_EPOCHS = 15

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ============================================================
# THREE CLASSES
# ============================================================

CLASS_NAMES = [
    "non_cancer",
    "cancer",
    "needs_check"
]

# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

print("=" * 70)
print("THREE-CLASS SKIN CANCER MODEL TRAINING")
print("=" * 70)

# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading metadata...")

if not os.path.exists(METADATA_PATH):

    print(
        "\nERROR: Metadata file not found:"
    )

    print(
        METADATA_PATH
    )

    raise SystemExit

df = pd.read_csv(
    METADATA_PATH
)

print(
    f"Total metadata rows: {len(df)}"
)

# ============================================================
# FIND IMAGE
# ============================================================

def find_image_path(image_id):

    path1 = os.path.join(
        IMAGE_DIR_1,
        image_id + ".jpg"
    )

    path2 = os.path.join(
        IMAGE_DIR_2,
        image_id + ".jpg"
    )

    if os.path.exists(path1):
        return path1

    if os.path.exists(path2):
        return path2

    return None


print("\nFinding image files...")

df["image_path"] = df[
    "image_id"
].apply(
    find_image_path
)

df = df.dropna(
    subset=["image_path"]
).copy()

print(
    f"Images found: {len(df)}"
)

# ============================================================
# CREATE LABEL
# ============================================================

def create_three_class_label(dx):

    # Cancer
    if dx in ["mel", "bcc"]:
        return "cancer"

    # Needs medical check
    if dx == "akiec":
        return "needs_check"

    # Non-cancer
    if dx in [
        "nv",
        "bkl",
        "df",
        "vasc"
    ]:
        return "non_cancer"

    return None


df["three_class_label"] = df[
    "dx"
].apply(
    create_three_class_label
)

df = df.dropna(
    subset=["three_class_label"]
).copy()

# ============================================================
# DISTRIBUTION
# ============================================================

print("\nThree-class distribution:")

for class_name in CLASS_NAMES:

    count = (
        df["three_class_label"]
        == class_name
    ).sum()

    print(
        f"{class_name:<15}: {count}"
    )

# ============================================================
# GROUPED SPLIT
# ============================================================

print(
    "\nCreating train / validation / test split..."
)

gss = GroupShuffleSplit(
    n_splits=1,
    train_size=0.70,
    random_state=RANDOM_SEED
)

train_idx, remaining_idx = next(
    gss.split(
        df,
        groups=df["lesion_id"]
    )
)

train_df = df.iloc[
    train_idx
].copy()

remaining_df = df.iloc[
    remaining_idx
].copy()

gss_second = GroupShuffleSplit(
    n_splits=1,
    train_size=0.50,
    random_state=RANDOM_SEED
)

val_idx, test_idx = next(
    gss_second.split(
        remaining_df,
        groups=remaining_df["lesion_id"]
    )
)

val_df = remaining_df.iloc[
    val_idx
].copy()

test_df = remaining_df.iloc[
    test_idx
].copy()

print("\nDataset split:")

print(
    f"Train:      {len(train_df)}"
)

print(
    f"Validation: {len(val_df)}"
)

print(
    f"Test:       {len(test_df)}"
)

# ============================================================
# GENERATORS
# ============================================================

print(
    "\nCreating image generators..."
)

train_datagen = (
    tf.keras.preprocessing
    .image.ImageDataGenerator(
        rotation_range=25,
        width_shift_range=0.10,
        height_shift_range=0.10,
        zoom_range=0.10,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="nearest"
    )
)

val_test_datagen = (
    tf.keras.preprocessing
    .image.ImageDataGenerator()
)

train_gen = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="image_path",
    y_col="three_class_label",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    classes=CLASS_NAMES,
    class_mode="categorical",
    shuffle=True,
    seed=RANDOM_SEED
)

val_gen = val_test_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col="image_path",
    y_col="three_class_label",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    classes=CLASS_NAMES,
    class_mode="categorical",
    shuffle=False
)

test_gen = val_test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col="image_path",
    y_col="three_class_label",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    classes=CLASS_NAMES,
    class_mode="categorical",
    shuffle=False
)

print(
    "\nGenerator class mapping:"
)

print(
    train_gen.class_indices
)

# ============================================================
# CLASS WEIGHTS
# ============================================================

print(
    "\nCalculating class weights..."
)

class_indices = {
    "non_cancer": 0,
    "cancer": 1,
    "needs_check": 2
}

y_train = (
    train_df[
        "three_class_label"
    ]
    .map(class_indices)
    .values
)

classes = np.array(
    [0, 1, 2]
)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = {
    int(index): float(weight)
    for index, weight in zip(
        classes,
        weights
    )
}

print(
    f"non_cancer  : "
    f"{class_weights[0]:.4f}"
)

print(
    f"cancer      : "
    f"{class_weights[1]:.4f}"
)

print(
    f"needs_check : "
    f"{class_weights[2]:.4f}"
)

# ============================================================
# BUILD OR LOAD MODEL
# ============================================================

if os.path.exists(MODEL_PATH):

    print("\n" + "=" * 70)
    print("EXISTING MODEL FOUND")
    print("=" * 70)

    print(
        "\nLoading saved Stage 1 model..."
    )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

else:

    print("\n" + "=" * 70)
    print("BUILDING NEW EFFICIENTNETB0")
    print("=" * 70)

    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(
            224,
            224,
            3
        )
    )

    base_model.trainable = False

    x = base_model.output

    x = GlobalAveragePooling2D(
        name="global_average_pooling"
    )(x)

    x = Dropout(
        0.30,
        name="dropout_layer"
    )(x)

    outputs = Dense(
        3,
        activation="softmax",
        name="prediction_layer"
    )(x)

    model = Model(
        inputs=base_model.input,
        outputs=outputs
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-4
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print(
        "\nStarting Stage 1..."
    )

    stage1_callbacks = [

        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),

        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-7,
            verbose=1
        )
    ]

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=STAGE1_EPOCHS,
        class_weight=class_weights,
        callbacks=stage1_callbacks,
        verbose=1
    )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

# ============================================================
# FINE-TUNING
# ============================================================

print("\n" + "=" * 70)
print("STAGE 2: FINE-TUNING")
print("=" * 70)

print(
    "\nThe EfficientNet layers are directly inside the model."
)

print(
    f"Total model layers: {len(model.layers)}"
)

# ------------------------------------------------------------
# Freeze everything first
# ------------------------------------------------------------

for layer in model.layers:

    layer.trainable = False

# ------------------------------------------------------------
# Unfreeze last 30 layers
# ------------------------------------------------------------

fine_tune_from = max(
    0,
    len(model.layers) - 30
)

for layer in model.layers[
    fine_tune_from:
]:

    layer.trainable = True

# ------------------------------------------------------------
# Keep BatchNormalization frozen
# ------------------------------------------------------------

for layer in model.layers:

    if isinstance(
        layer,
        tf.keras.layers.BatchNormalization
    ):

        layer.trainable = False

trainable_count = sum(
    1
    for layer in model.layers
    if layer.trainable
)

print(
    f"\nTrainable layers: "
    f"{trainable_count}"
)

print(
    f"Fine-tuning from layer index: "
    f"{fine_tune_from}"
)

# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=5e-6
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ============================================================
# CALLBACKS
# ============================================================

fine_tune_callbacks = [

    EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),

    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-8,
        verbose=1
    )
]

# ============================================================
# TRAIN STAGE 2
# ============================================================

print(
    "\nStarting Stage 2 fine-tuning..."
)

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=STAGE2_EPOCHS,
    class_weight=class_weights,
    callbacks=fine_tune_callbacks,
    verbose=1
)

# ============================================================
# LOAD BEST MODEL
# ============================================================

print(
    "\nLoading best final model..."
)

model = tf.keras.models.load_model(
    MODEL_PATH
)

# ============================================================
# TEST
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

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
# PREDICTIONS
# ============================================================

print(
    "\nGenerating predictions..."
)

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

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        true_classes,
        predicted_classes,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    true_classes,
    predicted_classes
)

print(cm)

# ============================================================
# SAVE CLASS MAPPING
# ============================================================

class_mapping = {
    "0": "non_cancer",
    "1": "cancer",
    "2": "needs_check"
}

with open(
    CLASS_NAMES_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        class_mapping,
        f,
        indent=4
    )

# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    "\nModel:"
)

print(
    MODEL_PATH
)

print(
    "\nClass mapping:"
)

print(
    CLASS_NAMES_PATH
)

print(
    "\n0 -> NOT A CANCER CELL"
)

print(
    "1 -> CANCER"
)

print(
    "2 -> NEEDS MEDICAL CHECK"
)

print("\n" + "=" * 70)