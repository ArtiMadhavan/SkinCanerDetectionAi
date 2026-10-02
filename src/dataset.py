import os
import pandas as pd
from utils.config import (
    HAM_METADATA_CSV,
    HAM_IMAGES_PART_1,
    HAM_IMAGES_PART_2,
    CLASSES
)


def get_image_path(image_id):
    """
    Locate the image file in either HAM10000 image directory.
    """

    path1 = os.path.join(
        HAM_IMAGES_PART_1,
        f"{image_id}.jpg"
    )

    path2 = os.path.join(
        HAM_IMAGES_PART_2,
        f"{image_id}.jpg"
    )

    if os.path.exists(path1):
        return path1

    if os.path.exists(path2):
        return path2

    return None


def load_metadata():
    """
    Load HAM10000 metadata and locate corresponding images.
    """

    if not os.path.exists(HAM_METADATA_CSV):
        raise FileNotFoundError(
            f"Metadata file not found at {HAM_METADATA_CSV}"
        )

    df = pd.read_csv(HAM_METADATA_CSV)

    # --------------------------------------------------
    # Locate image paths
    # --------------------------------------------------

    df["image_path"] = df["image_id"].apply(get_image_path)

    # Remove missing images
    missing_mask = df["image_path"].isnull()
    missing_count = missing_mask.sum()

    if missing_count > 0:
        print(
            f"Warning: {missing_count} images not found. "
            f"Removing them."
        )

        df = df[~missing_mask].copy()

    # Remove records without diagnosis
    df = df.dropna(subset=["dx"]).copy()

    # --------------------------------------------------
    # Keep only classes used by the model
    # --------------------------------------------------

    df = df[df["dx"].isin(CLASSES)].copy()

    # --------------------------------------------------
    # IMPORTANT:
    # Explicitly map diagnosis → model index
    #
    # This MUST match CLASSES exactly.
    # --------------------------------------------------

    class_to_index = {
        class_name: index
        for index, class_name in enumerate(CLASSES)
    }

    df["label"] = df["dx"].map(class_to_index)

    # Safety check
    if df["label"].isnull().any():
        raise ValueError(
            "Some labels could not be mapped to CLASSES."
        )

    df["label"] = df["label"].astype(int)

    # --------------------------------------------------
    # Print class distribution
    # --------------------------------------------------

    print("\nClass distribution:")

    for class_name in CLASSES:
        count = (df["dx"] == class_name).sum()
        print(f"  {class_name}: {count}")

    print("\nClass → index mapping:")

    for class_name, index in class_to_index.items():
        print(f"  {index} -> {class_name}")

    return df


def get_class_weights(train_df):
    """
    Calculate balanced class weights using the SAME
    class indices used by the Keras model.
    """

    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np

    # Explicit model class indices
    classes = np.arange(len(CLASSES))

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_df["label"].values
    )

    class_weights = {
        int(class_index): float(weight)
        for class_index, weight in zip(classes, weights)
    }

    print("\nClass weights:")

    for index, weight in class_weights.items():
        print(
            f"  {index} ({CLASSES[index]}): "
            f"{weight:.4f}"
        )

    return class_weights