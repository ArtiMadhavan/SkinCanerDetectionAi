import tensorflow as tf
from utils.config import IMG_SIZE, BATCH_SIZE, RANDOM_SEED, CLASSES


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def split_data(
    df,
    train_size=0.70,
    test_size=0.15,
    val_size=0.15
):
    """
    Split HAM10000 by lesion_id.

    This prevents images from the same lesion
    appearing in different datasets.
    """

    from sklearn.model_selection import GroupShuffleSplit

    # --------------------------------------------------------
    # TRAIN vs VALIDATION + TEST
    # --------------------------------------------------------

    gss = GroupShuffleSplit(
        n_splits=1,
        train_size=train_size,
        random_state=RANDOM_SEED
    )

    train_idx, val_test_idx = next(
        gss.split(
            df,
            groups=df["lesion_id"]
        )
    )

    train_df = df.iloc[train_idx].copy()

    val_test_df = df.iloc[val_test_idx].copy()

    # --------------------------------------------------------
    # VALIDATION vs TEST
    # --------------------------------------------------------

    relative_val_size = (
        val_size / (val_size + test_size)
    )

    gss_val_test = GroupShuffleSplit(
        n_splits=1,
        train_size=relative_val_size,
        random_state=RANDOM_SEED
    )

    val_idx, test_idx = next(
        gss_val_test.split(
            val_test_df,
            groups=val_test_df["lesion_id"]
        )
    )

    val_df = val_test_df.iloc[val_idx].copy()

    test_df = val_test_df.iloc[test_idx].copy()

    # --------------------------------------------------------
    # PRINT DATASET SIZES
    # --------------------------------------------------------

    print("\nDataset split:")
    print(f"Train:      {len(train_df)}")
    print(f"Validation: {len(val_df)}")
    print(f"Test:       {len(test_df)}")

    # --------------------------------------------------------
    # PRINT CLASS DISTRIBUTION
    # --------------------------------------------------------

    print("\nTraining class distribution:")

    for class_name in CLASSES:
        count = (
            train_df["dx"] == class_name
        ).sum()

        print(
            f"  {class_name}: {count}"
        )

    print("\nValidation class distribution:")

    for class_name in CLASSES:
        count = (
            val_df["dx"] == class_name
        ).sum()

        print(
            f"  {class_name}: {count}"
        )

    print("\nTest class distribution:")

    for class_name in CLASSES:
        count = (
            test_df["dx"] == class_name
        ).sum()

        print(
            f"  {class_name}: {count}"
        )

    return (
        train_df,
        val_df,
        test_df
    )


# ============================================================
# IMAGE GENERATORS
# ============================================================

def create_generators(
    train_df,
    val_df,
    test_df
):
    """
    Create 7-class image generators.

    IMPORTANT:
    The class order is exactly the order
    defined in utils/config.py:

        0 -> mel
        1 -> bcc
        2 -> akiec
        3 -> nv
        4 -> bkl
        5 -> df
        6 -> vasc
    """

    # --------------------------------------------------------
    # TRAINING AUGMENTATION
    # --------------------------------------------------------

    train_datagen = (
        tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode="nearest"
        )
    )

    # --------------------------------------------------------
    # VALIDATION / TEST
    # --------------------------------------------------------

    val_test_datagen = (
        tf.keras.preprocessing.image.ImageDataGenerator()
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train_gen = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="image_path",
        y_col="dx",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        classes=CLASSES,
        class_mode="categorical",
        shuffle=True,
        seed=RANDOM_SEED
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    val_gen = val_test_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col="image_path",
        y_col="dx",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        classes=CLASSES,
        class_mode="categorical",
        shuffle=False
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    test_gen = val_test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col="image_path",
        y_col="dx",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        classes=CLASSES,
        class_mode="categorical",
        shuffle=False
    )

    # --------------------------------------------------------
    # VERIFY CLASS MAPPING
    # --------------------------------------------------------

    print("\nGenerator class mapping:")

    print(train_gen.class_indices)

    expected_mapping = {
        class_name: index
        for index, class_name in enumerate(CLASSES)
    }

    if train_gen.class_indices != expected_mapping:
        raise ValueError(
            "\nERROR: Generator class mapping does not match "
            "the configured class order.\n"
            f"Expected: {expected_mapping}\n"
            f"Got:      {train_gen.class_indices}"
        )

    print("\n7-class generator mapping verified successfully.")

    return (
        train_gen,
        val_gen,
        test_gen
    )