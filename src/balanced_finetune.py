import os
import sys
from pathlib import Path

import pandas as pd
import tensorflow as tf
from sklearn.utils import resample


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():

    print("\n========================================")
    print("BALANCED FINE-TUNING - EFFICIENTNETB0")
    print("========================================")

    from src.dataset import load_metadata
    from src.preprocessing import split_data, create_generators
    from utils.config import MODELS_DIR

    # --------------------------------------------------
    # 1. Load metadata
    # --------------------------------------------------

    print("\nLoading metadata...")
    df = load_metadata()

    print(f"Total images: {len(df)}")

    # --------------------------------------------------
    # 2. Secure train / validation / test split
    # --------------------------------------------------

    print("\nSplitting data securely by lesion_id...")

    train_df, val_df, test_df = split_data(df)

    print(f"Training images:   {len(train_df)}")
    print(f"Validation images: {len(val_df)}")
    print(f"Test images:       {len(test_df)}")

    # --------------------------------------------------
    # 3. Balance training data
    # --------------------------------------------------

    print("\nOriginal training distribution:")
    print(train_df["dx"].value_counts())

    print("\nCreating balanced training dataset...")

    class_counts = train_df["dx"].value_counts()

    target_samples = int(class_counts.median())

    print(f"\nBalanced target per class: {target_samples}")

    balanced_parts = []

    for class_name in sorted(train_df["dx"].unique()):

        class_df = train_df[
            train_df["dx"] == class_name
        ]

        if len(class_df) < target_samples:

            class_balanced = resample(
                class_df,
                replace=True,
                n_samples=target_samples,
                random_state=42
            )

        else:

            class_balanced = resample(
                class_df,
                replace=False,
                n_samples=target_samples,
                random_state=42
            )

        balanced_parts.append(class_balanced)

    balanced_train_df = pd.concat(
        balanced_parts,
        ignore_index=True
    )

    balanced_train_df = balanced_train_df.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    print("\nBalanced training distribution:")
    print(balanced_train_df["dx"].value_counts())

    print(
        f"\nTotal balanced training images: "
        f"{len(balanced_train_df)}"
    )

    # --------------------------------------------------
    # 4. Create generators
    # --------------------------------------------------

    print("\nCreating data generators...")

    train_gen, val_gen, test_gen = create_generators(
        balanced_train_df,
        val_df,
        test_df
    )

    # --------------------------------------------------
    # 5. Load balanced model
    # --------------------------------------------------

    model_path = os.path.join(
        MODELS_DIR,
        "efficientnetb0_balanced_best.keras"
    )

    output_path = os.path.join(
        MODELS_DIR,
        "efficientnetb0_balanced_finetuned_best.keras"
    )

    print("\nLoading balanced model...")

    if not os.path.exists(model_path):

        raise FileNotFoundError(
            f"Model not found:\n{model_path}"
        )

    model = tf.keras.models.load_model(
        model_path
    )

    print("Balanced model loaded successfully.")

    # --------------------------------------------------
    # 6. Freeze everything first
    # --------------------------------------------------

    print("\nFreezing all layers...")

    for layer in model.layers:
        layer.trainable = False

    # --------------------------------------------------
    # 7. Unfreeze upper EfficientNet layers
    # --------------------------------------------------

    print("\nUnfreezing upper EfficientNet layers...")

    for layer in model.layers:

        if layer.name.startswith("block6") or \
           layer.name.startswith("block7"):

            layer.trainable = True

        # Keep BatchNorm frozen
        if isinstance(
            layer,
            tf.keras.layers.BatchNormalization
        ):

            layer.trainable = False

    trainable_count = sum(
        1 for layer in model.layers
        if layer.trainable
    )

    print(
        f"Trainable layers: {trainable_count}"
    )

    # --------------------------------------------------
    # 8. Compile
    # --------------------------------------------------

    print("\nCompiling model...")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-5
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # --------------------------------------------------
    # 9. Callbacks
    # --------------------------------------------------

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=output_path,
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

    # --------------------------------------------------
    # 10. Fine-tuning
    # --------------------------------------------------

    print("\n========================================")
    print("STARTING BALANCED FINE-TUNING")
    print("========================================")

    print("\nMaximum epochs: 10")
    print("Learning rate: 1e-5")
    print("Batch normalization layers remain frozen.")
    print("Test dataset is NOT used for training.")

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10,
        callbacks=[
            checkpoint,
            early_stopping,
            reduce_lr
        ],
        verbose=1
    )

    # --------------------------------------------------
    # 11. Finished
    # --------------------------------------------------

    print("\n========================================")
    print("BALANCED FINE-TUNING COMPLETED")
    print("========================================")

    print("\nBest model saved at:")
    print(output_path)


if __name__ == "__main__":
    main()