import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

from utils.config import RESULTS_DIR, CLASSES, MODELS_DIR
from src.dataset import load_metadata
from src.preprocessing import split_data, create_generators


def evaluate_model(model, test_gen):

    print("\n========================================")
    print("MODEL EVALUATION")
    print("========================================")

    test_gen.reset()

    print("\nRunning predictions on test set...")
    predictions = model.predict(
        test_gen,
        steps=len(test_gen),
        verbose=1
    )

    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes

    # Make sure predictions match test samples
    predictions = predictions[:len(y_true)]
    y_pred = y_pred[:len(y_true)]

    # Create result directories
    metrics_dir = os.path.join(RESULTS_DIR, "metrics")
    figures_dir = os.path.join(RESULTS_DIR, "figures")

    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # ========================================
    # 1. CLASSIFICATION REPORT
    # ========================================

    print("\n========================================")
    print("CLASSIFICATION REPORT")
    print("========================================")

    report_text = classification_report(
        y_true,
        y_pred,
        target_names=CLASSES,
        digits=4
    )

    print(report_text)

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=CLASSES,
        output_dict=True
    )

    pd.DataFrame(report_dict).transpose().to_csv(
        os.path.join(
            metrics_dir,
            "classification_report.csv"
        )
    )

    print("Classification report saved.")

    # ========================================
    # 2. TEST ACCURACY
    # ========================================

    accuracy = np.mean(y_true == y_pred)

    print("\n========================================")
    print(f"TEST ACCURACY: {accuracy * 100:.2f}%")
    print("========================================")

    with open(
        os.path.join(metrics_dir, "test_accuracy.txt"),
        "w"
    ) as f:
        f.write(f"Test Accuracy: {accuracy * 100:.2f}%\n")

    # ========================================
    # 3. SAVE PREDICTIONS
    # ========================================

    results_df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred
    })

    results_df.to_csv(
        os.path.join(
            metrics_dir,
            "predictions.csv"
        ),
        index=False
    )

    # ========================================
    # 4. CONFUSION MATRIX
    # ========================================

    print("\nGenerating confusion matrix...")

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES
    )

    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            figures_dir,
            "confusion_matrix.png"
        ),
        dpi=300
    )

    plt.close()

    print("Confusion matrix saved.")

    # ========================================
    # 5. ROC CURVES
    # ========================================

    print("\nGenerating ROC curves...")

    y_true_bin = label_binarize(
        y_true,
        classes=range(len(CLASSES))
    )

    plt.figure(figsize=(10, 8))

    for i in range(len(CLASSES)):

        fpr, tpr, _ = roc_curve(
            y_true_bin[:, i],
            predictions[:, i]
        )

        auc_score = roc_auc_score(
            y_true_bin[:, i],
            predictions[:, i]
        )

        plt.plot(
            fpr,
            tpr,
            label=f"{CLASSES[i]} (AUC = {auc_score:.2f})"
        )

    plt.plot(
        [0, 1],
        [0, 1],
        "k--"
    )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title(
        "ROC Curves - One-vs-Rest"
    )

    plt.legend(
        loc="lower right"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            figures_dir,
            "roc_curves.png"
        ),
        dpi=300
    )

    plt.close()

    print("ROC curves saved.")

    print("\n========================================")
    print("EVALUATION COMPLETED")
    print("========================================")

    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

    print("\nResults saved in:")
    print(metrics_dir)
    print(figures_dir)


if __name__ == "__main__":

    print("\nLoading test data...")

    df = load_metadata()

    _, _, test_df = split_data(df)

    # Create test generator
    _, _, test_gen = create_generators(
        df.iloc[0:1],
        df.iloc[0:1],
        test_df
    )

    print("\nLoading FINAL trained model...")

    model_path = os.path.join(
        MODELS_DIR,
        "efficientnetb0_best.keras"
    )

    print(f"Model: {model_path}")

    model = tf.keras.models.load_model(
        model_path
    )

    print("\nModel loaded successfully.")

    evaluate_model(
        model,
        test_gen
    )