import os
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "efficientnetb0_binary_cancer.keras"
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

print("=" * 60)
print("LOADING BINARY MODEL")
print("=" * 60)

model = tf.keras.models.load_model(MODEL_PATH)

df = pd.read_csv(METADATA_PATH)

def find_image(image_id):
    p1 = os.path.join(IMAGE_DIR_1, image_id + ".jpg")
    p2 = os.path.join(IMAGE_DIR_2, image_id + ".jpg")

    if os.path.exists(p1):
        return p1
    if os.path.exists(p2):
        return p2

    return None

# Same 20 images used in the previous test
cancer_df = df[df["dx"].isin(["mel", "bcc"])].head(10)

non_cancer_df = df[
    df["dx"].isin(["nv", "bkl", "df", "vasc"])
].head(10)

test_df = pd.concat([cancer_df, non_cancer_df])

correct = 0
total = 0

print("\n")
print("=" * 60)
print("TESTING BINARY MODEL")
print("=" * 60)

for _, row in test_df.iterrows():

    image_id = row["image_id"]
    actual_dx = row["dx"]

    image_path = find_image(image_id)

    if image_path is None:
        continue

    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))

    img_array = np.asarray(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)[0]

    non_cancer_prob = prediction[0] * 100
    cancer_prob = prediction[1] * 100

    predicted_category = "CANCER" if cancer_prob >= non_cancer_prob else "NON-CANCER"

    actual_category = (
        "CANCER"
        if actual_dx in ["mel", "bcc"]
        else "NON-CANCER"
    )

    is_correct = predicted_category == actual_category

    if is_correct:
        correct += 1

    total += 1

    result = "CORRECT" if is_correct else "WRONG"

    print(
        f"{image_id} | "
        f"Actual: {actual_category} | "
        f"Non-Cancer: {non_cancer_prob:.2f}% | "
        f"Cancer: {cancer_prob:.2f}% | "
        f"Predicted: {predicted_category} | "
        f"{result}"
    )

print("\n")
print("=" * 60)
print("FINAL RESULT")
print("=" * 60)

print(f"Correct: {correct}")
print(f"Total:   {total}")

if total:
    print(f"Accuracy: {correct / total * 100:.2f}%")

print("=" * 60)