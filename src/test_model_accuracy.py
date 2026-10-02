import os
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "efficientnetb0_best.keras"
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

CLASS_CODES = [
    "mel",
    "bcc",
    "akiec",
    "nv",
    "bkl",
    "df",
    "vasc"
]

CANCER_CLASSES = ["mel", "bcc"]

print("=" * 60)
print("LOADING MODEL")
print("=" * 60)

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

df = pd.read_csv(METADATA_PATH)

# ---------------------------------------------------------
# Find image path
# ---------------------------------------------------------

def find_image(image_id):

    path1 = os.path.join(IMAGE_DIR_1, image_id + ".jpg")
    path2 = os.path.join(IMAGE_DIR_2, image_id + ".jpg")

    if os.path.exists(path1):
        return path1

    if os.path.exists(path2):
        return path2

    return None


# ---------------------------------------------------------
# Select known cancer and non-cancer images
# ---------------------------------------------------------

cancer_df = df[df["dx"].isin(CANCER_CLASSES)].head(10)

non_cancer_df = df[
    df["dx"].isin(["nv", "bkl", "df", "vasc"])
].head(10)

test_df = pd.concat([cancer_df, non_cancer_df])

correct = 0
total = 0

print("\n")
print("=" * 60)
print("TESTING IMAGES")
print("=" * 60)

for _, row in test_df.iterrows():

    image_id = row["image_id"]
    actual_class = row["dx"]

    image_path = find_image(image_id)

    if image_path is None:
        print("Image not found:", image_id)
        continue

    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))

    img_array = np.asarray(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)[0]

    predicted_index = int(np.argmax(prediction))
    predicted_class = CLASS_CODES[predicted_index]

    actual_category = (
        "CANCER"
        if actual_class in CANCER_CLASSES
        else "NON-CANCER"
    )

    predicted_category = (
        "CANCER"
        if predicted_class in CANCER_CLASSES
        else "NON-CANCER"
    )

    is_correct = actual_category == predicted_category

    if is_correct:
        correct += 1

    total += 1

    result = "CORRECT" if is_correct else "WRONG"

    print(
        f"{image_id} | "
        f"Actual: {actual_class} ({actual_category}) | "
        f"Predicted: {predicted_class} ({predicted_category}) | "
        f"{result}"
    )

print("\n")
print("=" * 60)
print("FINAL RESULT")
print("=" * 60)

print(f"Correct: {correct}")
print(f"Total:   {total}")

if total > 0:
    accuracy = correct / total * 100
    print(f"Accuracy: {accuracy:.2f}%")

print("=" * 60)