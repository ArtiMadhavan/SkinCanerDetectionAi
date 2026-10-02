import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Skin Cancer Detection AI",
    page_icon="🩺",
    layout="centered"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "efficientnetb0_best.keras"


# ============================================================
# 7 CLASS ORDER
# IMPORTANT: This must match the model's training class order.
# ============================================================

CLASS_CODES = [
    "mel",
    "bcc",
    "akiec",
    "nv",
    "bkl",
    "df",
    "vasc"
]

CLASS_NAMES = {
    "mel": "Melanoma",
    "bcc": "Basal Cell Carcinoma",
    "akiec": "Actinic Keratoses",
    "nv": "Melanocytic Nevus",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "vasc": "Vascular Lesion"
}


# ============================================================
# RESULT MAPPING
# ============================================================

CANCER_CLASSES = {
    "mel",
    "bcc"
}

MEDICAL_CHECK_CLASSES = {
    "akiec"
}

NON_CANCER_CLASSES = {
    "nv",
    "bkl",
    "df",
    "vasc"
}


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    # Make sure this is really the expected 7-class model.
    if model.output_shape[-1] != 7:
        raise ValueError(
            f"Expected a 7-class model, but the loaded model "
            f"has {model.output_shape[-1]} outputs."
        )

    return model


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):
    image = image.convert("RGB")

    image = image.resize((224, 224))

    image_array = np.asarray(image).astype(np.float32)

    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# ============================================================
# PREDICTION
# ============================================================

def predict_image(model, image):

    processed_image = preprocess_image(image)

    predictions = model.predict(
        processed_image,
        verbose=0
    )[0]

    predicted_index = int(np.argmax(predictions))

    predicted_code = CLASS_CODES[predicted_index]

    predicted_name = CLASS_NAMES[predicted_code]

    confidence = float(predictions[predicted_index]) * 100

    return predicted_code, predicted_name, confidence


# ============================================================
# TITLE
# ============================================================

st.title("🩺 Skin Cancer Detection AI")

st.write(
    "Upload a skin lesion image and the AI model will "
    "classify it into one of the supported categories."
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a skin image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# ANALYZE
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button(
        "🔍 Analyze Image",
        use_container_width=True
    ):

        try:

            model = load_model()

            predicted_code, predicted_name, confidence = predict_image(
                model,
                image
            )


            # ==================================================
            # CANCER
            # ==================================================

            if predicted_code in CANCER_CLASSES:

                st.error(
                    f"🔴 CANCER DETECTED\n\n"
                    f"**Predicted Class:** {predicted_name}\n\n"
                    f"**Confidence:** {confidence:.2f}%"
                )


            # ==================================================
            # NEEDS MEDICAL CHECK
            # ==================================================

            elif predicted_code in MEDICAL_CHECK_CLASSES:

                st.warning(
                    f"🟠 NEEDS MEDICAL CHECK\n\n"
                    f"**Predicted Class:** {predicted_name}\n\n"
                    f"**Confidence:** {confidence:.2f}%"
                )


            # ==================================================
            # NOT CANCER
            # ==================================================

            elif predicted_code in NON_CANCER_CLASSES:

                st.success(
                    f"🟢 NOT CANCER\n\n"
                    f"**Predicted Class:** {predicted_name}\n\n"
                    f"**Confidence:** {confidence:.2f}%"
                )


            # ==================================================
            # SAFETY FALLBACK
            # ==================================================

            else:

                st.warning(
                    f"⚠️ UNRECOGNIZED RESULT\n\n"
                    f"**Predicted Class:** {predicted_name}\n\n"
                    f"**Confidence:** {confidence:.2f}%"
                )


        except Exception as e:

            st.error(
                f"Error while analyzing the image:\n\n{e}"
            )


