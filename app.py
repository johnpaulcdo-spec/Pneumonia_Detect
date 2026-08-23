import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

IMG_SIZE = 224
MODEL_PATH = "final_pneumonia_model.keras"


@st.cache_resource
def load_pneumonia_model():
    return load_model(MODEL_PATH)


def preprocess_uploaded_image(uploaded_file, img_size=224):
    img = Image.open(uploaded_file).convert("L")
    display_img = img.copy()

    img = img.resize((img_size, img_size))
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.repeat(img_array, 3, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array, display_img


st.set_page_config(page_title="Pneumonia Detection App", layout="centered")

st.title("Pneumonia Detection from Chest X-ray")
st.write(
    "Upload a chest X-ray image to obtain a pneumonia prediction. "
    "This tool is intended for educational and decision-support purposes only."
)

uploaded_file = st.file_uploader(
    "Upload a chest X-ray image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    try:
        model = load_pneumonia_model()
        processed_img, display_img = preprocess_uploaded_image(uploaded_file, IMG_SIZE)

        st.image(display_img, caption="Uploaded Chest X-ray")

        pred_prob = float(model.predict(processed_img, verbose=0)[0][0])
        pred_label = 1 if pred_prob > 0.5 else 0

        if pred_label == 1:
            st.error("Prediction: Pneumonia Positive")
            st.write(f"Confidence: {pred_prob:.2%}")
        else:
            st.success("Prediction: Pneumonia Negative")
            st.write(f"Confidence: {(1 - pred_prob):.2%}")

        st.caption(
            "Disclaimer: This output is intended as decision support only and "
            "must not be used as a standalone clinical diagnosis."
        )

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
