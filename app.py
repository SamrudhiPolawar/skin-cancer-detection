import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("resnet50_binary_skin_cancer.h5")

st.title("Skin Cancer Classification (Benign vs Malignant)")
uploaded = st.file_uploader("Upload a skin lesion image", type=["jpg", "png"])

if uploaded:
    img = Image.open(uploaded).resize((224, 224))
    st.image(img, caption="Uploaded Image")
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0][0]
    label = "MALIGNANT ❗" if pred >= 0.5 else "BENIGN ✔"

    st.markdown(f"### Prediction: **{label}**")
