import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from gradcam import generate_gradcam

MODEL_PATH = "models/resnet50_binary_skin_cancer.keras"
model = tf.keras.models.load_model(MODEL_PATH)

st.title("Skin Cancer Classifier (Benign vs Malignant)")
st.write("Upload an image to classify and generate Grad‑CAM visualization.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img_path = "uploaded.jpg"
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    img = image.load_img(img_path, target_size=(224, 224))
    st.image(img, caption="Uploaded Image", width=300)

    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0][0]
    label = "MALIGNANT" if pred >= 0.5 else "BENIGN"

    st.subheader(f"Prediction: **{label}**")
    st.write(f"Confidence: {pred:.4f}")

    # Grad‑CAM
    generate_gradcam(img_path, "gradcam.jpg")
    st.image("gradcam.jpg", caption="Grad-CAM Heatmap")
