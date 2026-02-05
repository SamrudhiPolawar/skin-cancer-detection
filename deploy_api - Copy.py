from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
model = tf.keras.models.load_model("resnet50_binary_skin_cancer.h5")

@app.post("/predict")
def predict():
    file = request.files["image"]
    img = image.load_img(file, target_size=(224, 224))
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]
    label = "malignant" if pred >= 0.5 else "benign"

    return jsonify({"prediction": label})

app.run()
