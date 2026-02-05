import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

MODEL_PATH = "resnet50_binary_skin_cancer.keras"

# Load model WITHOUT optimizer
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded successfully!")

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0][0]
    confidence = float(pred)

    if confidence >= 0.5:
        result = "MALIGNANT"
    else:
        result = "BENIGN"

    print(f"\nPrediction: {result}")
    print(f"Confidence Score: {confidence:.4f}")

    return result, confidence

if __name__ == "__main__":
    predict_image("test_image4.jpg")
