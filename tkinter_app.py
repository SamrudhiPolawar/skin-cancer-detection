import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import tensorflow as tf
import numpy as np
from gradcam import generate_gradcam

MODEL_PATH = "models/resnet50_binary_skin_cancer.keras"
model = tf.keras.models.load_model(MODEL_PATH)

def select_image():
    path = filedialog.askopenfilename()
    if not path:
        return

    img = Image.open(path).resize((224, 224))
    img_tk = ImageTk.PhotoImage(img)
    panel.config(image=img_tk)
    panel.image = img_tk

    # Prediction
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, 0)
    pred = model.predict(arr)[0][0]

    label = "MALIGNANT" if pred >= 0.5 else "BENIGN"
    result_text.set(f"Prediction: {label}  |  Score: {pred:.4f}")

    # Grad‑CAM
    generate_gradcam(path, "gradcam_gui.jpg")
    heat_img = Image.open("gradcam_gui.jpg").resize((224, 224))
    heat_tk = ImageTk.PhotoImage(heat_img)
    panel2.config(image=heat_tk)
    panel2.image = heat_tk

root = tk.Tk()
root.title("Skin Cancer Classifier")

btn = tk.Button(root, text="Select Image", command=select_image)
btn.pack()

panel = tk.Label(root)
panel.pack()

panel2 = tk.Label(root)
panel2.pack()

result_text = tk.StringVar()
result_label = tk.Label(root, textvariable=result_text, font=("Arial", 14))
result_label.pack()

root.mainloop()
