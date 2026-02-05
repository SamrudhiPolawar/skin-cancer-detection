import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

# Load model
model = tf.keras.models.load_model("resnet50_binary_skin_cancer.h5")

# Data generator for validation set
img_size = 224
batch_size = 32

datagen = ImageDataGenerator(rescale=1/255.)

val_gen = datagen.flow_from_directory(
    "dataset/val",
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="binary",
    shuffle=False
)

# Prediction
preds = model.predict(val_gen)
pred_labels = (preds > 0.5).astype(int)

# True labels
true_labels = val_gen.classes

# Confusion matrix
print(confusion_matrix(true_labels, pred_labels))
print(classification_report(true_labels, pred_labels))
