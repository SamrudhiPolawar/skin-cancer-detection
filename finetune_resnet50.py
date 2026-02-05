import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

train_dir = "dataset/train"
val_dir = "dataset/val"

# Load trained model
model = load_model("models/resnet50_binary_skin_cancer.keras")
print("Model loaded. Unfreezing deeper layers...")

# Unfreeze last 50 layers of ResNet50
for layer in model.layers[-50:]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

train_gen = ImageDataGenerator(
    rescale=1/255.,
    rotation_range=20,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True
).flow_from_directory(
    train_dir, target_size=(224, 224),
    batch_size=16, class_mode="binary"
)

val_gen = ImageDataGenerator(rescale=1/255.).flow_from_directory(
    val_dir, target_size=(224, 224),
    batch_size=16, class_mode="binary"
)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5
)

model.save("models/resnet50_finetuned.keras")
print("Fine-tuned model saved!")
