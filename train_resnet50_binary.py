import tensorflow as tf
import os
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -------------------------------------------------------
# 1. Paths
# -------------------------------------------------------
train_dir = "dataset/train"
val_dir = "dataset/val"

# -------------------------------------------------------
# 2. Count images for class weights
# -------------------------------------------------------
benign_count = len(os.listdir(os.path.join(train_dir, "benign")))
malignant_count = len(os.listdir(os.path.join(train_dir, "malignant")))
total = benign_count + malignant_count

class_weight = {
    0: total / benign_count,       # benign
    1: total / malignant_count     # malignant
}

print("\nClass counts:")
print("Benign:", benign_count)
print("Malignant:", malignant_count)
print("\nClass weights:", class_weight)

# -------------------------------------------------------
# 3. Load pretrained ResNet50
# -------------------------------------------------------
base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze all layers initially
for layer in base_model.layers:
    layer.trainable = False

# -------------------------------------------------------
# 4. Add classification head
# -------------------------------------------------------
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

# -------------------------------------------------------
# 5. Compile (initial training)
# -------------------------------------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------------------------------------
# 6. Data Generators
# -------------------------------------------------------
img_size = 224
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1/255.,
    rotation_range=20,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1/255.)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="binary"
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="binary"
)

# -------------------------------------------------------
# 7. Train initial model
# -------------------------------------------------------
print("\n===== STAGE 1: TRAINING TOP LAYERS ONLY =====\n")

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    class_weight=class_weight
)

# -------------------------------------------------------
# 8. FINE‑TUNING — Unfreeze last 30 layers
# -------------------------------------------------------
print("\n===== STAGE 2: FINE‑TUNING LAST 30 LAYERS =====\n")

for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # smaller LR
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history_finetune = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,     # Fine‑tuning stage
    class_weight=class_weight
)

# -------------------------------------------------------
# 9. Save final model
# -------------------------------------------------------
model.save("resnet50_binary_skin_cancer.keras")
print("\nModel saved successfully as: resnet50_binary_skin_cancer.keras")
