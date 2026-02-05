import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Folder Paths
train_dir = "dataset/train"
val_dir = "dataset/val"

# =============================
# Data Augmentation
# =============================
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)

val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary"
)

val_data = val_gen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary"
)

# =============================
# Load Pretrained ResNet50
# =============================
base = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base.trainable = False   # Freeze the feature extractor

# =============================
# Add Custom Layers
# =============================
x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(128, activation="relu")(x)
out = Dense(1, activation="sigmoid")(x)  # Binary classification

model = Model(inputs=base.input, outputs=out)

# =============================
# Compile Model
# =============================
model.compile(
    optimizer=Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

model.summary()

# =============================
# Train the Network
# =============================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=15
)

# =============================
# Save the Model
# =============================
model.save("resnet50_ham10000_supervised.h5")
print("Model saved as resnet50_ham10000_supervised.h5")
