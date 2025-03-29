import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
import os

# Define paths
efficientnet_path = "data/model_2_efficientnet/"
train_path = os.path.join(efficientnet_path, "train")
test_path = os.path.join(efficientnet_path, "test")

# Image data generator with augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

test_datagen = ImageDataGenerator(rescale=1.0/255)

# Load data
train_generator = train_datagen.flow_from_directory(
    efficientnet_path,
    target_size=(128, 128),
    batch_size=16,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    efficientnet_path,
    target_size=(128, 128),
    batch_size=16,
    class_mode='binary'
)

# Load EfficientNet model (without the top layer)
base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
x = Flatten()(base_model.output)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
output_layer = Dense(1, activation="sigmoid")(x)

# Build model
model = Model(inputs=base_model.input, outputs=output_layer)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train model
model.fit(train_generator, epochs=10, validation_data=test_generator)

# Save model
model.save(os.path.join(efficientnet_path, "efficientnet_model.h5"))

print("✅ EfficientNet Model Trained and Saved!")
