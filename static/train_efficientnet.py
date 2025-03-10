import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Data generators
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    "data/model_2_efficientnet/train",
    target_size=(224, 224),
    batch_size=8,
    class_mode='binary',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    "data/model_2_efficientnet/train",
    target_size=(224, 224),
    batch_size=8,
    class_mode='binary',
    subset='validation'
)

# EfficientNet model
base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, validation_data=val_generator, epochs=10)

# Save the model
model.save("models/efficientnet_model.h5")
