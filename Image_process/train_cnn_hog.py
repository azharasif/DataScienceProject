import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import os
import pandas as pd

# Load HOG features from processed data
def load_data(data_folder, labels_file):
    df = pd.read_csv(labels_file)
    print('df----',df["filename"] , df["label"])
    X, y = [], []

    for img_name, label in zip(df["filename"], df["label"]):
        feature_path = os.path.join(data_folder, img_name.replace('.jpg', '.npy'))
        if os.path.exists(feature_path):
            X.append(np.load(feature_path))
            y.append(1 if label == "valid" else 0) 

    return np.array(X), np.array(y)

X_train, y_train = load_data("data/processed/cnn_hog/train", "data/model_1_cnn_hog/labels_train.csv")
X_test, y_test = load_data("data/processed/cnn_hog/test", "data/model_1_cnn_hog/labels_test.csv")

model = Sequential([
    Flatten(input_shape=(X_train.shape[1],)),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=8)

model.save("models/cnn_hog_model.h5")
