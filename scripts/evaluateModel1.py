import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import os
def load_hog_data(data_folder, labels_file):
    df = pd.read_csv(labels_file)
    X, y = [], []
    for img_name, label in zip(df["filename"], df["label"]):
        feature_path = os.path.join(data_folder, img_name.replace('.jpg', '.npy'))
        if os.path.exists(feature_path):
            X.append(np.load(feature_path))
            y.append(1 if label == "valid" else 0)  # Binary classification: valid = 1, invalid = 0
    return np.array(X), np.array(y)

# Load the test data
X_test, y_test = load_hog_data("data/processed/cnn_hog/test", "data/model_1_cnn_hog/labels_test.csv")

# Load the trained CNN + HOG model
cnn_hog_model = load_model("models/cnn_hog_model.h5")

# Evaluate the model
y_pred = cnn_hog_model.predict(X_test)
y_pred = (y_pred > 0.5)  # Convert probabilities to class labels (0 or 1)

# Print accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy for CNN + HOG model: {accuracy * 100:.2f}%")
print("\nClassification Report for CNN + HOG model:\n")
print(classification_report(y_test, y_pred))
