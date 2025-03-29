import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Define paths
cnn_hog_path = "data/model_1_cnn_hog/"
train_path = os.path.join(cnn_hog_path, "train")
test_path = os.path.join(cnn_hog_path, "test")

def extract_hog_features(image_path):
    # image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # if image is None:
    #     print(f"Error loading image: {image_path}")
    #     return None
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) 
    print('image1: ', image)
    image = cv2.resize(image, (128, 128))  
    print('image2å: ', image)
    features, _ = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features

train_images = os.listdir(train_path)
train_features = []
train_labels = []

for img in train_images:
    img_path = os.path.join(train_path, img)
    print('img_path: ', img_path)
    features = extract_hog_features(img_path)
    train_features.append(features)
    label = 1 if "valid" in img else 0  
    train_labels.append(label)

train_features = np.array(train_features)
train_labels = np.array(train_labels)

svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(train_features, train_labels)

joblib.dump(svm_model, os.path.join(cnn_hog_path, "svm_model.pkl"))

test_images = os.listdir(test_path)
test_features = []
test_labels = []

for img in test_images:
    img_path = os.path.join(test_path, img)
    features = extract_hog_features(img_path)
    test_features.append(features)
    label = 1 if "valid" in img else 0  
    test_labels.append(label)

test_features = np.array(test_features)
test_labels = np.array(test_labels)

predictions = svm_model.predict(test_features)
accuracy = accuracy_score(test_labels, predictions)

print(f"✅ CNN + HOG Model Accuracy: {accuracy * 100:.2f}%")

