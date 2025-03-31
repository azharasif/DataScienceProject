import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog

cnn_hog_path = "data/model_1_cnn_hog/"
train_path = os.path.join(cnn_hog_path, "train")

df = pd.read_csv("data/labels.csv")

def extract_hog_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"⚠️ Error loading image: {image_path}")
        return None
    image = cv2.resize(image, (128, 128))
    features, _ = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features

hog_features = []
image_names = []

for img in df["filename"]:
    img_path = os.path.join(train_path, img)
    features = extract_hog_features(img_path)
    if features is not None:
        hog_features.append(features)
        image_names.append(img)

hog_features = np.array(hog_features)

# Save extracted HOG features
np.save("data/hog_features.npy", hog_features)

# Save corresponding filenames
pd.DataFrame({"filename": image_names}).to_csv("data/hog_filenames.csv", index=False)

print(" HOG feature extraction for CNN + HOG model completed!")
