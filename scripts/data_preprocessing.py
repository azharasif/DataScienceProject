import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog
import cv2
import numpy as np
import pandas as pd


def preprocess_images_and_labels(image_folder, labels_csv):
    """Preprocess images and extract labels."""
    # Read labels from CSV
    labels_df = pd.read_csv(labels_csv)
    image_files = labels_df['filename'].values
    labels = labels_df['label'].values
    
    images = []
    hog_features = []

    for image_file in image_files:
        # Read and preprocess image
        img_path = f'{image_folder}/{image_file}'
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load image in grayscale
        img = cv2.resize(img, (128, 128))  # Resize to consistent size
        
        # Extract HOG features
        fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
        
        images.append(img)
        hog_features.append(fd)
    
    images = np.array(images)
    hog_features = np.array(hog_features)
    
    return images, hog_features, labels

def preprocess_cnn_hog_data():
    """Preprocess data for CNN + HOG model."""
    # Preprocess data for CNN + HOG
    images_train_1, hog_train_1, labels_train_1 = preprocess_images_and_labels('data/model_1_cnn_hog/train', 'data/labels.csv')
    images_test_1, hog_test_1, labels_test_1 = preprocess_images_and_labels('data/model_1_cnn_hog/test', 'data/labels.csv')

    # Normalize images
    images_train_1 = images_train_1 / 255.0
    images_test_1 = images_test_1 / 255.0

    return images_train_1, hog_train_1, labels_train_1, images_test_1, hog_test_1, labels_test_1



def preprocess_images(image_folder, labels_csv):
    """Preprocess images for EfficientNet."""
    # Read labels from CSV
    labels_df = pd.read_csv(labels_csv)
    image_files = labels_df['filename'].values
    labels = labels_df['label'].values
    
    images = []
    
    for image_file in image_files:
        # Read image
        img_path = f'{image_folder}/{image_file}'
        img = cv2.imread(img_path)  # Load in RGB (default)
        img = cv2.resize(img, (128, 128))  # Resize for EfficientNet
        
        images.append(img)
    
    images = np.array(images)
    
    return images, labels

def preprocess_efficientnet_data():
    """Preprocess data for EfficientNet model."""
    # Preprocess data for EfficientNet
    images_train_2, labels_train_2 = preprocess_images('data/model_2_efficientnet/train', 'data/labels.csv')
    images_test_2, labels_test_2 = preprocess_images('data/model_2_efficientnet/test', 'data/labels.csv')

    # Normalize images
    images_train_2 = images_train_2 / 255.0
    images_test_2 = images_test_2 / 255.0

    return images_train_2, labels_train_2, images_test_2, labels_test_2
