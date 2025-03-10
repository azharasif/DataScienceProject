
import os
import cv2
import numpy as np
from skimage.feature import hog
from skimage import color
from PIL import Image
import pandas as pd
def preprocess_images(input_folder, output_folder, labels_file, use_hog=False):
    os.makedirs(output_folder, exist_ok=True)

    df = pd.read_csv(labels_file)

    print('df----',df["filename"])
    for img_name in df["filename"]:
        img_path = os.path.join(input_folder, img_name)
        
        img = cv2.imread(img_path)
        
        if img is None:
                print("could not be loaded due to error")
                continue
        
        img = cv2.resize(img, (224, 224))  
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  

        if use_hog:
            features, _ = hog(img_gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
            np.save(os.path.join(output_folder, img_name.replace('.jpg', '.npy')), features)
     

# Preprocess datasets for CNN + HOG
preprocess_images("data/model_1_cnn_hog/train", "data/processed/cnn_hog/train", "data/model_1_cnn_hog/labels_train.csv", use_hog=True)
preprocess_images("data/model_1_cnn_hog/test", "data/processed/cnn_hog/test", "data/model_1_cnn_hog/labels_test.csv", use_hog=True)

