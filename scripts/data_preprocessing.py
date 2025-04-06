import os
import cv2
import numpy as np
import pandas as pd

def load_data(image_folder, labels_csv, label_column='MEDICINE_NAME'):
    """Load and preprocess images and labels from CSV file."""
    labels_df = pd.read_csv(labels_csv)

    if label_column not in labels_df.columns:
        raise ValueError(f"'{label_column}' not found in CSV columns: {labels_df.columns.tolist()}")

    image_files = labels_df['IMAGE'].values
    labels = labels_df[label_column].values

    images = []
    valid_labels = []

    for img_file, label in zip(image_files, labels):
        img_path = os.path.join(image_folder, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # Load in BGR format

        if img is None:
            print(f"[Warning] Unable to read image {img_path}. Skipping.")
            continue

        img = cv2.resize(img, (128, 128))
        images.append(img)
        valid_labels.append(label)

    return np.array(images), np.array(valid_labels)
