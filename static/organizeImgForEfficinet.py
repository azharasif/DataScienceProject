import os
import shutil
import pandas as pd

# Load the CSV file containing the labels
labels_df = pd.read_csv('data/model_2_efficientnet/labels_train.csv')

# Define paths
base_path = 'data/model_2_efficientnet/train'
valid_folder = os.path.join(base_path, 'valid')
invalid_folder = os.path.join(base_path, 'invalid')

# Create subfolders if they don't exist
os.makedirs(valid_folder, exist_ok=True)
os.makedirs(invalid_folder, exist_ok=True)

# Loop through the CSV and move images to the appropriate folder
for _, row in labels_df.iterrows():
    img_name = row['filename']
    label = row['label']
    img_path = os.path.join(base_path, img_name)

    if label == 'valid':
        shutil.move(img_path, os.path.join(valid_folder, img_name))
    elif label == 'invalid':
        shutil.move(img_path, os.path.join(invalid_folder, img_name))

print("Images have been organized.")
