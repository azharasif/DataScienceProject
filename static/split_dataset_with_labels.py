import os
import shutil
import random
import pandas as pd

# Define paths
raw_data_path = "data/raw/"
cnn_hog_path = "data/model_1_cnn_hog/"
efficientnet_path = "data/model_2_efficientnet/"
labels_file = "data/labels.csv"

# Load labels
df = pd.read_csv(labels_file)

# Create train/test folders
for path in [cnn_hog_path, efficientnet_path]:
    os.makedirs(os.path.join(path, "train"), exist_ok=True)
    os.makedirs(os.path.join(path, "test"), exist_ok=True)

# Shuffle dataset
all_images = df["filename"].tolist()
random.shuffle(all_images)

# Split into CNN-HOG and EfficientNet datasets
cnn_hog_images = all_images[:64]
efficientnet_images = all_images[64:]

# Function to split images and labels
def split_and_copy(images, dest_folder, label_df):
    train_size = int(0.8 * len(images))  # 80% training, 20% testing
    train_images = images[:train_size]
    test_images = images[train_size:]

    # Copy images
    for img in train_images:
        shutil.copy(os.path.join(raw_data_path, img), os.path.join(dest_folder, "train", img))
    for img in test_images:
        shutil.copy(os.path.join(raw_data_path, img), os.path.join(dest_folder, "test", img))

    # Save label CSVs
    label_df[label_df["filename"].isin(train_images)].to_csv(os.path.join(dest_folder, "labels_train.csv"), index=False)
    label_df[label_df["filename"].isin(test_images)].to_csv(os.path.join(dest_folder, "labels_test.csv"), index=False)

# Split and copy images + labels
split_and_copy(cnn_hog_images, cnn_hog_path, df)
split_and_copy(efficientnet_images, efficientnet_path, df)

print("Dataset and labels successfully split into training and testing sets!")
