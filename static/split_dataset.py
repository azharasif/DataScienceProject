import os
import shutil
import random

# Define paths
raw_data_path = "data/raw/"
cnn_hog_path = "data/model_1_cnn_hog/"
efficientnet_path = "data/model_2_efficientnet/"

# Create train/test folders
for path in [cnn_hog_path, efficientnet_path]:
    os.makedirs(os.path.join(path, "train"), exist_ok=True)
    os.makedirs(os.path.join(path, "test"), exist_ok=True)

# Get all image filenames
all_images = sorted(os.listdir(raw_data_path))  # Sorting ensures consistency
random.shuffle(all_images)  # Shuffle images randomly

# Split dataset (64 images for CNN + HOG, 65 images for EfficientNet)
cnn_hog_images = all_images[:64]
efficientnet_images = all_images[64:]

# Function to split into train/test (80% train, 20% test)
def split_and_copy(images, dest_folder):
    train_size = int(0.8 * len(images))  # 80% training
    train_images = images[:train_size]
    test_images = images[train_size:]

    for img in train_images:
        shutil.copy(os.path.join(raw_data_path, img), os.path.join(dest_folder, "train", img))
    for img in test_images:
        shutil.copy(os.path.join(raw_data_path, img), os.path.join(dest_folder, "test", img))

# Perform the split
split_and_copy(cnn_hog_images, cnn_hog_path)
split_and_copy(efficientnet_images, efficientnet_path)

print("Dataset successfully split into training and testing sets!")
