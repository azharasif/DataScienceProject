import os
import shutil
import random
import pandas as pd

# Define paths
raw_data_path = "data/raw/"
cnn_hog_path = "data/model_1_cnn_hog/"
efficientnet_path = "data/model_2_efficientnet/"
real_test_path = "data/real_test/"
labels_file = "data/labels.csv"

# Load filenames (without labels)
df = pd.read_csv(labels_file)
all_images = df["filename"].tolist()

# Shuffle dataset for randomness
random.shuffle(all_images)

# **Split dataset (70% Train, 20% Test, 10% Real Test)**
train_size = int(0.7 * len(all_images))  # 90 images
test_size = int(0.2 * len(all_images))   # 26 images
real_test_size = len(all_images) - train_size - test_size  # 13 images

train_images = all_images[:train_size]
test_images = all_images[train_size:train_size + test_size]
real_test_images = all_images[train_size + test_size:]

# **Ensure output directories exist**
for path in [cnn_hog_path, efficientnet_path]:
    os.makedirs(os.path.join(path, "train"), exist_ok=True)
    os.makedirs(os.path.join(path, "test"), exist_ok=True)
os.makedirs(real_test_path, exist_ok=True)

# **Function to copy images**
def copy_images(images, dest_folder):
    for img in images:
        shutil.copy(os.path.join(raw_data_path, img), os.path.join(dest_folder, img))

# **Copy training & test images for both models**
copy_images(train_images, os.path.join(cnn_hog_path, "train"))
copy_images(train_images, os.path.join(efficientnet_path, "train"))

copy_images(test_images, os.path.join(cnn_hog_path, "test"))
copy_images(test_images, os.path.join(efficientnet_path, "test"))

# **Copy real test images (no labels)**
copy_images(real_test_images, real_test_path)

# **Save filenames for tracking**
pd.DataFrame({"filename": train_images}).to_csv(os.path.join(cnn_hog_path, "train_filenames.csv"), index=False)
pd.DataFrame({"filename": test_images}).to_csv(os.path.join(cnn_hog_path, "test_filenames.csv"), index=False)

pd.DataFrame({"filename": train_images}).to_csv(os.path.join(efficientnet_path, "train_filenames.csv"), index=False)
pd.DataFrame({"filename": test_images}).to_csv(os.path.join(efficientnet_path, "test_filenames.csv"), index=False)

pd.DataFrame({"filename": real_test_images}).to_csv(os.path.join(real_test_path, "real_test_filenames.csv"), index=False)

print("✅ Dataset successfully split into training, testing, and real test sets!")
