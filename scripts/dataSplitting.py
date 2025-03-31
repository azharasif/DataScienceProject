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

train_size = int(0.7 * len(all_images))  
test_size = int(0.2 * len(all_images))  
real_test_size = len(all_images) - train_size - test_size  

train_images = all_images[:train_size]
test_images = all_images[train_size:train_size + test_size]
real_test_images = all_images[train_size + test_size:]

for path in [cnn_hog_path, efficientnet_path]:
    os.makedirs(os.path.join(path, "train"), exist_ok=True)
    os.makedirs(os.path.join(path, "test"), exist_ok=True)
os.makedirs(real_test_path, exist_ok=True)

def copy_images(images, src_folder, dest_folder):
    for img in images:
        src_path = os.path.join(src_folder, img)
        dest_path = os.path.join(dest_folder, img)
        
        if os.path.exists(src_path):
            shutil.copy(src_path, dest_path)
        else:
            print(f"{img} not found in {src_folder}")

copy_images(train_images, raw_data_path, os.path.join(cnn_hog_path, "train"))
copy_images(train_images, raw_data_path, os.path.join(efficientnet_path, "train"))

copy_images(test_images, raw_data_path, os.path.join(cnn_hog_path, "test"))
copy_images(test_images, raw_data_path, os.path.join(efficientnet_path, "test"))

copy_images(real_test_images, raw_data_path, real_test_path)

def save_csv(filenames, folder, filename):
    df = pd.DataFrame({"filename": filenames})
    df.to_csv(os.path.join(folder, filename), index=False)

save_csv(train_images, cnn_hog_path, "train_filenames.csv")
save_csv(test_images, cnn_hog_path, "test_filenames.csv")

save_csv(train_images, efficientnet_path, "train_filenames.csv")
save_csv(test_images, efficientnet_path, "test_filenames.csv")

save_csv(real_test_images, real_test_path, "real_test_filenames.csv")

print(" Dataset successfully split into training, testing, and real test sets!")
