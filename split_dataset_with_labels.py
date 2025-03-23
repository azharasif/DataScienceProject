import os
import shutil
import random
import pandas as pd

raw_data_path = "data/raw/"
cnn_hog_path = "data/model_1_cnn_hog/"
efficientnet_path = "data/model_2_efficientnet/"
real_test_path = "data/real_test/" 
labels_file = "data/labels.csv"

df = pd.read_csv(labels_file)

for path in [cnn_hog_path, efficientnet_path]:
    os.makedirs(os.path.join(path, "train"), exist_ok=True)
    os.makedirs(os.path.join(path, "test"), exist_ok=True)
os.makedirs(real_test_path, exist_ok=True) 

all_images = df["filename"].tolist()
random.shuffle(all_images) 

train_size = int(0.7 * len(all_images))  
test_size = int(0.2 * len(all_images))  
real_test_size = len(all_images) - train_size - test_size  

train_images = all_images[:train_size]  
test_images = all_images[train_size:train_size + test_size]  
real_test_images = all_images[train_size + test_size:]  

def copy_and_save_labels(images, dest_folder, label_df):
    copy_images(images, dest_folder)
    label_df[label_df["filename"].isin(images)].to_csv(os.path.join(dest_folder, "labels.csv"), index=False)

def copy_images(images, dest_folder):
    for img in images:
        shutil.copy(os.path.join(raw_data_path, img), os.path.join(dest_folder, img))

copy_and_save_labels(train_images, os.path.join(cnn_hog_path, "train"), df)
copy_and_save_labels(train_images, os.path.join(efficientnet_path, "train"), df)

copy_and_save_labels(test_images, os.path.join(cnn_hog_path, "test"), df)
copy_and_save_labels(test_images, os.path.join(efficientnet_path, "test"), df)

copy_images(real_test_images, real_test_path)

print("Dataset successfully split into training, testing, and real test sets!")
