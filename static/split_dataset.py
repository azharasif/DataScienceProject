import os
import shutil
import random

raw_data_path = "./data/raw/"
cnn_hog_path = "./data/model_1_cnn_hog/"
efficientnet_path = "./data/model_2_efficientnet/"
real_test_path = "./data/real_test/"  

for path in [cnn_hog_path, efficientnet_path]:
    os.makedirs(os.path.join(path, "train"), exist_ok=True)
    os.makedirs(os.path.join(path, "test"), exist_ok=True)
os.makedirs(real_test_path, exist_ok=True)  
all_images = sorted(os.listdir(raw_data_path)) 
random.shuffle(all_images) 
train_size = int(0.7 * len(all_images))  
test_size = int(0.2 * len(all_images))  
real_test_size = len(all_images) - train_size - test_size  
train_images = all_images[:train_size]
test_images = all_images[train_size:train_size + test_size]
real_test_images = all_images[train_size + test_size:]

def copy_images(images, dest_folder):
    for img in images:
        shutil.copy(os.path.join(raw_data_path, img), os.path.join(dest_folder, img))

copy_images(train_images, os.path.join(cnn_hog_path, "train"))
copy_images(train_images, os.path.join(efficientnet_path, "train"))

copy_images(test_images, os.path.join(cnn_hog_path, "test"))
copy_images(test_images, os.path.join(efficientnet_path, "test"))

copy_images(real_test_images, real_test_path)

print("Dataset successfully split into training, testing, and real test sets!")
