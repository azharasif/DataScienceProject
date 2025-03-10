# import os
# import cv2
# import numpy as np
# from skimage.feature import hog
# from skimage import color
# from PIL import Image
# import pandas as pd
# def preprocess_images(input_folder, output_folder, labels_file, use_hog=False):
#     """Preprocess images (resize, grayscale, HOG if needed) and save."""

#     os.makedirs(output_folder, exist_ok=True)

#     # Load labels
#     df = pd.read_csv(labels_file)

#     for img_name in df["filename"]:
#         img_path = os.path.join(input_folder, img_name)
        
#         # Attempt to load the image using cv2 first
#         img = cv2.imread(img_path)
        
#         # If cv2 can't load the image, try PIL (Pillow)
#         if img is None:
#             try:
#                 img = np.array(Image.open(img_path).convert("RGB"))
#             except Exception as e:
#                 print(f"Warning: {img_path} could not be loaded due to error: {e}")
#                 continue  # Skip this image if it can't be loaded
        
#         img = cv2.resize(img, (224, 224))  # Resize to 224x224
#         img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

#         if use_hog:
#             # Convert to grayscale (required by HOG), and then extract HOG features
#             features, _ = hog(img_gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
#             # Save the HOG features as a numpy array file
#             np.save(os.path.join(output_folder, img_name.replace('.jpg', '.npy')), features)
#         else:
#             # If not using HOG, just save the grayscale image
#             cv2.imwrite(os.path.join(output_folder, img_name), img_gray)

# # Preprocess datasets for CNN + HOG
# preprocess_images("data/model_1_cnn_hog/train", "data/processed/cnn_hog/train", "data/model_1_cnn_hog/labels_train.csv", use_hog=True)
# preprocess_images("data/model_1_cnn_hog/test", "data/processed/cnn_hog/test", "data/model_1_cnn_hog/labels_test.csv", use_hog=True)

# # # Preprocess datasets for EfficientNet (without HOG)
# # preprocess_images("data/model_2_efficientnet/train", "data/processed/efficientnet/train", "data/model_2_efficientnet/labels_train.csv")
# # preprocess_images("data/model_2_efficientnet/test", "data/processed/efficientnet/test", "data/model_2_efficientnet/labels_test.csv")

