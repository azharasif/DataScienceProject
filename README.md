# DataScienceProject
Prescription Medicine Identification System

Overview

This project develops a robust system to accurately identify medicine names from prescription images, enhancing patient safety and reducing medication errors. By combining EasyOCR for text extraction and a ResNet18-based Convolutional Neural Network (CNN) for image classification, the system analyzes both textual and visual patterns in grayscale images. A voting mechanism ensures reliable detection across multilingual prescriptions. The system is trained on a synthetic dataset of 15,600 grayscale 256x256 images featuring 30 medicines, ethically addressing data scarcity. Validated against OpenFDA, this cost-effective and user-friendly solution supports low-quality images, making it ideal for small clinics and resource-constrained settings.

Features

Accurate Medicine Identification: Detects up to 30 medicine names from prescription images with high precision.
Robust Classification: Employs ResNet18-based CNN for visual pattern recognition, complemented by a voting mechanism for reliability.
Synthetic Dataset: Includes 15,600 grayscale 256x256 images (12,000 training, 1,800 validation, 1,800 testing) with 1–5 medicines per image.
Low-Quality Image Support: Handles noisy, rotated, or blurred prescriptions, suitable for real-world scenarios.
Cost-Effective: Designed for small clinics, streamlining verification and reducing errors.
Ethical Data Solution: Synthetic dataset avoids privacy concerns associated with real patient data.

Dataset
The system uses a synthetic dataset of 15,600 prescription images, generated to mimic real-world prescriptions:

Structure
Training: 12,000 images.
Validation: 1,800 images.
Testing: 1,800 images.
Labels: CSV files (training_labels.csv, validation_labels.csv, testing_labels.csv) with 30 binary columns for medicines (e.g., Aspirin, Paracetamol, ..., Insulin).
Medicines: 30 common medicines, including Aspirin, Paracetamol, Ibuprofen, Amoxicillin, Metformin, etc.
Distribution: Each image contains 1–3 medicines, with balanced representation (~800 occurrences per medicine in training).
The dataset is stored in synthetic_prescriptions_multilable/{training, validation, testing}/.

mainpage.py: Hanlde GUI for pictre upload and showing results
generateDataset: generate synthetic images using the set of 30 medicnes 
train_customCnn.py   traning of cnn model 
train_multilabel.py   training of resnet model



Requirements

Python 3.8+
PyTorch 2.0+
torchvision
Pillow
pandas
numpy
EasyOCR



matplotlib (for training visualizations)

Install dependencies:

pip install torch torchvision pillow pandas numpy easyocr matplotlib

Installation

Clone the repository:

git clone https://github.com/azharasif/DataScienceProject.git

