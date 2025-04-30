import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import multiprocessing
from tqdm import tqdm

# --------------------
# CONFIG
# --------------------
DATA_DIR = "synthetic_prescriptions_single_medicine"
OUTPUT_DIR = "output_single_medicine_training"
TRAIN_CSV = os.path.join(DATA_DIR, "training_labels.csv")
VAL_CSV = os.path.join(DATA_DIR, "validation_labels.csv")
IMG_SIZE = 128
NUM_CLASSES = 30
BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 0.001
NUM_WORKERS = 0  # Set to 0 to avoid multiprocessing issues
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EARLY_STOPPING_PATIENCE = 8  # Stop after 5 epochs without improvement
EARLY_STOPPING_DELTA = 0.001  # Minimum F1 improvement to reset patience

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------
# DATASET
# --------------------
class PrescriptionDataset(Dataset):
    def __init__(self, csv_file, data_dir, split, transform=None):
        try:
            self.df = pd.read_csv(csv_file)
            self.data_dir = os.path.join(data_dir, split)
            self.transform = transform
            # Map medicine names to class indices
            self.medicine_names = sorted(self.df['medicine_name'].unique())
            if len(self.medicine_names) != NUM_CLASSES:
                raise ValueError(f"Expected {NUM_CLASSES} medicines, found {len(self.medicine_names)}")
            self.label_map = {name: idx for idx, name in enumerate(self.medicine_names)}
        except Exception as e:
            raise RuntimeError(f"Failed to initialize dataset from {csv_file}: {str(e)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            img_name = self.df.iloc[idx]['image_name']
            img_path = os.path.join(self.data_dir, img_name)
            image = Image.open(img_path).convert('L')  # Grayscale
            label = self.label_map[self.df.iloc[idx]['medicine_name']]

            # Convert to 3 channels
            image = np.array(image) / 255.0  # Normalize to [0, 1]
            image = np.stack([image] * 3, axis=2)  # Shape: [128, 128, 3]
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # Shape: [3, 128, 128]

            if self.transform:
                image = self.transform(image)

            return image, label
        except Exception as e:
            raise RuntimeError(f"Failed to load sample {idx} from {img_path}: {str(e)}")

# Data transformations
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  # Normalizes to [0, 1]
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

def main():
    # Create datasets
    try:
        train_dataset = PrescriptionDataset(TRAIN_CSV, DATA_DIR, "training", transform=train_transform)
        val_dataset = PrescriptionDataset(VAL_CSV, DATA_DIR, "validation", transform=val_transform)
    except Exception as e:
        print(f"Error creating datasets: {str(e)}")
        exit(1)

    # Create data loaders
    try:
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    except Exception as e:
        print(f"Error creating data loaders: {str(e)}")
        exit(1)

    # --------------------
    # MODEL
    # --------------------
    try:
        model = models.resnet18(weights=None)
        # Add dropout to prevent overfitting
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.fc.in_features, NUM_CLASSES)
        )
        model = model.to(DEVICE)
    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        exit(1)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # --------------------
    # TRAINING LOOP
    # --------------------
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_f1s = []
    val_f1s = []

    best_f1 = 0.0
    patience_counter = 0
    best_model_path = os.path.join(OUTPUT_DIR, "best_model.pth")
    final_model_path = os.path.join(OUTPUT_DIR, "final_model.pth")

    try:
        for epoch in range(NUM_EPOCHS):
            print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
            print("-" * 50)

            # Training
            model.train()
            train_loss = 0.0
            train_preds = []
            train_labels = []

            # Progress bar for training
            train_bar = tqdm(train_loader, desc="Training", leave=False)
            for images, labels in train_bar:
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                train_preds.extend(preds.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())

                # Update progress bar
                train_bar.set_postfix(loss=loss.item())

            train_loss /= len(train_loader.dataset)
            train_acc = accuracy_score(train_labels, train_preds)
            train_f1 = f1_score(train_labels, train_preds, average='macro')

            # Validation
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_labels = []

            # Progress bar for validation
            val_bar = tqdm(val_loader, desc="Validation", leave=False)
            with torch.no_grad():
                for images, labels in val_bar:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * images.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

                    # Update progress bar
                    val_bar.set_postfix(loss=loss.item())

            val_loss /= len(val_loader.dataset)
            val_acc = accuracy_score(val_labels, val_preds)
            val_f1 = f1_score(val_labels, val_preds, average='macro')

            # Save metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            train_f1s.append(train_f1)
            val_f1s.append(val_f1)

            # Print metrics
            print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

            # Early stopping
            if val_f1 > best_f1 + EARLY_STOPPING_DELTA:
                best_f1 = val_f1
                patience_counter = 0
                print(f"New best F1 score: {best_f1:.4f}. Saving model to {best_model_path}")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_f1': best_f1
                }, best_model_path)
            else:
                patience_counter += 1
                print(f"No improvement in F1 score. Patience counter: {patience_counter}/{EARLY_STOPPING_PATIENCE}")

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

            # Step scheduler
            scheduler.step()

    except Exception as e:
        print(f"Error during training: {str(e)}")
        exit(1)

    # Save final model
    try:
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_f1': best_f1
        }, final_model_path)
    except Exception as e:
        print(f"Error saving final model: {str(e)}")
        exit(1)

    # --------------------
    # PLOT METRICS
    # --------------------
    try:
        epochs = range(1, len(train_losses) + 1)

        # Plot Loss
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, label='Training Loss')
        plt.plot(epochs, val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(OUTPUT_DIR, 'loss_plot.png'))
        plt.close()

        # Plot Accuracy
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_accuracies, label='Training Accuracy')
        plt.plot(epochs, val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_plot.png'))
        plt.close()

        # Plot F1 Score
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_f1s, label='Training F1 Score')
        plt.plot(epochs, val_f1s, label='Validation F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.title('Training and Validation F1 Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(OUTPUT_DIR, 'f1_plot.png'))
        plt.close()

    except Exception as e:
        print(f"Error generating plots: {str(e)}")
        exit(1)

    # Print completion
    print(f"\nTraining complete after {len(train_losses)} epochs")
    print(f"Best F1 score: {best_f1:.4f}")
    print(f"Best model saved to {best_model_path}")
    print(f"Final model saved to {final_model_path}")
    print(f"Plots saved to {OUTPUT_DIR}")

if __name__ == '__main__':
    # Set multiprocessing start method (for Windows compatibility)
    multiprocessing.set_start_method('spawn', force=True)
    main()