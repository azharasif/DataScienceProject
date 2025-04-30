import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from PIL import Image
import random
from tqdm import tqdm

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

data_dir = "synthetic_prescriptions_multiple"
output_dir = "output_Resnet_multilabel"

medicine_names = [
    "Aspirin", "Paracetamol", "Ibuprofen", "Amoxicillin", "Ciprofloxacin", "Metformin", "Atorvastatin",
    "Lisinopril", "Levothyroxine", "Metoprolol", "Omeprazole", "Simvastatin", "Losartan", "Amlodipine",
    "Hydrochlorothiazide", "Gabapentin", "Prednisone", "Sertraline", "Fluoxetine", "Citalopram",
    "Albuterol", "Montelukast", "Cetirizine", "Ranitidine", "Doxycycline", "Azithromycin", "Tramadol",
    "Warfarin", "Clopidogrel", "Insulin"
]

class PrescriptionDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        self.num_classes = len(medicine_names)
        self.label_map = {name: idx for idx, name in enumerate(medicine_names)}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]["image_name"]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("L")  
        
        medicines = self.data.iloc[idx]["medicine_names"].split(",")
        labels = torch.zeros(self.num_classes)
        for med in medicines:
            if med in self.label_map:
                labels[self.label_map[med]] = 1
        
        if self.transform:
            image = self.transform(image)
        
        return image, labels

train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomRotation(degrees=5),
    transforms.RandomAffine(degrees=0, shear=0.1),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.05, contrast=0.05)], p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  
])

val_test_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

class EarlyStopping:
    def __init__(self, patience=5, delta=0, path=os.path.join(output_dir, "best_model.pth")):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_loss = np.Inf
        self.path = path
    
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            torch.save(model.state_dict(), self.path)

def main():
    os.makedirs(output_dir, exist_ok=True)
    
    train_dataset = PrescriptionDataset(
        csv_file=os.path.join(data_dir, "training_labels.csv"),
        image_dir=os.path.join(data_dir, "training"),
        transform=train_transforms
    )
    val_dataset = PrescriptionDataset(
        csv_file=os.path.join(data_dir, "validation_labels.csv"),
        image_dir=os.path.join(data_dir, "validation"),
        transform=val_test_transforms
    )
    test_dataset = PrescriptionDataset(
        csv_file=os.path.join(data_dir, "testing_labels.csv"),
        image_dir=os.path.join(data_dir, "testing"),
        transform=val_test_transforms
    )
    
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, len(medicine_names))
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    num_epochs = 6
    early_stopping = EarlyStopping(patience=5, delta=0.001)
    
    train_losses, val_losses = [], []
    train_f1s, val_f1s = [], []
    train_accuracies, val_accuracies = [], []
    train_precisions, val_precisions = [], []
    train_recalls, val_recalls = [], []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds, all_labels = [], []
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds == labels).all(dim=1).sum().item()
            total += images.size(0)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            train_bar.set_postfix({"Train Loss": loss.item()})
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = correct / total
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        epoch_f1 = f1_score(all_labels, all_preds, average="macro")
        epoch_precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        epoch_recall = recall_score(all_labels, all_labels, average="macro", zero_division=0)
        
        train_losses.append(epoch_loss)
        train_f1s.append(epoch_f1)
        train_accuracies.append(epoch_accuracy)
        train_precisions.append(epoch_precision)
        train_recalls.append(epoch_recall)
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                preds = torch.sigmoid(outputs) > 0.5
                val_correct += (preds == labels).all(dim=1).sum().item()
                val_total += images.size(0)
                val_preds.append(preds.cpu().numpy())
                val_labels.append(labels.cpu().numpy())
        
        val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = val_correct / val_total
        val_preds = np.vstack(val_preds)
        val_labels = np.vstack(val_labels)
        val_f1 = f1_score(val_labels, val_preds, average="macro")
        val_priority = precision_score(val_labels, val_preds, average="macro", zero_division=0)
        val_recall = recall_score(val_labels, val_preds, average="macro", zero_division=0)
        
        val_losses.append(val_loss)
        val_f1s.append(val_f1)
        val_accuracies.append(val_accuracy)
        val_recalls.append(val_recall)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train F1: {epoch_f1:.4f}, Val F1: {val_f1:.4f}, Train Acc: {epoch_accuracy:.4f}, Val Acc: {val_accuracy:.4f}, "
              f"Train Precision: {epoch_precision:.4f}, Train Recall: {epoch_recall:.4f}, Val Recall: {val_recall:.4f}")
        
        metrics_df = pd.DataFrame({
            "Epoch": list(range(1, len(train_losses) + 1)),
            "Train Loss": train_losses,
            "Val Loss": val_losses,
            "Train F1": train_f1s,
            "Val F1": val_f1s,
            "Train Acc": train_accuracies,
            "Val Acc": val_accuracies,
            "Train Precision": train_precisions,
            "Train Recall": train_recalls,
            "Val Recall": val_recalls
        })
        metrics_df.to_csv(os.path.join(output_dir, "epoch_results.csv"), index=False)
        
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
        
        scheduler.step()
    
    model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pth")))
    
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    test_preds, test_labels = [], []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            preds = torch.sigmoid(outputs) > 0.5
            test_correct += (preds == labels).all(dim=1).sum().item()
            test_total += images.size(0)
            test_preds.append(preds.cpu().numpy())
            test_labels.append(labels.cpu().numpy())
    
    test_loss = test_loss / len(test_loader.dataset)
    test_accuracy = test_correct / test_total
    test_preds = np.vstack(test_preds)
    test_labels = np.vstack(test_labels)
    test_f1 = f1_score(test_labels, test_preds, average="macro")
    test_precision = precision_score(test_labels, test_preds, average="macro", zero_division=0)
    test_recall = recall_score(test_labels, test_preds, average="macro", zero_division=0)
    
    print(f"\nTest Results: Loss: {test_loss:.4f}, F1: {test_f1:.4f}, Precision: {test_precision:.4f}, "
          f"Recall: {test_recall:.4f}, Accuracy: {test_accuracy:.4f}")
    
    torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pth"))
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "loss_plot.png"))
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_f1s, label="Train Macro F1")
    plt.plot(epochs, val_f1s, label="Validation Macro F1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1 Score")
    plt.title("Training and Validation Macro F1 Score")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "f1_plot.png"))
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.plot(epochs, val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "accuracy_plot.png"))
    plt.close()
    
  
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_recalls, label="Train Recall")
    plt.plot(epochs, val_recalls, label="Validation Recall")
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.title("Training and Validation Recall")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "recall_plot.png"))
    plt.close()
    
    print(f"All outputs (CSV, plots, models) saved in {output_dir}")

if __name__ == '__main__':
    main()
