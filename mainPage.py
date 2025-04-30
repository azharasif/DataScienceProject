
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageEnhance
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import pandas as pd
import easyocr
import cv2
import requests 
import time     

data_dir = "synthetic_prescriptions_multiple"
resnet_model_path = "output_multilabel/best_model.pth"
cnn_model_path = "output_cnn_multilabel/best_model.pth"

medicine_names = [
    "Aspirin", "Paracetamol", "Ibuprofen", "Amoxicillin", "Ciprofloxacin", "Metformin", "Atorvastatin",
    "Lisinopril", "Levothyroxine", "Metoprolol", "Omeprazole", "Simvastatin", "Losartan", "Amlodipine",
    "Hydrochlorothiazide", "Gabapentin", "Prednisone", "Sertraline", "Fluoxetine", "Citalopram",
    "Albuterol", "Montelukast", "Cetirizine", "Ranitidine", "Doxycycline", "Azithromycin", "Tramadol",
    "Warfarin", "Clopidogrel", "Insulin"
]


class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 512),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class PredictionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Prescription Medicine Prediction")
        self.root.geometry("1600x900")  

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

        self.resnet_model = self.load_resnet_model()
        self.cnn_model = self.load_cnn_model()

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.image_path = None
        self.image_label = tk.Label(self.root, text="No image selected")
        self.image_label.pack(pady=10)

        self.canvas = tk.Canvas(self.root, width=250, height=250, bg="white")
        self.canvas.pack()

        self.browse_button = tk.Button(self.root, text="Browse Image", command=self.browse_image)
        self.browse_button.pack(pady=5)

        self.predict_button = tk.Button(self.root, text="Predict", command=self.predict, state=tk.DISABLED)
        self.predict_button.pack(pady=5)

        self.output_frame = tk.Frame(self.root)
        self.output_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        self.resnet_label = tk.Label(self.output_frame, text="ResNet18 Predictions")
        self.resnet_label.pack(side=tk.LEFT, padx=0)
        self.resnet_text = tk.Text(self.output_frame, height=10, width=20)
        self.resnet_text.pack(side=tk.LEFT, padx=0)

        self.cnn_label = tk.Label(self.output_frame, text="CNN Predictions")
        self.cnn_label.pack(side=tk.LEFT, padx=0)
        self.cnn_text = tk.Text(self.output_frame, height=10, width=20)
        self.cnn_text.pack(side=tk.LEFT, padx=0)

        self.ocr_label = tk.Label(self.output_frame, text="OCR Results")
        self.ocr_label.pack(side=tk.LEFT, padx=0)
        self.ocr_text = tk.Text(self.output_frame, height=10, width=20)
        self.ocr_text.pack(side=tk.LEFT, padx=0)

        self.final_label = tk.Label(self.output_frame, text="Final Results")
        self.final_label.pack(side=tk.LEFT, padx=0)
        self.final_text = tk.Text(self.output_frame, height=10, width=20)
        self.final_text.pack(side=tk.LEFT, padx=0)

        self.status_label = tk.Label(self.root, text="")
        self.status_label.pack(pady=10)


        self.details_label = tk.Label(self.root, text="Medicine Details FDA")
        self.details_label.pack(pady=5)

        self.details_text = tk.Text(self.root, height=70, width=120)
        self.details_text.pack(pady=7)

    def load_resnet_model(self):
        from torchvision.models import resnet18
        model = resnet18(weights=None)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.fc.in_features, len(medicine_names))
        )
        if os.path.exists(resnet_model_path):
            model.load_state_dict(torch.load(resnet_model_path))
            model = model.to(self.device)
            model.eval()
            return model
        else:
            raise FileNotFoundError(f"ResNet18 model not found at {resnet_model_path}")

    def load_cnn_model(self):
        model = CustomCNN(num_classes=len(medicine_names))
        if os.path.exists(cnn_model_path):
            model.load_state_dict(torch.load(cnn_model_path))
            model = model.to(self.device)
            model.eval()
            return model
        else:
            raise FileNotFoundError(f"Custom CNN model not found at {cnn_model_path}")

    def browse_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("PNG files", "*.png")])
        if self.image_path:
            try:
                image = Image.open(self.image_path).convert("L")
                image = image.resize((300, 300))
                self.photo = ImageTk.PhotoImage(image)
                self.canvas.create_image(150, 150, image=self.photo)
                self.image_label.config(text=f"Image: {os.path.basename(self.image_path)}")
                self.predict_button.config(state=tk.NORMAL)
                self.status_label.config(text="Image loaded")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {e}")
                self.image_path = None
                self.predict_button.config(state=tk.DISABLED)

    def preprocess_image_for_ocr(self, image_path):
        image = Image.open(image_path).convert("L")
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        image_np = np.array(image)
        image_np = cv2.resize(image_np, (512, 512), interpolation=cv2.INTER_CUBIC)
        image_np = cv2.adaptiveThreshold(
            image_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return image_np

    def extract_ocr_medicines(self, image_path):
        image_np = self.preprocess_image_for_ocr(image_path)
        results = self.ocr_reader.readtext(image_np, detail=1)
        ocr_medicines = []
        for (bbox, text, conf) in results:
            text = text.strip().lower()
            for med in medicine_names:
                if med.lower() in text and conf > 0.5:
                    ocr_medicines.append((med, conf))
        return list(set(ocr_medicines))

    def combine_predictions(self, resnet_preds, cnn_preds, ocr_medicines):
        all_preds = []
        for med, prob in resnet_preds:
            all_preds.append((med, prob, "ResNet"))
        for med, prob in cnn_preds:
            all_preds.append((med, prob, "CNN"))
        for med, conf in ocr_medicines:
            all_preds.append((med, conf, "OCR"))

        medicine_dict = {}
        for med, prob, source in all_preds:
            if med not in medicine_dict or prob > medicine_dict[med][0]:
                medicine_dict[med] = (prob, source)

        final_preds = []
        for med, (prob, source) in medicine_dict.items():
            if (source in ["ResNet", "CNN"] and prob > 0.5) or (source == "OCR" and prob > 0.8):
                final_preds.append((med, prob, source))
            elif source == "OCR" and prob > 0.5:
                for other_med, other_prob, other_source in all_preds:
                    if other_med == med and other_source in ["ResNet", "CNN"] and other_prob > 0.3:
                        if other_prob > prob:
                            final_preds.append((med, other_prob, other_source))
                        else:
                            final_preds.append((med, prob, source))
                        break

        final_preds.sort(key=lambda x: x[1], reverse=True)
        return final_preds

    def fetch_medicine_details(self, medicine_list):
        self.details_text.delete(1.0, tk.END)

        for med in medicine_list:
            try:
                url = f"https://api.fda.gov/drug/label.json?search=openfda.brand_name:\"{med}\"&limit=1"
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [{}])[0]

                    generic = results.get("openfda", {}).get("generic_name", ["Unknown"])[0]
                    pharm_class = results.get("openfda", {}).get("route", ["Unknown"])[0]
                    product_type = results.get("openfda" , {}).get("product_type" , [""])[0]
                    self.details_text.insert(tk.END, f"\nMedicine: {med}\n")
                    self.details_text.insert(tk.END, f"  Generic Name: {generic}\n")
                    self.details_text.insert(tk.END, f"  Type: {pharm_class}\n")
                    self.details_text.insert(tk.END, f"  Type: {product_type}\n")
                    self.details_text.insert(tk.END, "-"*50 + "\n")
                else:
                    self.details_text.insert(tk.END, f"\nMedicine: {med} - Details Not Available\n")
                    self.details_text.insert(tk.END, "-"*50 + "\n")
                time.sleep(0.5)  
            except Exception as e:
                self.details_text.insert(tk.END, f"\nError fetching details for {med}: {e}\n")
                self.details_text.insert(tk.END, "-"*50 + "\n")

    def predict(self):
        if not self.image_path:
            messagebox.showwarning("Warning", "Please select an image first")
            return

        try:
            image = Image.open(self.image_path).convert("L")
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                resnet_output = self.resnet_model(image_tensor)
                resnet_probs = torch.sigmoid(resnet_output).cpu().numpy()[0]

            with torch.no_grad():
                cnn_output = self.cnn_model(image_tensor)
                cnn_probs = torch.sigmoid(cnn_output).cpu().numpy()[0]

            ocr_medicines = self.extract_ocr_medicines(self.image_path)

            resnet_preds = [(medicine_names[i], prob) for i, prob in enumerate(resnet_probs) if prob > 0.5]
            cnn_preds = [(medicine_names[i], prob) for i, prob in enumerate(cnn_probs) if prob > 0.5]
            final_preds = self.combine_predictions(resnet_preds, cnn_preds, ocr_medicines)

            self.resnet_text.delete(1.0, tk.END)
            self.cnn_text.delete(1.0, tk.END)
            self.ocr_text.delete(1.0, tk.END)
            self.final_text.delete(1.0, tk.END)

            if resnet_preds:
                for med, prob in resnet_preds:
                    self.resnet_text.insert(tk.END, f"{med}: {prob:.4f}\n")
            else:
                self.resnet_text.insert(tk.END, "No medicines predicted (prob > 0.5)\n")

            if cnn_preds:
                for med, prob in cnn_preds:
                    self.cnn_text.insert(tk.END, f"{med}: {prob:.4f}\n")
            else:
                self.cnn_text.insert(tk.END, "No medicines predicted (prob > 0.5)\n")

            if ocr_medicines:
                for med, conf in ocr_medicines:
                    self.ocr_text.insert(tk.END, f"{med}: {conf:.4f}\n")
            else:
                self.ocr_text.insert(tk.END, "No medicines detected by OCR\n")

            if final_preds:
                medicine_names_final = []
                for med, prob, source in final_preds:
                    self.final_text.insert(tk.END, f"{med}: {prob:.4f} ({source})\n")
                    medicine_names_final.append(med)
                
                self.fetch_medicine_details(medicine_names_final)
            else:
                self.final_text.insert(tk.END, "No medicines selected\n")

            self.status_label.config(text="Predictions and Details fetched successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Prediction or OCR failed: {e}")

def main():
    root = tk.Tk()
    app = PredictionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
