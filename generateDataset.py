import os
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import pandas as pd

random.seed(42)
np.random.seed(42)

medicine_names = [
    "Aspirin", "Paracetamol", "Ibuprofen", "Amoxicillin", "Ciprofloxacin", "Metformin", "Atorvastatin",
    "Lisinopril", "Levothyroxine", "Metoprolol", "Omeprazole", "Simvastatin", "Losartan", "Amlodipine",
    "Hydrochlorothiazide", "Gabapentin", "Prednisone", "Sertraline", "Fluoxetine", "Citalopram",
    "Albuterol", "Montelukast", "Cetirizine", "Ranitidine", "Doxycycline", "Azithromycin", "Tramadol",
    "Warfarin", "Clopidogrel", "Insulin"
]

data_dir = "synthetic_prescriptions_multiple"
os.makedirs(data_dir, exist_ok=True)
os.makedirs(os.path.join(data_dir, "training"), exist_ok=True)
os.makedirs(os.path.join(data_dir, "validation"), exist_ok=True)
os.makedirs(os.path.join(data_dir, "testing"), exist_ok=True)

font_dir = "fonts/"
handwritten_fonts = [
    os.path.join(font_dir, "Satisfy-Regular.ttf"),
    os.path.join(font_dir, "CedarvilleCursive-Regular.ttf"),
]

train_images = 12000  
val_images = 1800    
test_images = 1800    

image_size = (256, 256)  
bg_color_range = (240, 255) 
text_color_range = (0, 30)  

gaussian_noise_std_range = (0.5, 1.5)
salt_pepper_prob = 0.02
salt_pepper_ratio = 0.01

def add_noise(image):
    """Add minimal Gaussian and salt-and-pepper noise to the image."""
    img = np.array(image).astype(np.float32)
    
    std = random.uniform(*gaussian_noise_std_range)
    noise = np.random.normal(0, std, img.shape)
    img = img + noise
    
    if random.random() < salt_pepper_prob:
        num_pixels = int(salt_pepper_ratio * img.size)
        coords = [np.random.randint(0, i - 1, num_pixels) for i in img.shape]
        img[coords[0], coords[1]] = random.choice([0, 255])
    
    img = np.clip(img, 0, 255).astype(np.uint8)
    return Image.fromarray(img)

def get_text_size(draw, text, font):
    """Calculate text width and height using textbbox."""
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    width = right - left
    height = bottom - top
    return width, height

def generate_image(num_medicines, image_id, split):
    """Generate a single image with 1–3 medicine names."""
    image = Image.new("L", image_size, random.randint(*bg_color_range))
    draw = ImageDraw.Draw(image)
    
    font_path = random.choice(handwritten_fonts)
    
    medicines = random.sample(medicine_names, num_medicines)
    medicines.sort()  
    
    if num_medicines == 1:
        font_size = random.randint(24, 30)
    elif num_medicines == 2:
        font_size = random.randint(24, 30)
    else: 
        font_size = random.randint(20, 28)
    
    font = ImageFont.truetype(font_path, font_size)
    
    total_height = 0
    text_sizes = []
    for med in medicines:
        text = f"Tab {med}"
        width, height = get_text_size(draw, text, font)
        text_sizes.append((width, height))
        total_height += height
    
    spacing = 10 if num_medicines <= 2 else 15  
    total_height += spacing * (num_medicines - 1)
    
    y_start = (image_size[1] - total_height) // 2
    y = y_start
    
    for i, med in enumerate(medicines):
        text = f"Tab {med}"
        text_width, text_height = text_sizes[i]
        x = (image_size[0] - text_width) // 2 + random.randint(-20, 20)
        draw.text((x, y), text, fill=random.randint(*text_color_range), font=font)
        y += text_height + spacing
    
    image = add_noise(image)
    
    split_dir = os.path.join(data_dir, split)
    image_path = os.path.join(split_dir, f"{image_id}.png")
    image.save(image_path)
    
    return image_path, ",".join(medicines)

def generate_dataset(split, num_images, split_name):
    """Generate dataset for a given split (training, validation, testing)."""
    image_data = []
    
    num_singles = num_images // 3
    num_doubles = num_images // 3
    num_triples = num_images - num_singles - num_doubles 
    
    for i in range(num_images):
        if i < num_singles:
            num_meds = 1
        elif i < num_singles + num_doubles:
            num_meds = 2
        else:
            num_meds = 3
        
        image_path, medicines = generate_image(num_meds, i, split_name)
        image_data.append({"image_name": os.path.basename(image_path), "medicine_names": medicines})
    
    df = pd.DataFrame(image_data)
    csv_path = os.path.join(data_dir, f"{split_name}_labels.csv")
    df.to_csv(csv_path, index=False)
    
    return df

print("Generating training dataset...")
train_df = generate_dataset("training", train_images, "training")
print("Generating validation dataset...")
val_df = generate_dataset("validation", val_images, "validation")
print("Generating testing dataset...")
test_df = generate_dataset("testing", test_images, "testing")

def check_balance(df, split_name):
    medicine_counts = {med: 0 for med in medicine_names}
    for medicines in df["medicine_names"]:
        for med in medicines.split(","):
            medicine_counts[med] += 1
    print(f"\nBalance for {split_name}:")
    for med, count in medicine_counts.items():
        print(f"{med}: {count} appearances")
    avg_count = sum(medicine_counts.values()) / len(medicine_counts)
    print(f"Average appearances per medicine: {avg_count:.2f}")

check_balance(train_df, "training")
check_balance(val_df, "validation")
check_balance(test_df, "testing")

print("\nDataset generation complete!")