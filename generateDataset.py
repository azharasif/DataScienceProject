import os
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import pandas as pd

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Medicine names
medicine_names = [
    "Aspirin", "Paracetamol", "Ibuprofen", "Amoxicillin", "Ciprofloxacin", "Metformin", "Atorvastatin",
    "Lisinopril", "Levothyroxine", "Metoprolol", "Omeprazole", "Simvastatin", "Losartan", "Amlodipine",
    "Hydrochlorothiazide", "Gabapentin", "Prednisone", "Sertraline", "Fluoxetine", "Citalopram",
    "Albuterol", "Montelukast", "Cetirizine", "Ranitidine", "Doxycycline", "Azithromycin", "Tramadol",
    "Warfarin", "Clopidogrel", "Insulin"
]

# Paths
data_dir = "synthetic_prescriptions_multiple"
os.makedirs(data_dir, exist_ok=True)
os.makedirs(os.path.join(data_dir, "training"), exist_ok=True)
os.makedirs(os.path.join(data_dir, "validation"), exist_ok=True)
os.makedirs(os.path.join(data_dir, "testing"), exist_ok=True)

# Fonts
font_dir = "fonts/"
handwritten_fonts = [
    os.path.join(font_dir, "Satisfy-Regular.ttf"),
    os.path.join(font_dir, "CedarvilleCursive-Regular.ttf"),
]

# Dataset sizes
train_images = 12000  # 4000 singles, 4000 doubles, 4000 triples
val_images = 1800     # 600 singles, 600 doubles, 600 triples
test_images = 1800    # 600 singles, 600 doubles, 600 triples

# Image parameters
image_size = (256, 256)  # 256x256 grayscale
bg_color_range = (240, 255)  # Off-white background
text_color_range = (0, 30)   # Dark text

# Noise parameters
gaussian_noise_std_range = (0.5, 1.5)
salt_pepper_prob = 0.02
salt_pepper_ratio = 0.01

def add_noise(image):
    """Add minimal Gaussian and salt-and-pepper noise to the image."""
    # Convert to numpy array
    img = np.array(image).astype(np.float32)
    
    # Gaussian noise
    std = random.uniform(*gaussian_noise_std_range)
    noise = np.random.normal(0, std, img.shape)
    img = img + noise
    
    # Salt-and-pepper noise
    if random.random() < salt_pepper_prob:
        num_pixels = int(salt_pepper_ratio * img.size)
        coords = [np.random.randint(0, i - 1, num_pixels) for i in img.shape]
        img[coords[0], coords[1]] = random.choice([0, 255])
    
    # Clip to [0, 255] and convert back to uint8
    img = np.clip(img, 0, 255).astype(np.uint8)
    return Image.fromarray(img)

def get_text_size(draw, text, font):
    """Calculate text width and height using textbbox."""
    # textbbox returns (left, top, right, bottom)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    width = right - left
    height = bottom - top
    return width, height

def generate_image(num_medicines, image_id, split):
    """Generate a single image with 1–3 medicine names."""
    # Create blank grayscale image
    image = Image.new("L", image_size, random.randint(*bg_color_range))
    draw = ImageDraw.Draw(image)
    
    # Select random font
    font_path = random.choice(handwritten_fonts)
    
    # Select number of medicines and their names
    medicines = random.sample(medicine_names, num_medicines)
    medicines.sort()  # Sort for consistent labeling
    
    # Adjust font size based on number of medicines
    if num_medicines == 1:
        font_size = random.randint(24, 30)
    elif num_medicines == 2:
        font_size = random.randint(24, 30)
    else:  # 3 medicines
        font_size = random.randint(20, 28)
    
    font = ImageFont.truetype(font_path, font_size)
    
    # Calculate text positions
    total_height = 0
    text_sizes = []
    for med in medicines:
        text = f"Tab {med}"
        width, height = get_text_size(draw, text, font)
        text_sizes.append((width, height))
        total_height += height
    
    spacing = 10 if num_medicines <= 2 else 15  # More spacing for triples
    total_height += spacing * (num_medicines - 1)
    
    # Start y-position to center text vertically
    y_start = (image_size[1] - total_height) // 2
    y = y_start
    
    # Draw each medicine name
    for i, med in enumerate(medicines):
        text = f"Tab {med}"
        text_width, text_height = text_sizes[i]
        # Random x-offset for realism, centered horizontally
        x = (image_size[0] - text_width) // 2 + random.randint(-20, 20)
        draw.text((x, y), text, fill=random.randint(*text_color_range), font=font)
        y += text_height + spacing
    
    # Add minimal noise
    image = add_noise(image)
    
    # Save image
    split_dir = os.path.join(data_dir, split)
    image_path = os.path.join(split_dir, f"{image_id}.png")
    image.save(image_path)
    
    return image_path, ",".join(medicines)

def generate_dataset(split, num_images, split_name):
    """Generate dataset for a given split (training, validation, testing)."""
    image_data = []
    
    # Calculate number of images per category
    num_singles = num_images // 3
    num_doubles = num_images // 3
    num_triples = num_images - num_singles - num_doubles  # Ensure total matches
    
    for i in range(num_images):
        if i < num_singles:
            num_meds = 1
        elif i < num_singles + num_doubles:
            num_meds = 2
        else:
            num_meds = 3
        
        image_path, medicines = generate_image(num_meds, i, split_name)
        image_data.append({"image_name": os.path.basename(image_path), "medicine_names": medicines})
    
    # Save labels to CSV
    df = pd.DataFrame(image_data)
    csv_path = os.path.join(data_dir, f"{split_name}_labels.csv")
    df.to_csv(csv_path, index=False)
    
    return df

# Generate datasets
print("Generating training dataset...")
train_df = generate_dataset("training", train_images, "training")
print("Generating validation dataset...")
val_df = generate_dataset("validation", val_images, "validation")
print("Generating testing dataset...")
test_df = generate_dataset("testing", test_images, "testing")

# Verify balance
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