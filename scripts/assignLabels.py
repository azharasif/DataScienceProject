import os
import pandas as pd

raw_data_path = "data/raw/"

all_images = [f for f in os.listdir(raw_data_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

df = pd.DataFrame({"filename": all_images, "label": 1}) 

df.to_csv("data/labels.csv", index=False)

print(" pre process images labeld")
