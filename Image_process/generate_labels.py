import os
import pandas as pd
imageFolder = "data/raw/"
labels_file = "data/labels.csv"


image_files = sorted([filename for filename in os.listdir(imageFolder) if filename.endswith(('.jpg', '.png'))])


if not image_files:
  print("No image found")
labels = ["valid" if i % 2 == 0 else "invalid" for i in range(len(image_files))]

df = pd.DataFrame({"filename": image_files, "label": labels})
df.to_csv(labels_file, index=False)

print('Labels added to labels.csv')
