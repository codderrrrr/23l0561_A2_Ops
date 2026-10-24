import os
import hashlib
from PIL import Image
import pandas as pd
from collections import Counter

DATA_DIR = "data/archive/"

img_hashes = []
corrupt_files = 0
duplicates = 0
data = []

for class_name in os.listdir(DATA_DIR):
    class_path = os.path.join(DATA_DIR, class_name)
    for img in os.listdir(class_path):
        img_path = os.path.join(class_path, img)
        img_size = os.path.getsize(img_path)
        with open(img_path, "rb") as f:
            img_hash = hashlib.file_digest(f, "sha256").hexdigest()
            img_hashes.append(img_hash)

        try:
            with Image.open(img_path) as img:
                width, height = img.size
                color_mode = img.mode
                format = img.format
        except Exception as e:
            corrupt_files += 1
            width, height, color_mode, img_format = None, None, None, None

        data.append({
            "class" : class_name,
            "file name": img,
            "file path": img_path,
            "file size": img_size,
            "file hash": img_hash,
            "width": width,
            "height": height,
            "color mode": color_mode,
            "format": format,
        })
        
counts = Counter(img_hashes)
duplicates = {item: count for item, count in counts.items() if count > 1}

df = pd.DataFrame(data)
df.to_csv("Artifacts/data_audit.csv")

print(f"Total Corrupt/Unreadable Files: {corrupt_files}")
print(f"Total Exact Duplicates Found: {duplicates}")
print("\nCounts per class:")
print(df['class'].value_counts())
print("\nStatistical breakdown of dimensions and file sizes:")
print(df[['file size', 'width', 'height']].describe())