import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import random
from PIL import Image

DATA_DIR = "data/archive"
SAMPLE_IMG = 5
df = pd.read_csv("Artifacts/data_audit.csv")

plt.subplot(1, 3, 1)
sns.histplot(data=df, x="width")

plt.subplot(1, 3, 2)
sns.histplot(data=df, x="height")

plt.subplot(1, 3, 3)
sns.histplot(data=df, x="file size")

plt.savefig("Artifacts/eda_histograms.png")
plt.show()

classes = df['class'].unique()
fig, axes = plt.subplots(len(classes), SAMPLE_IMG, figsize=(9, 6))

for i, class_name in enumerate(classes):
    random_img = df[df['class'] == class_name]['file path'].sample(SAMPLE_IMG).values
    
    for j, img in enumerate(random_img):
        img = Image.open(img)
        ax = axes[i, j] if len(classes) > 1 else axes[j]
        ax.imshow(img)
        
        if j == 0:
            ax.set_ylabel(class_name)

plt.savefig("Artifacts/eda_sample_grid.png")
plt.show()
        