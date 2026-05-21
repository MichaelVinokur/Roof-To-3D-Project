import cv2
import matplotlib.pyplot as plt
import os

#Show first 6 crops in a grid
crops = sorted(os.listdir("data/crops/"))[:6]

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

for i, filename in enumerate(crops):
    img = cv2.imread(f"data/crops/{filename}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axes[i].imshow(img)
    axes[i].set_title(filename)
    axes[i].axis("off")

plt.tight_layout()
plt.savefig("crops_check.png")
print("Saved crops_check.png")