import cv2
import numpy as np
import matplotlib.pyplot as plt

img  = cv2.imread("inria/data/train/images/austin1.tif")
mask = cv2.imread("inria/data/train/gt/austin1.tif", 0)

overlay = img.copy()
overlay[mask > 0] = [0, 255, 0]  # green = building pixels

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original image")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.title("Buildings highlighted green")

plt.savefig("sanity_check.png")
print("Saved sanity_check.png")