# Perform image enhancement, smoothing and sharpening, in spatial domain 
# using different spatial filters and compare the performances  

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read Image

img = cv2.imread(r"E:\Image-and-Speech-Lab-3.2\cameraman.jpg")

if img is None:
    print("Error: Image not found. Check the file path.")
    exit()

# Convert BGR to RGB for display
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 1. IMAGE ENHANCEMENT

# Histogram Equalization
hist_eq = cv2.equalizeHist(gray)

# CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

clahe_img = clahe.apply(gray)

# 2. IMAGE SMOOTHING

# Mean / Average Filter
mean_filter = cv2.blur(gray, (5, 5))

# Gaussian Filter
gaussian_filter = cv2.GaussianBlur(
    gray,
    (5, 5),
    0
)

# Median Filter
median_filter = cv2.medianBlur(
    gray,
    5
)

# 3. IMAGE SHARPENING

# ---------- Laplacian Sharpening ----------

laplacian = cv2.Laplacian(
    gray,
    cv2.CV_64F
)

laplacian = np.uint8(
    np.absolute(laplacian)
)

laplacian_sharp = cv2.add(
    gray,
    laplacian
)

# ---------- Unsharp Masking ----------

blurred = cv2.GaussianBlur(
    gray,
    (5, 5),
    0
)

# Unsharp masking
unsharp = cv2.addWeighted(
    gray,
    1.5,
    blurred,
    -0.5,
    0
)

# DISPLAY RESULTS

plt.figure(figsize=(14, 12))

# Original
plt.subplot(3, 3, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")

# Histogram Equalization
plt.subplot(3, 3, 2)
plt.imshow(hist_eq, cmap="gray")
plt.title("Histogram Equalization")
plt.axis("off")

# CLAHE
plt.subplot(3, 3, 3)
plt.imshow(clahe_img, cmap="gray")
plt.title("CLAHE")
plt.axis("off")

# Mean Filter
plt.subplot(3, 3, 4)
plt.imshow(mean_filter, cmap="gray")
plt.title("Mean Filter")
plt.axis("off")

# Gaussian Filter
plt.subplot(3, 3, 5)
plt.imshow(gaussian_filter, cmap="gray")
plt.title("Gaussian Filter")
plt.axis("off")

# Median Filter
plt.subplot(3, 3, 6)
plt.imshow(median_filter, cmap="gray")
plt.title("Median Filter")
plt.axis("off")

# Laplacian Sharpening
plt.subplot(3, 3, 7)
plt.imshow(laplacian_sharp, cmap="gray")
plt.title("Laplacian Sharpening")
plt.axis("off")

# Unsharp Masking
plt.subplot(3, 3, 8)
plt.imshow(unsharp, cmap="gray")
plt.title("Unsharp Masking")
plt.axis("off")

# Edge Image
plt.subplot(3, 3, 9)
plt.imshow(laplacian, cmap="gray")
plt.title("Laplacian Edge")
plt.axis("off")

plt.tight_layout()
plt.show()
