# Write a MATLAB/Python program to read an image and perform Lapliciant 
# filter mask. 

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
image = cv2.imread(r"E:\Image-and-Speech-Lab-3.2\cameraman.jpg", cv2.IMREAD_GRAYSCALE)

# Define 3x3 Laplacian filter mask
mask = np.array([
    [0, -1, 0],
    [-1, 4, -1],
    [0, -1, 0]
])

# Apply Laplacian filter
laplacian = cv2.filter2D(image, cv2.CV_64F, mask)

# Convert result to displayable image
laplacian = np.absolute(laplacian)
laplacian = np.uint8(np.clip(laplacian, 0, 255))

# Display results
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(laplacian, cmap="gray")
plt.title("Laplacian Filtered Image")
plt.axis("off")

plt.tight_layout()
plt.show()