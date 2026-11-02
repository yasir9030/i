# Display following image operation in MATLAB/Python - i) Histogram  
# image  ii) Low pass filter mage  iii) High pass image. 

import cv2
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------
# Read Image
# --------------------------------

img = cv2.imread( r"E:\Image-and-Speech-Lab-3.2\cameraman.jpg")

# Check if image loaded properly
if img is None:
    print("Error: Image not found. Check the file path/name.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --------------------------------
# i) Histogram
# --------------------------------

hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

# --------------------------------
# ii) Low Pass Filter
# --------------------------------

low_pass = cv2.GaussianBlur(gray, (5, 5), 0)

# --------------------------------
# iii) High Pass Filter
# --------------------------------

kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])

high_pass = cv2.filter2D(gray, -1, kernel)

# --------------------------------
# Display Results
# --------------------------------

plt.figure(figsize=(12, 8))

# Original Image
plt.subplot(2, 2, 1)
plt.imshow(gray, cmap='gray')
plt.title('Original Grayscale Image')
plt.axis('off')

# Histogram
plt.subplot(2, 2, 2)
plt.plot(hist)
plt.title('Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

# Low Pass Filter
plt.subplot(2, 2, 3)
plt.imshow(low_pass, cmap='gray')
plt.title('Low Pass Filtered Image')
plt.axis('off')

# High Pass Filter
plt.subplot(2, 2, 4)
plt.imshow(high_pass, cmap='gray')
plt.title('High Pass Filtered Image')
plt.axis('off')

plt.tight_layout()
plt.show()