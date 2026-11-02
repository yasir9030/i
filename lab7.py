# For the given image perform edge detection using different operators and 
# compare the results.

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------
# 1. Read the image
# ---------------------------------------
image = cv2.imread(r"E:\Image-and-Speech-Lab-3.2\cameraman.jpg", cv2.IMREAD_GRAYSCALE)

# ---------------------------------------
# 2. Sobel Edge Detection
# ---------------------------------------
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

sobel = cv2.magnitude(sobel_x, sobel_y)
sobel = cv2.convertScaleAbs(sobel)

# ---------------------------------------
# 3. Prewitt Edge Detection
# ---------------------------------------
prewitt_x = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])

prewitt_y = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
])

px = cv2.filter2D(image, cv2.CV_64F, prewitt_x)
py = cv2.filter2D(image, cv2.CV_64F, prewitt_y)

prewitt = cv2.magnitude(px, py)
prewitt = cv2.convertScaleAbs(prewitt)

# ---------------------------------------
# 4. Roberts Edge Detection
# ---------------------------------------
roberts_x = np.array([
    [1, 0],
    [0, -1]
])

roberts_y = np.array([
    [0, 1],
    [-1, 0]
])

rx = cv2.filter2D(image, cv2.CV_64F, roberts_x)
ry = cv2.filter2D(image, cv2.CV_64F, roberts_y)

roberts = cv2.magnitude(rx, ry)
roberts = cv2.convertScaleAbs(roberts)

# ---------------------------------------
# 5. Canny Edge Detection
# ---------------------------------------
canny = cv2.Canny(image, 100, 200)

# ---------------------------------------
# 6. Display and Compare
# ---------------------------------------
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(sobel, cmap="gray")
plt.title("Sobel Operator")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(prewitt, cmap="gray")
plt.title("Prewitt Operator")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(roberts, cmap="gray")
plt.title("Roberts Operator")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(canny, cmap="gray")
plt.title("Canny Edge Detection")
plt.axis("off")

plt.tight_layout()
plt.show()

