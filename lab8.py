# Write a MATLAB/Python program to read coins.png, leveling all coins and 
# display area of all coins.

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread(r"E:\Image-and-Speech-Lab-3.2\coins.png")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blur image
gray = cv2.GaussianBlur(gray, (7, 7), 0)

# Convert to binary using Otsu thresholding
_, binary = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

# Find contours
contours, _ = cv2.findContours(
    binary,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# Copy image
result = img.copy()

coins = []

# Find coin contours
for contour in contours:

    # Calculate actual area of coin
    area = cv2.contourArea(contour)

    # Remove very small objects
    if area > 5000:

        # Find bounding circle
        (x, y), radius = cv2.minEnclosingCircle(contour)

        x = int(x)
        y = int(y)
        radius = int(radius)

        coins.append((x, y, radius, area))

# Sort coins from top to bottom and left to right
coins.sort(key=lambda c: (c[1], c[0]))

print("======================================")
print("          COIN MEASUREMENTS")
print("======================================")

print("Number of coins:", len(coins))
print()

print("Area of each coin:")
print("---------------------------")

# Display each coin
for i, (x, y, radius, area) in enumerate(coins, 1):

    # Draw circle around coin
    cv2.circle(
        result,
        (x, y),
        radius,
        (0, 255, 0),
        3
    )

    # Draw center
    cv2.circle(
        result,
        (x, y),
        5,
        (255, 0, 0),
        -1
    )

    # Display coin number and area
    text = f"Coin {i}: {area:.0f} px"

    cv2.putText(
        result,
        text,
        (x - 70, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2
    )

    print(f"Coin {i}: Area = {area:.2f} pixels")


# Convert BGR to RGB for displaying with Matplotlib
img_rgb = cv2.cvtColor(
    img,
    cv2.COLOR_BGR2RGB
)

result_rgb = cv2.cvtColor(
    result,
    cv2.COLOR_BGR2RGB
)

# Display results
plt.figure(figsize=(18, 6))

# Original image
plt.subplot(1, 3, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")

# Binary image
plt.subplot(1, 3, 2)
plt.imshow(binary, cmap="gray")
plt.title("Binary Image")
plt.axis("off")

# Detected coins with area
plt.subplot(1, 3, 3)
plt.imshow(result_rgb)
plt.title(f"Detected Coins (Total: {len(coins)})")
plt.axis("off")

plt.tight_layout()
plt.show()