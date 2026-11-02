# Write a MATLAB/Python program to Character Segment of an image.

import cv2
import matplotlib.pyplot as plt

# -----------------------------------
# 1. Read image
# -----------------------------------
image = cv2.imread("image6.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Image not found!")
    exit()

# Display original image
plt.figure(figsize=(8, 4))
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")
plt.show()

# -----------------------------------
# 2. Convert image to binary
# -----------------------------------
_, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# If background is white, invert so characters become white
if binary[0, 0] == 255:
    binary = cv2.bitwise_not(binary)

# Display binary image
plt.figure(figsize=(8, 4))
plt.imshow(binary, cmap="gray")
plt.title("Binary Image")
plt.axis("off")
plt.show()

# -----------------------------------
# 3. Find connected components
# -----------------------------------
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

# -----------------------------------
# 4. Extract individual characters
# -----------------------------------
characters = []

for i in range(1, num_labels):
    x, y, w, h, area = stats[i]

    # Filter out small noise
    if area > 50:
        character = binary[y:y+h, x:x+w]
        characters.append((x, character))

# -----------------------------------
# 5. Sort characters from left to right
# -----------------------------------
characters.sort(key=lambda c: c[0])
print("Number of segmented characters:", len(characters))

# -----------------------------------
# 6. Display segmented characters
# -----------------------------------
plt.figure(figsize=(10, 3))

for i, (x, character) in enumerate(characters):
    plt.subplot(1, len(characters), i + 1)
    plt.imshow(character, cmap="gray")
    plt.title(f"Char {i + 1}")
    plt.axis("off")

plt.tight_layout()
plt.show()
