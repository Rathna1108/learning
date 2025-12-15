import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load and convert to grayscale
image = cv2.imread('images/Dhingra ENT 8th Edition_Split_page-0003.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Simple thresholding
# If pixel > 127, make it white (255), else black (0)
ret, binary_simple = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Adaptive thresholding (BETTER for uneven lighting)
# It calculates threshold for small regions separately
binary_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 12)

# Display comparison
fig, axes = plt.subplots(1, 1, figsize=(18, 6))

# axes[0].imshow(gray, cmap='gray')
# axes[0].set_title('Original Grayscale')
# axes[0].axis('off')

# axes[1].imshow(binary_simple, cmap='gray')
# axes[1].set_title('Simple Threshold')
# axes[1].axis('off')

# axes.imshow(binary_adaptive, cmap='gray')
# axes.set_title('Adaptive Threshold (Better!)')
# axes.axis('off')

axes.imshow(binary_adaptive, cmap='gray')
axes.set_title('Adaptive Threshold (Better!)')
axes.axis('off')

plt.tight_layout()
plt.show()