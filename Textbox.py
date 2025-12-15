import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load and preprocess
image = cv2.imread('images/Dhingra ENT 8th Edition_Split_page-0013.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11,2)
binary_inv = cv2.bitwise_not(binary)

# MORPHOLOGICAL OPERATIONS - Connect nearby text
# Create a rectangular kernel (structuring element)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))
# This means: 20 pixels wide, 3 pixels tall - good for connecting words in a line

# Dilate = expand white regions
dilated = cv2.dilate(binary_inv, kernel, iterations=1)

# Find contours on dilated image
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"After dilation, found {len(contours)} contours")

# Filter contours - remove very small ones (noise)
min_width = 20   # Minimum width in pixels
min_height = 10  # Minimum height in pixels

filtered_contours = []
bounding_boxes = []

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    
    # Keep only if larger than minimum size
    if w > min_width and h > min_height:
        filtered_contours.append(contour)
        bounding_boxes.append((x, y, w, h))

print(f"After filtering, {len(filtered_contours)} contours remain")

# Draw bounding boxes
image_with_boxes = image.copy()
for (x, y, w, h) in bounding_boxes:
    cv2.rectangle(image_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Display comparison
fig, axes = plt.subplots(1,2, figsize=(16, 16))

# axes[0, 0].imshow(binary_inv, cmap='gray')
# axes[0, 0].set_title('Binary Inverted (Original)')
# axes[0, 0].axis('off')

# axes[0, 1].imshow(dilated, cmap='gray')
# axes[0, 1].set_title('After Dilation (Text Connected)')
# axes[0, 1].axis('off')

axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))
axes[1].set_title(f'Text Regions: {len(filtered_contours)} boxes')
axes[1].axis('off')

plt.tight_layout()
plt.show()

# Print some bounding box info
print("\nFirst 10 bounding boxes (x, y, width, height):")
for i, (x, y, w, h) in enumerate(bounding_boxes[:10]):
    print(f"Box {i+1}: x={x}, y={y}, w={w}, h={h}")


