import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load and preprocess
image = cv2.imread('images/Dhingra ENT 8th Edition_Split_page-0014.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
binary_inv = cv2.bitwise_not(binary)

# Morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))
dilated = cv2.dilate(binary_inv, kernel, iterations=1)

# Find contours
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

min_width = 20
min_height = 10
all_boxes = []

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if w > min_width and h > min_height:
        all_boxes.append((x, y, w, h))

print(f"Found {len(all_boxes)} total boxes")

# GET IMAGE DIMENSIONS
height, width = image.shape[:2]
print(f"Image dimensions: {width}x{height}")

# ============================================
# IMPROVED FILTERING - Less Aggressive
# ============================================

text_boxes_only = []
filtered_boxes = []

for box in all_boxes:
    x, y, w, h = box
    aspect_ratio = w / h
    area = w * h
    
    # DEFAULT: Assume it's text unless proven otherwise
    is_image = False
    
    # RULE 1: Very wide boxes spanning most of page = likely images
    if w > width * 0.7:  # More than 70% of page width
        is_image = True
        reason = "too wide (spanning)"
    
    # RULE 2: Very tall and narrow = likely diagram element
    elif h > height * 0.4 and w < width * 0.2:  # Tall and thin
        is_image = True
        reason = "tall thin element"
    
    # RULE 3: Nearly square AND large = likely image
    elif 0.7 < aspect_ratio < 1.4 and area > (width * 0.4) * (height * 0.3):
        is_image = True
        reason = "large square region"
    
    # RULE 4: Check pixel density
    else:
        # Sample the region to see if it's mostly empty (image border)
        roi = binary[y:y+h, x:x+w]
        if roi.size > 0:
            white_pixels = np.sum(roi == 255)
            total_pixels = roi.size
            white_ratio = white_pixels / total_pixels
            
            # If more than 85% white AND large, it's likely an image border
            if white_ratio > 0.85 and area > (width * 0.3) * (height * 0.2):
                is_image = True
                reason = "mostly empty (image border)"
    
    if is_image:
        filtered_boxes.append((box, reason))
    else:
        text_boxes_only.append(box)

print(f"\nAfter filtering:")
print(f"Text boxes kept: {len(text_boxes_only)}")
print(f"Image boxes filtered: {len(filtered_boxes)}")

if len(filtered_boxes) > 0:
    print("\nFiltered boxes (first 10):")
    for i, (box, reason) in enumerate(filtered_boxes[:10]):
        x, y, w, h = box
        print(f"  Box: x={x}, y={y}, w={w}, h={h} - Reason: {reason}")

# ============================================
# COLUMN DETECTION ON TEXT BOXES
# ============================================

if len(text_boxes_only) < 5:
    print("\n⚠️ WARNING: Very few text boxes detected!")
    print("This filtering might be too aggressive. Proceeding anyway...")

if len(text_boxes_only) == 0:
    print("❌ ERROR: No text boxes found!")
    exit()

# Simple column detection: use middle 50% of page
column_boundary = width // 2

# # Try to detect gap
# text_centers = sorted([x + w//2 for (x, y, w, h) in text_boxes_only])
# max_gap = 0

# for i in range(1, len(text_centers)):
#     gap = text_centers[i] - text_centers[i-1]
#     if gap > max_gap and gap > width * 0.1:  # At least 10% of width
#         max_gap = gap
#         column_boundary = (text_centers[i] + text_centers[i-1]) // 2

# print(f"\nColumn boundary at X = {column_boundary}")

# ASSIGN TO COLUMNS
left_column = []
right_column = []

for box in text_boxes_only:
    x, y, w, h = box
    center_x = x + w // 2
    
    if center_x < column_boundary:
        left_column.append(box)
    else:
        right_column.append(box)

print(f"Left column: {len(left_column)} boxes")
print(f"Right column: {len(right_column)} boxes")

# SORT
left_column.sort(key=lambda box: box[1])
right_column.sort(key=lambda box: box[1])

text_boxes_sorted = left_column + right_column

# ============================================
# VISUALIZE
# ============================================

image_filtered = image.copy()
image_text_only = image.copy()

# Draw column boundary
cv2.line(image_filtered, (column_boundary, 0), (column_boundary, height), (255, 0, 255), 2)
cv2.line(image_text_only, (column_boundary, 0), (column_boundary, height), (255, 0, 255), 2)

# Draw filtered boxes in gray
for (box, reason) in filtered_boxes:
    x, y, w, h = box
    cv2.rectangle(image_filtered, (x, y), (x+w, y+h), (150, 150, 150), 2)

# Draw kept text boxes
for i, box in enumerate(text_boxes_sorted):
    x, y, w, h = box
    
    if i < len(left_column):
        color = (0, 255, 0)  # Green
    else:
        color = (0, 165, 255)  # Orange
    
    cv2.rectangle(image_filtered, (x, y), (x+w, y+h), color, 2)
    cv2.rectangle(image_text_only, (x, y), (x+w, y+h), color, 2)
    
    cv2.putText(image_filtered, str(i+1), (x+5, y+20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(image_text_only, str(i+1), (x+5, y+20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

# Display
fig, axes = plt.subplots(1, 3, figsize=(24, 12))

axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original Image', fontsize=12)
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(image_filtered, cv2.COLOR_BGR2RGB))
axes[1].set_title(f'All Boxes\nGray=Filtered ({len(filtered_boxes)}) | Colored=Text ({len(text_boxes_only)})', fontsize=12)
axes[1].axis('off')

axes[2].imshow(cv2.cvtColor(image_text_only, cv2.COLOR_BGR2RGB))
axes[2].set_title(f'TEXT ONLY ({len(text_boxes_sorted)} boxes)\nGreen=Left | Orange=Right', fontsize=12)
axes[2].axis('off')

plt.tight_layout()
plt.show()

print(f"\n=== SUMMARY ===")
print(f"Total boxes detected: {len(all_boxes)}")
print(f"Text boxes kept: {len(text_boxes_only)} ({len(text_boxes_only)/len(all_boxes)*100:.1f}%)")
print(f"Filtered as images: {len(filtered_boxes)} ({len(filtered_boxes)/len(all_boxes)*100:.1f}%)")