import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load and preprocess (same as before)
image = cv2.imread('images/Dhingra ENT 8th Edition_Split_page-0006.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
binary_inv = cv2.bitwise_not(binary)

# Morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))
dilated = cv2.dilate(binary_inv, kernel, iterations=1)

# Find and filter contours
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

min_width = 20
min_height = 10
bounding_boxes = []

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if w > min_width and h > min_height:
        bounding_boxes.append((x, y, w, h))

print(f"Found {len(bounding_boxes)} bounding boxes")

# GET IMAGE DIMENSIONS
height, width = image.shape[:2]
print(f"Image dimensions: {width}x{height}")

# SIMPLE APPROACH: Use middle of page as boundary
# But adjust based on actual content margins
left_edges = [x for (x, y, w, h) in bounding_boxes]
right_edges = [x + w for (x, y, w, h) in bounding_boxes]

# Find the leftmost and rightmost content
content_left = min(left_edges)
content_right = max(right_edges)
content_width = content_right - content_left

# Column boundary = middle of the content area
column_boundary = 642

# print(f"Content area: {content_left} to {content_right}")
# print(f"Column boundary at X = {column_boundary}")

# # ALTERNATIVE METHOD: Find the gap
# # Group boxes by their center X positions
# centers = sorted([(x + w//2, (x, y, w, h)) for (x, y, w, h) in bounding_boxes])

# # Find largest gap between consecutive centers
# max_gap = 0
# gap_position = width // 2

# for i in range(1, len(centers)):
#     gap = centers[i][0] - centers[i-1][0]
#     if gap > max_gap:
#         max_gap = gap
#         gap_position = (centers[i][0] + centers[i-1][0]) // 2

# # Use the gap method if gap is significant
# if max_gap > width * 0.1:  # Gap is at least 10% of page width
#     column_boundary = gap_position
#     print(f"Large gap detected ({max_gap} pixels), using gap position: {gap_position}")

# print(f"Final column boundary: {column_boundary}")

# ASSIGN BOXES TO COLUMNS based on their CENTER position
left_column_boxes = []
right_column_boxes = []

for box in bounding_boxes:
    x, y, w, h = box
    box_center_x = x + w // 2
    
    if box_center_x < column_boundary:
        left_column_boxes.append(box)
    else:
        right_column_boxes.append(box)

print(f"\nLeft column: {len(left_column_boxes)} boxes")
print(f"Right column: {len(right_column_boxes)} boxes")

# SORT EACH COLUMN BY Y-COORDINATE
left_column_boxes.sort(key=lambda box: box[1])
right_column_boxes.sort(key=lambda box: box[1])

# COMBINE
bounding_boxes_sorted = left_column_boxes + right_column_boxes

# VISUALIZE
image_with_order = image.copy()

# Draw column boundary
cv2.line(image_with_order, (column_boundary, 0), (column_boundary, height), (255, 0, 255), 3)

# Draw boxes with numbers
for i, (x, y, w, h) in enumerate(bounding_boxes_sorted):
    # Color by column
    if i < len(left_column_boxes):
        color = (0, 255, 0)  # Green for left
    else:
        color = (0, 165, 255)  # Orange for right
    
    cv2.rectangle(image_with_order, (x, y), (x+w, y+h), color, 2)
    
    # Draw number
    cv2.putText(
        image_with_order, 
        str(i+1),
        (x, y - 5),  # Inside the box, top-left
        cv2.FONT_HERSHEY_SIMPLEX,
        0.2,
        (255, 0, 0),
        2
    )

# Display
fig, axes = plt.subplots(1, 2, figsize=(20, 14))

axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original Image', fontsize=14)
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(image_with_order, cv2.COLOR_BGR2RGB))
axes[1].set_title(f'Column-Aware Sorting ({len(bounding_boxes_sorted)} boxes)\nGreen=Left Column | Orange=Right Column | Magenta=Boundary', fontsize=14)
axes[1].axis('off')

plt.tight_layout()
plt.show()

# Print sample boxes from each column
print("\n=== FIRST 5 BOXES IN EACH COLUMN ===")
print("\nLEFT COLUMN:")
for i, (x, y, w, h) in enumerate(left_column_boxes[:5]):
    print(f"  Box {i+1}: Position(x={x}, y={y}), Size(w={w}, h={h})")

print("\nRIGHT COLUMN:")
for i, (x, y, w, h) in enumerate(right_column_boxes[:5]):
    print(f"  Box {len(left_column_boxes)+i+1}: Position(x={x}, y={y}), Size(w={w}, h={h})")