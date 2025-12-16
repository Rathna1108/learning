import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load and preprocess
# image = cv2.imread('textbook_images/page_003.jpg')
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

# SMART COLUMN DETECTION
# Strategy: Use only NARROW boxes (text blocks) to find column boundary
# Ignore WIDE boxes (images, diagrams that span both columns)

# Calculate page width thresholds
page_width = width
half_page = page_width * 0.5
max_column_width = page_width * 0.55  # Text column shouldn't be more than 55% of page

print(f'page width: {width}')
print(f'half page: {half_page}')
print(f'max column width: {max_column_width}')


# Filter: Keep only boxes that are NARROW (likely to be single-column text)
text_boxes = []
wide_boxes = []  # These are likely images/tables

for box in bounding_boxes:
    x, y, w, h = box
    if w < max_column_width:
        text_boxes.append(box)
    else:
        wide_boxes.append(box)

print(f"\nText boxes (narrow): {len(text_boxes)}")
print(f"Wide elements (images/tables): {len(wide_boxes)}")

# Now use ONLY text boxes to find column boundary
if len(text_boxes) > 0:
    # Get center X of each text box
    text_centers = sorted([x + w//2 for (x, y, w, h) in text_boxes])
    
    # Find the largest gap between consecutive text box centers
    max_gap = 0
    column_boundary = width // 2  # Default fallback
    
    for i in range(1, len(text_centers)):
        gap = text_centers[i] - text_centers[i-1]
        if gap > max_gap:
            max_gap = gap
            gap_center = (text_centers[i] + text_centers[i-1]) // 2
            # Only consider this as column boundary if gap is significant
            if gap > page_width * 0.08:  # At least 8% of page width
                column_boundary = gap_center

    print(f"Column boundary detected at X = {column_boundary}")
    print(f"Largest gap between text boxes: {max_gap} pixels ({max_gap/page_width*100:.1f}% of page width)")
else:
    column_boundary = width // 2
    print("No text boxes found, using page center")

# ASSIGN ALL BOXES (including wide ones) TO COLUMNS
left_column_boxes = []
right_column_boxes = []
spanning_boxes = []  # Boxes that span both columns

for box in bounding_boxes:
    x, y, w, h = box
    box_left = x
    box_right = x + w
    box_center = x + w // 2
    
    # If box is very wide and crosses the boundary, mark as spanning
    if w > max_column_width and box_left < column_boundary and box_right > column_boundary:
        spanning_boxes.append(box)
    # Otherwise assign based on center
    elif box_center < column_boundary:
        left_column_boxes.append(box)
    else:
        right_column_boxes.append(box)

print(f"\nLeft column: {len(left_column_boxes)} boxes")
print(f"Right column: {len(right_column_boxes)} boxes")
print(f"Spanning elements (images/tables): {len(spanning_boxes)} boxes")

# SORT EACH COLUMN BY Y-COORDINATE
left_column_boxes.sort(key=lambda box: box[1])
right_column_boxes.sort(key=lambda box: box[1])

# FOR SPANNING BOXES: Insert them in reading order based on Y position
# We'll merge them with the sorted columns later
all_boxes_with_column = []

for box in left_column_boxes:
    all_boxes_with_column.append((box, 'left'))

for box in right_column_boxes:
    all_boxes_with_column.append((box, 'right'))

for box in spanning_boxes:
    all_boxes_with_column.append((box, 'span'))

# Sort everything by Y position first
all_boxes_with_column.sort(key=lambda item: item[0][1])

# Now create final reading order: process top-to-bottom
# For each Y-level, read left column first, then right column
bounding_boxes_sorted = []
column_labels = []

# Group boxes by approximate Y position (within 50 pixels = same "row")
y_tolerance = 50
current_row = []
current_y = -1000

for box, col in all_boxes_with_column:
    x, y, w, h = box
    
    if abs(y - current_y) > y_tolerance:
        # Process previous row
        if current_row:
            # Sort row: left column first, then spanning, then right column
            row_sorted = sorted(current_row, key=lambda item: (
                0 if item[1] == 'left' else (1 if item[1] == 'span' else 2),
                item[0][0]  # Then by X position
            ))
            for b, c in row_sorted:
                bounding_boxes_sorted.append(b)
                column_labels.append(c)
        
        # Start new row
        current_row = [(box, col)]
        current_y = y
    else:
        current_row.append((box, col))

# Don't forget the last row
if current_row:
    row_sorted = sorted(current_row, key=lambda item: (
        0 if item[1] == 'left' else (1 if item[1] == 'span' else 2),
        item[0][0]
    ))
    for b, c in row_sorted:
        bounding_boxes_sorted.append(b)
        column_labels.append(c)

print(f"\nTotal sorted boxes: {len(bounding_boxes_sorted)}")

# VISUALIZE
image_with_order = image.copy()

# Draw column boundary
cv2.line(image_with_order, (column_boundary, 0), (column_boundary, height), (255, 0, 255), 2)

# Draw boxes with numbers and colors based on column
for i, (box, label) in enumerate(zip(bounding_boxes_sorted, column_labels)):
    x, y, w, h = box
    
    # Color by column
    if label == 'left':
        color = (0, 255, 0)  # Green
    elif label == 'right':
        color = (0, 165, 255)  # Orange
    else:  # spanning
        color = (255, 0, 255)  # Magenta
    
    cv2.rectangle(image_with_order, (x, y), (x+w, y+h), color, 2)
    
    # Draw number
    cv2.putText(
        image_with_order, 
        str(i+1),
        (x + 5, y + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),  # Red text
        2
    )

# Display
fig, axes = plt.subplots(1, 2, figsize=(20, 14))

axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original Image', fontsize=14)
axes[0].axis('on')

axes[1].imshow(cv2.cvtColor(image_with_order, cv2.COLOR_BGR2RGB))
axes[1].set_title(f'Smart Column Detection ({len(bounding_boxes_sorted)} boxes)\nGreen=Left | Orange=Right | Magenta=Spanning/Images', fontsize=14)
axes[1].axis('on')

plt.tight_layout()
plt.show()

# Print statistics
print("\n=== DETECTION SUMMARY ===")
print(f"Total boxes: {len(bounding_boxes_sorted)}")
print(f"Left column: {column_labels.count('left')}")
print(f"Right column: {column_labels.count('right')}")
print(f"Spanning (images/tables): {column_labels.count('span')}")