import cv2
import numpy as np
import pytesseract
from PIL import Image

# -----------------------------
# 1. Load image
# -----------------------------
# image = cv2.imread("images/Dhingra ENT 8th Edition_Split_page-0015.jpg")
image = cv2.imread("textbook_images/page_004.jpg")

if image is None:
    raise ValueError("Image not found")

orig = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# -----------------------------
# 2. Binarization (text = white)
# -----------------------------
_, binary = cv2.threshold(
    gray, 0, 255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

# -----------------------------
# 3. Multi-kernel Connected Components
# -----------------------------
kernels = [
    (10, 5),   # small text
    (25, 10),  # subheadings
    (30, 15)   # paragraphs
]

all_boxes = []

for kw, kh in kernels:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        dilated, connectivity=8
    )

    for i in range(1, num_labels):  # skip background
        x, y, w, h, area = stats[i]
        if area > 300:
            all_boxes.append((x, y, w, h))

# -----------------------------
# 4. IOU function
# -----------------------------
def iou(a, b):
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[0]+a[2], b[0]+b[2])
    yB = min(a[1]+a[3], b[1]+b[3])
    if xB <= xA or yB <= yA:
        return 0.0
    inter = (xB-xA)*(yB-yA)
    areaA = a[2]*a[3]
    areaB = b[2]*b[3]
    return inter / float(areaA + areaB - inter)

# -----------------------------
# 5. De-duplicate boxes
# -----------------------------
all_boxes = sorted(all_boxes, key=lambda b: b[2]*b[3], reverse=True)

final_boxes = []
for box in all_boxes:
    if all(iou(box, fb) < 0.4 for fb in final_boxes):
        final_boxes.append(box)

# -----------------------------
# 6. Paragraph grouping helpers
# -----------------------------
def horizontal_overlap(a, b, ratio=0.5):
    x1 = max(a[0], b[0])
    x2 = min(a[0] + a[2], b[0] + b[2])
    overlap = max(0, x2 - x1)
    return overlap / min(a[2], b[2]) >= ratio

final_boxes = sorted(final_boxes, key=lambda b: b[1])

MAX_LINE_GAP = 25

# -----------------------------
# 7. Paragraph grouping
# -----------------------------
paragraphs = []
current = list(final_boxes[0])

for box in final_boxes[1:]:
    x, y, w, h = box
    cx, cy, cw, ch = current

    gap = y - (cy + ch)

    if 0 <= gap <= MAX_LINE_GAP and horizontal_overlap(current, box):
        nx = min(cx, x)
        ny = min(cy, y)
        nw = max(cx+cw, x+w) - nx
        nh = max(cy+ch, y+h) - ny
        current = [nx, ny, nw, nh]
    else:
        paragraphs.append(tuple(current))
        current = list(box)

paragraphs.append(tuple(current))


PAGE_HEIGHT, PAGE_WIDTH = orig.shape[:2]

blocks = []

for (x, y, w, h) in paragraphs:

    roi = orig[y:y+h, x:x+w]

    # OCR config
    config = "--oem 3 --psm 6"
    text = pytesseract.image_to_string(roi, config=config).strip()

    # ---------- Classification ----------
    block_type = "paragraph"

    # Page number (small, top/bottom)
    if h < 40 and w < 80 and (y < 80 or y > PAGE_HEIGHT - 80):
        if text.isdigit():
            block_type = "page_number"

    # Heading
    elif h < 80 and w > PAGE_WIDTH * 0.4:
        block_type = "heading"

    # Subheading
    elif h < 60 and w > PAGE_WIDTH * 0.25:
        block_type = "subheading"

    # List
    elif text.startswith(tuple(str(i) for i in range(1, 10))):
        block_type = "list"

    # Figure / Diagram
    elif h > 250 and w > PAGE_WIDTH * 0.4 and len(text) < 30:
        block_type = "figure"

    blocks.append({
        "type": block_type,
        "bbox": [int(x), int(y), int(w), int(h)],
        "text": text
    })

    
# -----------------------------
# 8. Draw paragraph boxes
# -----------------------------
for (x, y, w, h) in paragraphs:
    if h > 60:  # paragraph height filter
        cv2.rectangle(
            orig,
            (x, y),
            (x+w, y+h),
            (255, 0, 0),  # BLUE
            2
        )

# -----------------------------
# 9. Save output
# -----------------------------
cv2.imwrite("paragraph_blocks.png", orig)
print("Saved paragraph_blocks.png")
