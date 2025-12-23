import cv2
import numpy as np
import pytesseract
import json
from PIL import Image
import re

# ===============================
# CONFIG
# ===============================
IMAGE_PATH = "images/Dhingra ENT 8th Edition_Split_page-0007.jpg"

# Uncomment & update if on Windows
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

KERNELS = [(10, 5), (25, 10), (30, 15)]
AREA_THRESHOLD = 300
IOU_THRESHOLD = 0.4
MAX_LINE_GAP = 25
MIN_PARAGRAPH_HEIGHT = 60

# ===============================
# 1. LOAD IMAGE
# ===============================
image = cv2.imread(IMAGE_PATH)
if image is None:
    raise ValueError("Image not found")

orig = image.copy()
PAGE_H, PAGE_W = image.shape[:2]

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ===============================
# 2. BINARIZATION
# ===============================
_, binary = cv2.threshold(
    gray, 0, 255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

# ===============================
# 3. MULTI-KERNEL CONNECTED COMPONENTS
# ===============================
all_boxes = []

for kw, kh in KERNELS:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        dilated, connectivity=8
    )

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area > AREA_THRESHOLD:
            all_boxes.append((x, y, w, h))

# ===============================
# 4. IOU FUNCTION
# ===============================
def iou(a, b):
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[0] + a[2], b[0] + b[2])
    yB = min(a[1] + a[3], b[1] + b[3])

    if xB <= xA or yB <= yA:
        return 0.0

    inter = (xB - xA) * (yB - yA)
    areaA = a[2] * a[3]
    areaB = b[2] * b[3]
    return inter / float(areaA + areaB - inter)

# ===============================
# 5. DE-DUPLICATION
# ===============================
all_boxes = sorted(all_boxes, key=lambda b: b[2] * b[3], reverse=True)

final_boxes = []
for box in all_boxes:
    if all(iou(box, fb) < IOU_THRESHOLD for fb in final_boxes):
        final_boxes.append(box)

# ===============================
# 6. PARAGRAPH GROUPING
# ===============================
def horizontal_overlap(a, b, ratio=0.5):
    x1 = max(a[0], b[0])
    x2 = min(a[0] + a[2], b[0] + b[2])
    overlap = max(0, x2 - x1)
    return overlap / min(a[2], b[2]) >= ratio

final_boxes = sorted(final_boxes, key=lambda b: b[1])

paragraphs = []
current = list(final_boxes[0])

for box in final_boxes[1:]:
    x, y, w, h = box
    cx, cy, cw, ch = current

    gap = y - (cy + ch)

    if 0 <= gap <= MAX_LINE_GAP and horizontal_overlap(current, box):
        nx = min(cx, x)
        ny = min(cy, y)
        nw = max(cx + cw, x + w) - nx
        nh = max(cy + ch, y + h) - ny
        current = [nx, ny, nw, nh]
    else:
        paragraphs.append(tuple(current))
        current = list(box)

paragraphs.append(tuple(current))

# ===============================
# 7. DRAW PARAGRAPH BLOCKS
# ===============================
para_vis = orig.copy()
for (x, y, w, h) in paragraphs:
    if h >= MIN_PARAGRAPH_HEIGHT:
        cv2.rectangle(para_vis, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv2.imwrite("paragraph_blocks.png", para_vis)

# ===============================
# 8. BLOCK CLASSIFICATION + OCR
# ===============================
blocks = []

for (x, y, w, h) in paragraphs:
    roi = orig[y:y+h, x:x+w]

    text = pytesseract.image_to_string(
        roi, config="--oem 3 --psm 6"
    ).strip()


    def is_page_number(text):
        text = text.strip()
        text = re.sub(r"[^\d]", "", text)
        return len(text) > 0 and len(text) <= 3

        block_type = "paragraph"

    if ((y < PAGE_H * 0.15 or y > PAGE_H * 0.85) and
        w < PAGE_W * 0.15 and
        h < 80 and
        is_page_number(text)):
            block_type = "page_number"
    elif h < 80 and w > PAGE_W * 0.45:
            block_type = "heading"
    elif h < 60 and w > PAGE_W * 0.25:
            block_type = "subheading"
    elif text.startswith(tuple(str(i) for i in range(1, 10))):
            block_type = "list"
    elif h > 250 and w > PAGE_W * 0.4 and len(text) < 40:
            block_type = "figure"

    blocks.append({
            "type": block_type,
            "bbox": [int(x), int(y), int(w), int(h)],
            "text": text
        })

# ===============================
# 9. VISUALIZE CLASSIFIED BLOCKS
# ===============================
colors = {
    "page_number": (0, 255, 255),
    "heading": (0, 0, 255),
    "subheading": (255, 0, 0),
    "paragraph": (0, 255, 0),
    "list": (255, 255, 0),
    "figure": (128, 0, 128)
}

cls_vis = orig.copy()
for b in blocks:
    x, y, w, h = b["bbox"]
    color = colors.get(b["type"], (255, 255, 255))
    cv2.rectangle(cls_vis, (x, y), (x + w, y + h), color, 2)
    cv2.putText(
        cls_vis, b["type"],
        (x, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5, color, 1
    )

cv2.imwrite("classified_blocks.png", cls_vis)

# ===============================
# 10. SAVE OUTPUTS
# ===============================
with open("output.json", "w", encoding="utf-8") as f:
    json.dump(blocks, f, indent=2, ensure_ascii=False)

with open("output.txt", "w", encoding="utf-8") as f:
    for b in blocks:
        f.write(f"[{b['type'].upper()}]\n")
        f.write(b["text"] + "\n\n")

print("✅ Done!")
print("Generated:")
print(" - paragraph_blocks.png")
print(" - classified_blocks.png")
print(" - output.json")
print(" - output.txt")
