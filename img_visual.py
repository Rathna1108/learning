import cv2
import json
import random

with open("dataset_subset/annotations.json") as f:
    data = json.load(f)

for img in (data['images'][:2]):
    print(img['file_name'])
    img_info = img
    image = cv2.imread(f"dataset_subset/{img_info['file_name']}")

    for ann in data["annotations"]:
        if ann["image_id"] == img_info["id"]:
            x, y, w, h = map(int, ann["bbox"])
            cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0), 2)

    label_map = {
        0: "text",
        1: "title",
        2: "figure",
        3: "figure_caption",
        4: "pagenumber"
    }

    cv2.putText(
        image,
        label_map[ann["category_id"]],
        (x, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1
)


cv2.imshow("check", image)
cv2.waitKey(0)



