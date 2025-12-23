import json

# Your label IDs
YOUR_LABELS = {
    "plain_text": 0,
    "title": 1,
    "figure": 2,
    "figure_caption": 3,
    "page_number": 4
}

# PubLayNet category ID → name
PUBL_CATEGORIES = {
    1: "text",
    2: "title",
    3: "list",
    4: "table",
    5: "figure"
}

# Mapping rule
PUBL_TO_YOUR_LABEL = {
    "text": "plain_text",
    "list": "plain_text",
    "title": "title",
    "figure": "figure",
    "table": "page_number"
}

with open("dataset_subset/annotations.json") as f:
    coco = json.load(f)

new_annotations = []

for ann in coco["annotations"]:
    publ_cat_name = PUBL_CATEGORIES[ann["category_id"]]
    your_cat_name = PUBL_TO_YOUR_LABEL[publ_cat_name]
    your_cat_id = YOUR_LABELS[your_cat_name]

    ann["category_id"] = your_cat_id
    new_annotations.append(ann)

coco["annotations"] = new_annotations

# Replace categories with your labels
coco["categories"] = [
    {"id": v, "name": k} for k, v in YOUR_LABELS.items()
]

with open("dataset_subset/annotations_mapped.json", "w") as f:
    json.dump(coco, f, indent=2)

print("✅ Label mapping completed!")
