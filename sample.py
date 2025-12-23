import json
import random
from pathlib import Path
import shutil

ARCHIVE_DIR = "/Users/gecko/Downloads/archive"
IMAGE_DIR = Path(f"{ARCHIVE_DIR}/train-0/publaynet/train")
ANNOTATION_JSON = f"{ARCHIVE_DIR}/labels/publaynet/train.json"
TARGET_DIR = Path("small_dataset")
NUM_IMAGES = 50

TARGET_DIR.mkdir(exist_ok=True)

# List available image filenames
available_images = {p.name for p in IMAGE_DIR.glob("*.jpg")}

print("Images available on disk:", len(available_images))

with open(ANNOTATION_JSON) as f:
    data = json.load(f)

# Keep only images that exist on disk
filtered_images = [
    img for img in data["images"]
    if img["file_name"] in available_images
]

print("Images usable after filtering:", len(filtered_images))

NUM_IMAGES = min(NUM_IMAGES, len(filtered_images))

sampled_images = random.sample(filtered_images, NUM_IMAGES)
sampled_ids = {img["id"] for img in sampled_images}

subset_annotations = [
    ann for ann in data["annotations"]
    if ann["image_id"] in sampled_ids
]


print ('Images in filtered annotations', len(subset_annotations))

subset_data = {
    "images": sampled_images,
    "annotations": subset_annotations,
    "categories": data["categories"]
}

with open(TARGET_DIR / "annotations.json", "w") as f:
    json.dump(subset_data, f, indent=2)

for img in sampled_images:
    shutil.copy(
        IMAGE_DIR / img["file_name"],
        TARGET_DIR / img["file_name"]
    )

print("✅ Dataset prepared successfully")
