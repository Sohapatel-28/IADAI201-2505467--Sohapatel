import json
import os
import shutil

BASE = os.getcwd()

splits = ["train", "valid", "test"]

# YOLO dataset folders
for split in splits:
    os.makedirs(f"yolo_dataset/images/{split}", exist_ok=True)
    os.makedirs(f"yolo_dataset/labels/{split}", exist_ok=True)

for split in splits:
    folder = os.path.join(BASE, split)

    json_files = [
        f for f in os.listdir(folder)
        if f.endswith(".json")
    ]

    if not json_files:
        print(f"No JSON found in {split}")
        continue

    json_path = os.path.join(folder, json_files[0])

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # COCO category ID -> YOLO class ID
    categories = data["categories"]

    category_map = {}
    for index, category in enumerate(categories):
        category_map[category["id"]] = index

    images = {
        img["id"]: img
        for img in data["images"]
    }

    annotations_by_image = {}

    for ann in data["annotations"]:
        image_id = ann["image_id"]
        annotations_by_image.setdefault(image_id, []).append(ann)

    print(f"\nProcessing {split}...")
    print(f"Images: {len(images)}")
    print(f"Annotations: {len(data['annotations'])}")

    for image_id, image_info in images.items():

        filename = image_info["file_name"]
        width = image_info["width"]
        height = image_info["height"]

        source_image = os.path.join(folder, filename)

        if not os.path.exists(source_image):
            print("Missing image:", filename)
            continue

        # Copy image
        destination_image = os.path.join(
            BASE,
            "yolo_dataset",
            "images",
            split,
            filename
        )

        shutil.copy2(source_image, destination_image)

        # YOLO label filename
        label_name = os.path.splitext(filename)[0] + ".txt"

        label_path = os.path.join(
            BASE,
            "yolo_dataset",
            "labels",
            split,
            label_name
        )

        with open(label_path, "w", encoding="utf-8") as label_file:

            for ann in annotations_by_image.get(image_id, []):

                category_id = ann["category_id"]

                if category_id not in category_map:
                    continue

                class_id = category_map[category_id]

                x, y, w, h = ann["bbox"]

                # Convert COCO -> YOLO
                x_center = (x + w / 2) / width
                y_center = (y + h / 2) / height

                w_norm = w / width
                h_norm = h / height

                label_file.write(
                    f"{class_id} "
                    f"{x_center:.6f} "
                    f"{y_center:.6f} "
                    f"{w_norm:.6f} "
                    f"{h_norm:.6f}\n"
                )

    print(f"{split} conversion complete.")

print("\n================================")
print("COCO -> YOLO conversion DONE")
print("================================")