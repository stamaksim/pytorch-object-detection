import os
import csv

images_dir = "data/images"
labels_dir = "data/labels"
csv_dir = "data/CSVs"
csv_file = os.path.join(csv_dir, "dataset.csv")

os.makedirs(csv_dir, exist_ok=True)

img_exts = {".jpg", ".jpeg", ".png"}

image_files = sorted(
    f for f in os.listdir(images_dir)
    if os.path.splitext(f.lower())[1] in img_exts
)

with open(csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["images", "labels"])

    for img in image_files:
        name, _ = os.path.splitext(img)
        image_path = os.path.join(images_dir, img)
        label_path = os.path.join(labels_dir, name + ".txt")

        if os.path.exists(label_path):
            writer.writerow([image_path, label_path])
        else:
            print(f"Warning: Missing label for {img}")

print("dataset.csv created successfully!")
print("Saved to:", csv_file)
