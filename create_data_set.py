# import os
# import csv
#
# images_dir = "data/images"
# labels_dir = "data/labels"
# csv_dir = "data/CSVs"
# csv_file = os.path.join(csv_dir, "dataset.csv")
#
# os.makedirs(csv_dir, exist_ok=True)
#
# img_exts = {".jpg", ".jpeg", ".png"}
#
# image_files = sorted(
#     f for f in os.listdir(images_dir)
#     if os.path.splitext(f.lower())[1] in img_exts
# )
#
# with open(csv_file, "w", newline="", encoding="utf-8") as f:
#     writer = csv.writer(f)
#     writer.writerow(["images", "labels"])
#
#     for img in image_files:
#         name, _ = os.path.splitext(img)
#         image_path = os.path.join(images_dir, img)
#         label_path = os.path.join(labels_dir, name + ".txt")
#
#         if os.path.exists(label_path):
#             writer.writerow([image_path, label_path])
#         else:
#             print(f"Warning: Missing label for {img}")
#
# print("dataset.csv created successfully!")
# print("Saved to:", csv_file)

import os
import csv
import random

images_dir = "data/images"
labels_dir = "data/labels"
csv_dir = "data/CSVs"

train_csv = os.path.join(csv_dir, "train_df.csv")
val_csv = os.path.join(csv_dir, "val_df.csv")

os.makedirs(csv_dir, exist_ok=True)

img_exts = {".jpg", ".jpeg", ".png"}

image_files = sorted(
    f for f in os.listdir(images_dir)
    if os.path.splitext(f.lower())[1] in img_exts
)

rows = []
for img in image_files:
    name, _ = os.path.splitext(img)
    image_path = os.path.join(images_dir, img)
    label_path = os.path.join(labels_dir, name + ".txt")

    if os.path.exists(label_path):
        rows.append([image_path, label_path])
    else:
        print(f"Warning: Missing label for {img}")

if not rows:
    raise RuntimeError("No valid (image, label) pairs found. Check data/images and data/labels.")

# shuffle to avoid biased split
random.seed(42)
random.shuffle(rows)

split_idx = int(0.8 * len(rows))
train_rows = rows[:split_idx]
val_rows = rows[split_idx:]

def write_csv(path, data):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["images", "labels"])
        writer.writerows(data)

write_csv(train_csv, train_rows)
write_csv(val_csv, val_rows)

print("CSV files created successfully!")
print("Train:", train_csv, f"({len(train_rows)} rows)")
print("Val:  ", val_csv, f"({len(val_rows)} rows)")
