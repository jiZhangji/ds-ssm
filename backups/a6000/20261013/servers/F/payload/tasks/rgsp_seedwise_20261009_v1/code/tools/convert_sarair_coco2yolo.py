import json
import os
import shutil
import argparse
from pathlib import Path
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--src", required=True, help="Path to raw SAR_AIRcraft directory")
parser.add_argument("--dst", default="datasets/SAR_AIRcraft_YOLO", help="Output YOLO-format dataset directory")
parser.add_argument("--copy", action="store_true", help="Copy images instead of creating symlinks")
args = parser.parse_args()

src = Path(args.src)
dst = Path(args.dst)
use_symlink = not args.copy

splits = {
    "train": "train.json",
    "val": "val.json",
    "test": "test.json",
}

dst.mkdir(parents=True, exist_ok=True)

# 统一类别映射，以 train.json 里的 categories 为准
train_json = src / "annotations" / "train.json"
train_data = json.load(open(train_json, "r"))
categories = sorted(train_data["categories"], key=lambda x: x["id"])

cat_id_to_yolo = {cat["id"]: i for i, cat in enumerate(categories)}
names = {i: cat["name"] for i, cat in enumerate(categories)}

print("类别映射:")
for old_id, new_id in cat_id_to_yolo.items():
    cat_name = [c["name"] for c in categories if c["id"] == old_id][0]
    print(f"COCO id {old_id} -> YOLO id {new_id}: {cat_name}")

for split, json_name in splits.items():
    json_path = src / "annotations" / json_name
    data = json.load(open(json_path, "r"))

    img_dir = dst / "images" / split
    label_dir = dst / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    images = {img["id"]: img for img in data["images"]}

    anns_by_img = defaultdict(list)
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            continue
        anns_by_img[ann["image_id"]].append(ann)

    for img_id, img_info in images.items():
        file_name = img_info["file_name"]
        src_img = src / "images" / file_name
        if not src_img.exists():
            print(f"[WARN] image not found: {src_img}")
            continue

        dst_img = img_dir / file_name

        if not dst_img.exists():
            if use_symlink:
                os.symlink(src_img, dst_img)
            else:
                shutil.copy2(src_img, dst_img)

        w_img = float(img_info["width"])
        h_img = float(img_info["height"])

        label_lines = []
        for ann in anns_by_img.get(img_id, []):
            category_id = ann["category_id"]
            if category_id not in cat_id_to_yolo:
                continue

            x, y, w, h = ann["bbox"]

            if w <= 0 or h <= 0:
                continue

            cls = cat_id_to_yolo[category_id]
            x_center = (x + w / 2.0) / w_img
            y_center = (y + h / 2.0) / h_img
            bw = w / w_img
            bh = h / h_img

            # 防止极少数框越界
            x_center = min(max(x_center, 0.0), 1.0)
            y_center = min(max(y_center, 0.0), 1.0)
            bw = min(max(bw, 0.0), 1.0)
            bh = min(max(bh, 0.0), 1.0)

            label_lines.append(
                f"{cls} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}"
            )

        label_path = label_dir / f"{Path(file_name).stem}.txt"
        label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""))

    print(f"{split}: images={len(images)}, labels written to {label_dir}")

yaml_path = dst / "SAR_AIRcraft.yaml"
with open(yaml_path, "w") as f:
    f.write(f"path: {dst}\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n")
    f.write("test: images/test\n\n")
    f.write("names:\n")
    for k, v in names.items():
        f.write(f"  {k}: {v}\n")

print("=" * 60)
print(f"YOLO dataset saved to: {dst}")
print(f"YAML saved to: {yaml_path}")
