"""Read-only source verification plus task-owned YOLO cache/symlink staging."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

from detector_spec import EXPECTED_COUNTS, digest, sha


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def write_json(path, value, replace=False):
    path = Path(path)
    if path.exists() and not replace:
        raise FileExistsError("Refusing to overwrite task artifact: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if not replace and path.exists():
        raise FileExistsError("Artifact appeared during publication: " + str(path))
    os.replace(temp, path)


def names_list(value):
    if isinstance(value, list):
        values = value
    elif isinstance(value, dict):
        normalized = {int(key): item for key, item in value.items()}
        if sorted(normalized) != list(range(len(normalized))):
            raise ValueError("Class ids must be contiguous from zero")
        values = [normalized[index] for index in range(len(normalized))]
    else:
        raise ValueError("Explicit class names are required")
    if not values or any(not isinstance(name, str) or not name for name in values):
        raise ValueError("Invalid class names")
    return values


def relative_to(base, value):
    path = Path(value)
    return path if path.is_absolute() else base / path


def load_declaration(data_path, manifest_path):
    descriptor, inventory, evidence = {}, None, {}
    if manifest_path is not None:
        manifest_path = Path(manifest_path).resolve(strict=True)
        evidence["input_manifest"] = {"path": str(manifest_path), "sha256": sha(manifest_path)}
        item = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(item, list):
            inventory = item
        elif isinstance(item, dict):
            descriptor = item
            inventory = item.get("records")
            if inventory is None and item.get("manifest"):
                nested = relative_to(manifest_path.parent, item["manifest"]).resolve(strict=True)
                if item.get("manifest_sha256") and sha(nested) != item["manifest_sha256"]:
                    raise RuntimeError("Nested dataset inventory SHA mismatch")
                evidence["inventory"] = {"path": str(nested), "sha256": sha(nested)}
                inventory = json.loads(nested.read_text(encoding="utf-8"))
            if data_path is None:
                declared = item.get("data_yaml") or item.get("yaml")
                if declared:
                    data_path = relative_to(manifest_path.parent, declared)
        else:
            raise ValueError("Dataset manifest must be a descriptor or row list")
        if not isinstance(inventory, list) or not inventory:
            raise ValueError("Explicit nonempty records are required in a dataset manifest")
    if data_path is None:
        raise ValueError("Pass --data or a descriptor with data_yaml/yaml")
    data_path = Path(data_path).resolve(strict=True)
    yaml_digest = sha(data_path)
    expected_yaml = descriptor.get("data_yaml_sha256") or descriptor.get("yaml_sha256")
    if expected_yaml and expected_yaml != yaml_digest:
        raise RuntimeError("Dataset YAML SHA mismatch")
    evidence["data_yaml"] = {"path": str(data_path), "sha256": yaml_digest}
    import yaml
    declaration = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    if not isinstance(declaration, dict) or declaration.get("download"):
        raise ValueError("Only explicit local YAML without download commands is allowed")
    root = relative_to(data_path.parent, declaration.get("path", ".")).resolve(strict=True)
    if descriptor.get("root") and Path(descriptor["root"]).resolve() != root:
        raise RuntimeError("Manifest and YAML dataset roots differ")
    class_names = names_list(declaration.get("names"))
    if "nc" in declaration and declaration["nc"] != len(class_names):
        raise RuntimeError("YAML nc differs from names")
    if descriptor.get("names") is not None and names_list(descriptor["names"]) != class_names:
        raise RuntimeError("Manifest and YAML class order differ")
    return declaration, descriptor, inventory, evidence, root, class_names


def validate_label(path, nc):
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        items = line.split()
        if len(items) != 5:
            raise ValueError("Expected five-column YOLO box label: " + str(path))
        row = [float(item) for item in items]
        if (not all(math.isfinite(item) for item in row) or row[0] != int(row[0])
                or not 0 <= row[0] < nc or not all(0 <= item <= 1 for item in row[1:])
                or row[3] <= 0 or row[4] <= 0):
            raise ValueError("Invalid YOLO box/class; labels are never repaired: " + str(path))
        rows.append(row)
    return len(rows)


def validate_image_readonly(path):
    # Ultralytics' upstream verifier repairs a JPEG missing its EOI marker.
    # Reject that case here so a symlink can never lead it to rewrite a source.
    from PIL import Image
    with Image.open(path) as image:
        image_format = image.format
        if min(image.size) < 10:
            raise ValueError("Image smaller than supported detection input: " + str(path))
        image.verify()
    if image_format == "JPEG":
        with path.open("rb") as stream:
            stream.seek(-2, os.SEEK_END)
            if stream.read() != b"\xff\xd9":
                raise ValueError("JPEG requires repair; source is left unchanged: " + str(path))


def prepare_data(args):
    if args.dataset_id == "sar_aircraft":
        raise RuntimeError("BLOCKED_AIR_LABEL_CONFLICT: authority/protocol unresolved; no training or probe")
    declaration, descriptor, inventory, evidence, root, class_names = load_declaration(
        args.data, args.dataset_manifest)
    if len(class_names) != 1:
        raise ValueError("This campaign currently admits only the audited single-class SSDD/SADD/Ship sets")
    if args.dataset_id == "sadd" and descriptor.get("protocol") != "sadd-full2966-posneg-stratified-80-10-10-rgbgroup-v1":
        raise RuntimeError("SADD requires the new full2966 positive+negative manifest, not old positive-only data")
    split_names = ("train",) if args.mode == "probe" else ("train", "val", "test")
    expected_counts = EXPECTED_COUNTS[args.dataset_id]
    if descriptor.get("counts") and descriptor["counts"] != expected_counts:
        raise RuntimeError("Declared counts differ from this frozen dataset protocol")
    expected = {}
    if inventory is not None:
        for row in inventory:
            if row.get("split") not in {"train", "val", "test"}:
                raise ValueError("Unknown manifest split")
            if row["split"] not in split_names:
                continue  # Probe must not open or enumerate held-out image/label files.
            name = row.get("name") or Path(row.get("image", row.get("path", ""))).name
            if not name or Path(name).name != name or (row["split"], name) in expected:
                raise ValueError("Duplicate or unsafe manifest image name")
            expected[row["split"], name] = row
    staged_root = args.output / "data"
    records, counts, split_hashes = [], {}, {}
    for split in split_names:
        value = declaration.get(split)
        if not isinstance(value, str):
            raise ValueError("Explicit local image-directory splits are required: " + split)
        images = relative_to(root, value)
        if not images.is_dir() or images.name != split or images.parent.name != "images":
            raise ValueError("Split must be an images/<split> directory; no inferred alternate protocol")
        labels = images.parent.parent / "labels" / split
        found = sorted(path for path in images.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        if len(found) != expected_counts[split]:
            raise RuntimeError(f"{split}: expected {expected_counts[split]} images, found {len(found)}")
        if inventory is not None and {path.name for path in found} != {name for part, name in expected if part == split}:
            raise RuntimeError("Directory members differ from the fixed manifest: " + split)
        out_images, out_labels = staged_root / "images" / split, staged_root / "labels" / split
        out_images.mkdir(parents=True)
        out_labels.mkdir(parents=True)
        used_label_names = set()
        for image in found:
            label = labels / (image.stem + ".txt")
            if not label.is_file() or label.name in used_label_names:
                raise ValueError("Missing/ambiguous label mapping: " + str(label))
            used_label_names.add(label.name)
            image_hash, label_hash = sha(image), sha(label)
            declared = expected.get((split, image.name))
            if declared and (declared.get("image_sha256") != image_hash or declared.get("label_sha256") != label_hash):
                raise RuntimeError("Manifest image/label SHA mismatch: " + str(image))
            validate_image_readonly(image)
            boxes = validate_label(label, len(class_names))
            (out_images / image.name).symlink_to(image.resolve())
            (out_labels / label.name).symlink_to(label.resolve())
            records.append({"split": split, "name": image.name,
                            "image": str(image.resolve()), "label": str(label.resolve()),
                            "image_sha256": image_hash, "label_sha256": label_hash,
                            "box_count": boxes, "positive": boxes > 0})
            split_hashes.setdefault(split, set()).add(image_hash)
        counts[split] = len(found)
    for left in split_hashes:
        for right in split_hashes:
            if left < right and split_hashes[left] & split_hashes[right]:
                raise RuntimeError("Byte-identical images cross splits; no automatic relabel/repartition")
    task_data = {"path": str(staged_root.resolve()), "nc": len(class_names),
                 "names": {index: name for index, name in enumerate(class_names)}}
    task_data.update({split: str(staged_root / "images" / split) for split in split_names})
    write_json(args.output / "data.yaml", task_data)
    write_json(args.output / "verified_data_records.json", records)
    content_records = [{key: row[key] for key in ("split", "name", "image_sha256", "label_sha256")}
                       for row in records]
    evidence.update({"dataset_id": args.dataset_id, "source_root": str(root), "counts_verified": counts,
                     "expected_counts": expected_counts, "names": class_names,
                     "actual_records_sha256": digest({"names": class_names, "records": content_records}),
                     "location_records_sha256": digest(records), "files_verified": len(records) * 2,
                     "source_data_protocol": descriptor.get("protocol"),
                     "probe_only": args.mode == "probe",
                     "heldout_image_or_label_files_opened": args.mode != "probe",
                     "source_images_repaired": False, "cache_location": str(staged_root)})
    write_json(args.output / "data_audit.json", evidence)
    return task_data, records, evidence


def recheck_data(records, evidence):
    for field in ("input_manifest", "inventory", "data_yaml"):
        entry = evidence.get(field)
        if entry and sha(entry["path"]) != entry["sha256"]:
            raise RuntimeError("Dataset declaration changed while the run was active: " + field)
    for row in records:
        if sha(row["image"]) != row["image_sha256"] or sha(row["label"]) != row["label_sha256"]:
            raise RuntimeError("Input data changed while the run was active: " + row["image"])
