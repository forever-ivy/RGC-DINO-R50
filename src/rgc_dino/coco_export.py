"""Export the competition RGB labels into a COCO layout for official DINO."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from PIL import Image

from .constants import CLASS_NAMES
from .labels import DetectionLabel, load_label_file, load_label_file_clipped


def write_coco_rgb_dataset(
    *,
    dataset_root: str | Path,
    labels_dir: str | Path,
    output_root: str | Path,
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    clip_labels: bool = False,
) -> None:
    """Write RGB-only COCO train/val folders backed by symlinks to visible images."""
    root = Path(dataset_root)
    labels = Path(labels_dir)
    output = Path(output_root)
    annotations_dir = output / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    train = _build_split(root, labels, output / "train2017", train_ids, clip_labels=clip_labels)
    val = _build_split(root, labels, output / "val2017", val_ids, clip_labels=clip_labels)

    (annotations_dir / "instances_train2017.json").write_text(
        json.dumps(train, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (annotations_dir / "instances_val2017.json").write_text(
        json.dumps(val, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_split_ids(assignments_path: str | Path, *, fold: int) -> tuple[list[str], list[str]]:
    """Load train and val sample ids for one fold from ``fold_assignments.jsonl``."""
    train_ids: list[str] = []
    val_ids: list[str] = []
    for line in Path(assignments_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if int(row["fold"]) != fold:
            continue
        if row["split"] == "train":
            train_ids.append(row["sample_id"])
        elif row["split"] == "val":
            val_ids.append(row["sample_id"])
        else:
            raise ValueError(f"unknown split name: {row['split']}")
    if not train_ids or not val_ids:
        raise ValueError(f"fold {fold} does not contain both train and val ids")
    return sorted(train_ids), sorted(val_ids)


def _build_split(
    dataset_root: Path,
    labels_dir: Path,
    image_output_dir: Path,
    sample_ids: Sequence[str],
    *,
    clip_labels: bool,
) -> dict[str, object]:
    image_output_dir.mkdir(parents=True, exist_ok=True)
    images: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    annotation_id = 1

    for image_id, sample_id in enumerate(sorted(sample_ids), start=1):
        image_path = _find_visible_image(dataset_root, sample_id)
        linked_path = image_output_dir / image_path.name
        if linked_path.exists() or linked_path.is_symlink():
            linked_path.unlink()
        linked_path.symlink_to(image_path.resolve())

        with Image.open(image_path) as image:
            width, height = image.size

        images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )

        records = _load_records(labels_dir / f"{sample_id}.txt", clip_labels=clip_labels)
        for record in records:
            bbox = _label_to_coco_bbox(record, width=width, height=height)
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": record.class_id,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": class_id, "name": name}
            for class_id, name in enumerate(CLASS_NAMES)
        ],
    }


def _load_records(path: Path, *, clip_labels: bool) -> list[DetectionLabel]:
    if clip_labels:
        return load_label_file_clipped(path)[0]
    return load_label_file(path)


def _find_visible_image(dataset_root: Path, sample_id: str) -> Path:
    visible_dir = dataset_root / "visible"
    matches = sorted(visible_dir.glob(f"{sample_id}.*"))
    if not matches:
        raise FileNotFoundError(f"visible image not found for sample id: {sample_id}")
    if len(matches) > 1:
        raise ValueError(f"multiple visible images found for sample id: {sample_id}")
    return matches[0]


def _label_to_coco_bbox(record: DetectionLabel, *, width: int, height: int) -> list[float]:
    box_w = record.norm_w * width
    box_h = record.norm_h * height
    x = record.norm_center_x * width - box_w / 2.0
    y = record.norm_center_y * height - box_h / 2.0
    x = max(0.0, min(float(width), x))
    y = max(0.0, min(float(height), y))
    box_w = max(0.0, min(float(width) - x, box_w))
    box_h = max(0.0, min(float(height) - y, box_h))
    return [round(x, 6), round(y, 6), round(box_w, 6), round(box_h, 6)]
