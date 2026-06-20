#!/usr/bin/env python
"""Export AIC2026 test visible images to a COCO-style test set for Co-DETR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rgc_dino.constants import CLASS_NAMES  # noqa: E402
from rgc_dino.dataset import discover_aligned_samples  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "source" / "AIC2026_PHASE_1_1000")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "codetr_coco" / "aic2026_test")
    parser.add_argument("--image-dir-name", default="test2017")
    parser.add_argument("--ann-name", default="instances_test2017.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    samples = discover_aligned_samples(args.dataset_root)
    if not samples:
        print(f"no aligned samples found under {args.dataset_root}", file=sys.stderr)
        return 2

    image_dir = args.output_root / args.image_dir_name
    ann_dir = args.output_root / "annotations"
    image_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    images: list[dict[str, object]] = []
    for image_id, sample in enumerate(samples, start=1):
        image_path = sample.visible_path
        linked_path = image_dir / image_path.name
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

    payload = {
        "images": images,
        "annotations": [],
        "categories": [
            {"id": class_id, "name": name}
            for class_id, name in enumerate(CLASS_NAMES)
        ],
    }
    ann_path = ann_dir / args.ann_name
    ann_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"wrote: {args.output_root}")
    print(f"images: {len(images)}")
    print(f"ann_file: {ann_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
