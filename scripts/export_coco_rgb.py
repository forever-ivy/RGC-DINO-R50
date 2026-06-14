#!/usr/bin/env python
"""Export one RGB-only fold to the COCO layout expected by official DINO."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rgc_dino.coco_export import load_split_ids, write_coco_rgb_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "source" / "训练集")
    parser.add_argument("--labels", type=Path, default=ROOT / "source" / "训练集" / "labels")
    parser.add_argument(
        "--assignments",
        type=Path,
        default=ROOT / "outputs" / "splits" / "fold_assignments.jsonl",
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs" / "dino_a0_rgb" / "fold0_coco",
    )
    parser.add_argument("--clip-labels", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_ids, val_ids = load_split_ids(args.assignments, fold=args.fold)
    write_coco_rgb_dataset(
        dataset_root=args.dataset_root,
        labels_dir=args.labels,
        output_root=args.output_root,
        train_ids=train_ids,
        val_ids=val_ids,
        clip_labels=args.clip_labels,
    )
    print(f"wrote: {args.output_root}")
    print(f"train_ids: {len(train_ids)}")
    print(f"val_ids: {len(val_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
