#!/usr/bin/env python
"""Evaluate prediction TXT files against YOLO-style ground-truth labels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rgc_dino.dataset import discover_aligned_samples  # noqa: E402
from rgc_dino.labels import DetectionLabel, load_label_dir, load_label_file  # noqa: E402
from rgc_dino.metrics import evaluate_detection_map  # noqa: E402
from rgc_dino.sample_ids import load_sample_ids_file, restrict_mapping_to_sample_ids  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=ROOT / "source" / "训练集" / "labels")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--clip-labels", action="store_true", help="clip minor label drift to valid ranges")
    parser.add_argument("--sample-ids-file", type=Path, help="optional newline-delimited sample IDs to evaluate")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument(
        "--restrict-to-aligned-images",
        action="store_true",
        help="evaluate only label IDs that also have visible/infrared/depth images",
    )
    parser.add_argument("--strict", action="store_true", help="fail when prediction files are missing")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ground_truths = load_label_dir(args.labels, clip=args.clip_labels)
    if args.restrict_to_aligned_images:
        if args.dataset_root is None:
            print("--restrict-to-aligned-images requires --dataset-root", file=sys.stderr)
            return 2
        aligned_ids = {sample.sample_id for sample in discover_aligned_samples(args.dataset_root)}
        ground_truths = {
            sample_id: records
            for sample_id, records in ground_truths.items()
            if sample_id in aligned_ids
        }
    if args.sample_ids_file is not None:
        ground_truths = restrict_mapping_to_sample_ids(
            ground_truths,
            load_sample_ids_file(args.sample_ids_file),
        )

    if not ground_truths:
        print("no ground-truth labels found for evaluation", file=sys.stderr)
        return 2

    predictions: dict[str, list[DetectionLabel]] = {}
    missing_predictions: list[str] = []

    for sample_id in sorted(ground_truths):
        pred_path = args.predictions / f"{sample_id}.txt"
        if pred_path.exists():
            predictions[sample_id] = load_label_file(pred_path, require_confidence=True)
        else:
            predictions[sample_id] = []
            missing_predictions.append(sample_id)

    result = evaluate_detection_map(ground_truths, predictions)
    print(f"images: {len(ground_truths)}")
    print(f"ground_truth_objects: {result.ground_truth_count}")
    print(f"prediction_objects: {result.prediction_count}")
    print(f"mAP@50:95: {result.map:.6f}")
    print(f"mAP@50: {result.map50:.6f}")
    print(f"missing_prediction_files: {len(missing_predictions)}")
    if missing_predictions[:10]:
        print("missing_prediction_sample_ids:")
        for sample_id in missing_predictions[:10]:
            print(f"  {sample_id}")

    if args.strict and missing_predictions:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
