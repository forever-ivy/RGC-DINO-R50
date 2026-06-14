#!/usr/bin/env python
"""Merge multiple prediction directories into one submission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rgc_dino.ensemble import ensemble_prediction_dirs  # noqa: E402
from rgc_dino.submission import validate_submission_dir, zip_submission_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--zip-path", type=Path)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.8)
    parser.add_argument("--min-model-votes", type=int, default=1)
    parser.add_argument("--max-detections", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = ensemble_prediction_dirs(
        args.prediction_dirs,
        args.output_dir,
        nms_iou_threshold=args.nms_iou_threshold,
        min_model_votes=args.min_model_votes,
        max_predictions_per_image=args.max_detections,
    )
    image_ids = [path.stem for path in sorted(args.prediction_dirs[0].glob("*.txt"))]
    errors = validate_submission_dir(image_ids, args.output_dir, max_predictions_per_image=args.max_detections)
    if errors:
        print("ensemble validation failed:", file=sys.stderr)
        for error in errors[:20]:
            print(f"  {error}", file=sys.stderr)
        return 1
    if args.zip_path is not None:
        zip_submission_dir(args.output_dir, args.zip_path)
        summary["zip_path"] = str(args.zip_path)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
