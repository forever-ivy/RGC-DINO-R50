#!/usr/bin/env python
"""Validate and zip competition submission TXT files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rgc_dino.dataset import discover_aligned_samples  # noqa: E402
from rgc_dino.submission import validate_submission_dir, zip_submission_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "source" / "AIC2026_PHASE_1_1000")
    parser.add_argument("--submission-dir", type=Path, required=True)
    parser.add_argument("--zip-path", type=Path, required=True)
    parser.add_argument("--skip-validation", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_ids = [sample.sample_id for sample in discover_aligned_samples(args.dataset_root)]
    if not image_ids:
        print("no aligned samples found", file=sys.stderr)
        return 2

    if not args.skip_validation:
        errors = validate_submission_dir(image_ids, args.submission_dir)
        if errors:
            print("submission validation failed:", file=sys.stderr)
            for error in errors[:20]:
                print(f"  {error}", file=sys.stderr)
            if len(errors) > 20:
                print(f"  ... {len(errors) - 20} more", file=sys.stderr)
            return 1

    zip_path = zip_submission_dir(args.submission_dir, args.zip_path)
    print(f"wrote: {zip_path}")
    print(f"files: {len(image_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
