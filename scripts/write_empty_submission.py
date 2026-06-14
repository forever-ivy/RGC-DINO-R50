#!/usr/bin/env python
"""Write a legal empty submission skeleton for aligned images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rgc_dino.dataset import discover_aligned_samples  # noqa: E402
from rgc_dino.submission import validate_submission_dir, write_empty_submission  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "source" / "AIC2026_PHASE_1_1000")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "empty_submission")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    samples = discover_aligned_samples(args.dataset_root)
    image_ids = [sample.sample_id for sample in samples]
    if not image_ids:
        print("no aligned samples found", file=sys.stderr)
        return 2

    write_empty_submission(image_ids, args.output_dir)
    errors = validate_submission_dir(image_ids, args.output_dir)
    if errors:
        print("empty submission validation failed:", file=sys.stderr)
        for error in errors[:20]:
            print(f"  {error}", file=sys.stderr)
        return 1

    print(f"wrote: {args.output_dir}")
    print(f"files: {len(image_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
