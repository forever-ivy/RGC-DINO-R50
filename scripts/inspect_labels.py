#!/usr/bin/env python
"""Inspect YOLO-style training labels without starting model training."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rgc_dino.constants import CLASS_NAMES, NUM_CLASSES  # noqa: E402
from rgc_dino.labels import iter_label_files, parse_label_line  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=ROOT / "source" / "labels")
    parser.add_argument("--max-errors", type=int, default=10)
    parser.add_argument("--strict", action="store_true", help="exit non-zero when invalid labels exist")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.labels.exists():
        print(f"labels directory not found: {args.labels}", file=sys.stderr)
        return 2

    files = list(iter_label_files(args.labels))
    class_counts: Counter[int] = Counter()
    empty_files = 0
    object_count = 0
    invalid_count = 0
    errors: list[str] = []

    for label_file in files:
        records_in_file = 0
        for line_number, line in enumerate(label_file.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = parse_label_line(stripped, source=f"{label_file}:{line_number}")
            except ValueError as exc:
                invalid_count += 1
                if len(errors) < args.max_errors:
                    errors.append(str(exc))
                continue

            records_in_file += 1
            object_count += 1
            class_counts.update([record.class_id])

        if records_in_file == 0:
            empty_files += 1

    print(f"labels_dir: {args.labels}")
    print(f"files: {len(files)}")
    print(f"empty_files: {empty_files}")
    print(f"objects: {object_count}")
    print(f"invalid_objects: {invalid_count}")
    print("class_counts:")
    for class_id in range(NUM_CLASSES):
        print(f"  {class_id:2d} {CLASS_NAMES[class_id]:12s} {class_counts[class_id]}")

    if errors:
        print("warnings:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        if args.strict:
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
