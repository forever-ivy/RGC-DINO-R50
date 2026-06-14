#!/usr/bin/env python
"""Create deterministic grouped fold manifests from label TXT files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rgc_dino.splits import (  # noqa: E402
    build_grouped_stratified_splits,
    load_label_class_ids,
    write_split_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=ROOT / "source" / "labels")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "splits")
    parser.add_argument("--clip-labels", action="store_true", help="clip minor label drift")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        labels = load_label_class_ids(args.labels, clip=args.clip_labels)
        splits = build_grouped_stratified_splits(labels, folds=args.folds)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    manifest_path = write_split_manifest(splits, args.output_dir)
    print(f"wrote: {manifest_path}")
    for fold in splits:
        print(
            f"fold {fold.fold_index}: train={len(fold.train_ids)} "
            f"val={len(fold.val_ids)} objects={sum(fold.val_class_counts.values())}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
