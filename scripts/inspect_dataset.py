#!/usr/bin/env python
"""Inspect aligned visible/infrared/depth samples without loading image pixels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rgc_dino.dataset import (  # noqa: E402
    MODALITIES,
    discover_aligned_samples,
    summarize_multimodal_dataset,
    write_manifest_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "source" / "AIC2026_PHASE_1_1000")
    parser.add_argument("--labels", type=Path, default=ROOT / "source" / "labels")
    parser.add_argument("--max-items", type=int, default=10)
    parser.add_argument("--manifest", type=Path, help="optional JSONL manifest output")
    parser.add_argument("--require-labels", action="store_true", help="manifest only labeled samples")
    parser.add_argument("--strict", action="store_true", help="exit non-zero on alignment gaps")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = summarize_multimodal_dataset(args.root, args.labels)

    print(f"dataset_root: {summary.root}")
    print(f"labels_dir: {summary.labels_dir}")
    print("modality_counts:")
    for modality in MODALITIES:
        print(f"  {modality}: {summary.modality_counts[modality]}")
    print(f"aligned_samples: {summary.aligned_count}")
    print(f"labeled_aligned_samples: {summary.labeled_aligned_count}")
    print(f"extra_labels: {len(summary.extra_label_ids)}")

    has_gaps = False
    for modality in MODALITIES:
        missing = summary.missing_by_modality[modality]
        if missing:
            has_gaps = True
            print(f"missing_{modality}: {len(missing)}")
            for sample_id in missing[: args.max_items]:
                print(f"  {sample_id}")

    if summary.extra_label_ids:
        print("extra_label_ids:")
        for sample_id in summary.extra_label_ids[: args.max_items]:
            print(f"  {sample_id}")

    if args.manifest is not None:
        samples = discover_aligned_samples(
            args.root,
            labels_dir=args.labels,
            require_labels=args.require_labels,
        )
        write_manifest_jsonl(samples, args.manifest)
        print(f"manifest: {args.manifest}")
        print(f"manifest_samples: {len(samples)}")

    if args.strict and (has_gaps or summary.extra_label_ids):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
