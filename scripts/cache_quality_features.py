#!/usr/bin/env python
"""Precompute static 24-D quality vectors for aligned RGC-DINO samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rgc_dino.dataset import discover_aligned_samples  # noqa: E402
from rgc_dino.quality_features import load_quality_features, write_quality_feature_cache  # noqa: E402
from rgc_dino.sample_ids import load_sample_ids_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "source" / "训练集")
    parser.add_argument("--labels", type=Path, default=ROOT / "source" / "训练集" / "labels")
    parser.add_argument("--no-labels", action="store_true", help="cache an unlabeled prediction/test set")
    parser.add_argument("--sample-ids-file", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "cache" / "quality_features_train.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    labels_dir = None if args.no_labels else args.labels
    samples = discover_aligned_samples(
        args.dataset_root,
        labels_dir=labels_dir,
        require_labels=not args.no_labels,
    )
    if args.sample_ids_file is not None:
        wanted_ids = load_sample_ids_file(args.sample_ids_file)
        by_id = {sample.sample_id: sample for sample in samples}
        missing = [sample_id for sample_id in wanted_ids if sample_id not in by_id]
        if missing:
            raise ValueError(f"sample ids not found in aligned dataset: {missing[:5]}")
        samples = [by_id[sample_id] for sample_id in wanted_ids]

    cache = {
        sample.sample_id: load_quality_features(sample.visible_path, sample.infrared_path, sample.depth_path)
        for sample in samples
    }
    write_quality_feature_cache(args.output, cache)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "samples": len(cache),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
