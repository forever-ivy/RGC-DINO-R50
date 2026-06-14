#!/usr/bin/env python
"""Precompute static 24-D quality vectors for aligned RGC-DINO samples."""

from __future__ import annotations

import argparse
import json
from multiprocessing import get_context
from pathlib import Path
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rgc_dino.dataset import MultimodalSample, discover_aligned_samples  # noqa: E402
from rgc_dino.quality_features import load_quality_features, write_quality_feature_cache  # noqa: E402
from rgc_dino.sample_ids import load_sample_ids_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "source" / "训练集")
    parser.add_argument("--labels", type=Path, default=ROOT / "source" / "训练集" / "labels")
    parser.add_argument("--no-labels", action="store_true", help="cache an unlabeled prediction/test set")
    parser.add_argument("--sample-ids-file", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "cache" / "quality_features_train.json")
    parser.add_argument(
        "--max-side-for-quality",
        type=int,
        default=640,
        help="resize each modality's longest side before computing static quality features; use 0 for full resolution",
    )
    parser.add_argument("--num-workers", type=int, default=1, help="parallel workers for cache generation")
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

    max_side = args.max_side_for_quality if args.max_side_for_quality > 0 else None
    cache = build_quality_cache(samples, max_side=max_side, num_workers=args.num_workers)
    write_quality_feature_cache(args.output, cache)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "samples": len(cache),
                "max_side_for_quality": max_side,
                "num_workers": args.num_workers,
            },
            sort_keys=True,
        )
    )
    return 0


def build_quality_cache(
    samples: Iterable[MultimodalSample],
    *,
    max_side: int | None,
    num_workers: int = 1,
) -> dict[str, dict[str, float]]:
    sample_list = list(samples)
    if num_workers <= 1:
        return dict(_compute_one(sample, max_side=max_side) for sample in sample_list)
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")
    with get_context("spawn").Pool(processes=num_workers) as pool:
        rows = pool.starmap(_compute_one_from_paths, [(_sample_payload(sample), max_side) for sample in sample_list])
    return dict(rows)


def _sample_payload(sample: MultimodalSample) -> tuple[str, str, str, str]:
    return (
        sample.sample_id,
        str(sample.visible_path),
        str(sample.infrared_path),
        str(sample.depth_path),
    )


def _compute_one(sample: MultimodalSample, *, max_side: int | None) -> tuple[str, dict[str, float]]:
    return (
        sample.sample_id,
        load_quality_features(
            sample.visible_path,
            sample.infrared_path,
            sample.depth_path,
            max_side=max_side,
        ),
    )


def _compute_one_from_paths(
    payload: tuple[str, str, str, str],
    max_side: int | None,
) -> tuple[str, dict[str, float]]:
    sample_id, visible_path, infrared_path, depth_path = payload
    return (
        sample_id,
        load_quality_features(
            visible_path,
            infrared_path,
            depth_path,
            max_side=max_side,
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
