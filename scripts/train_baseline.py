#!/usr/bin/env python
"""Prepare a v0 baseline training run manifest without starting heavy training."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rgc_dino.constants import NUM_CLASSES  # noqa: E402
from rgc_dino.dataset import discover_aligned_samples, summarize_multimodal_dataset  # noqa: E402

DEFAULT_DATASET_ROOT = ROOT / "source" / "训练集"
DEFAULT_LABELS_DIR = DEFAULT_DATASET_ROOT / "labels"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "baseline_v0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--allow-heavy-training",
        action="store_true",
        help="Reserved for future real training; v0 still exits before training.",
    )
    parser.add_argument(
        "--require-labeled",
        action="store_true",
        help="exit non-zero when no aligned samples have labels",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.config.exists():
        print(f"config not found: {args.config}", file=sys.stderr)
        return 2

    summary = summarize_multimodal_dataset(args.dataset_root, args.labels)
    labeled_samples = discover_aligned_samples(
        args.dataset_root,
        labels_dir=args.labels,
        require_labels=True,
    )
    if args.require_labeled and not labeled_samples:
        print("no labeled aligned samples found", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "prepared_only",
        "note": "v0 baseline does not start heavy training from Codex sessions",
        "config": str(args.config),
        "dataset_root": str(args.dataset_root),
        "labels_dir": str(args.labels),
        "output_dir": str(args.output_dir),
        "num_classes": NUM_CLASSES,
        "aligned_samples": summary.aligned_count,
        "labeled_aligned_samples": len(labeled_samples),
        "extra_label_ids": len(summary.extra_label_ids),
        "has_labeled_aligned_samples": bool(labeled_samples),
        "allow_heavy_training_requested": args.allow_heavy_training,
    }
    manifest_path = args.output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"wrote: {manifest_path}")
    if not labeled_samples:
        print("warning: no labels match the aligned image IDs; prepared inference/submission loop only")
    print("heavy training was not started")
    if args.allow_heavy_training:
        print("real RGC-DINO training entry is not implemented in v0", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
