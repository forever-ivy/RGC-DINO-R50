#!/usr/bin/env python
"""Merge base quality features with RDT diagnostics for gate ablations.

This creates a sample_id -> base_rdt feature cache consumed by train/infer with
``--quality-feature-set base_rdt``. It does not alter RGB pixels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rgc_dino.quality_features import (  # noqa: E402
    QUALITY_FEATURE_NAMES,
    RDT_QUALITY_FEATURE_NAMES,
    load_quality_feature_cache,
    write_quality_feature_cache,
)
from rgc_dino.sample_ids import load_sample_ids_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-quality-cache", type=Path, required=True)
    parser.add_argument("--rdt-stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-ids-file", type=Path, help="optional newline-delimited sample IDs to keep")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_cache = load_quality_feature_cache(args.base_quality_cache, feature_set="base")
    rdt_stats = _load_rdt_stats(args.rdt_stats)
    if args.sample_ids_file is not None:
        sample_ids = load_sample_ids_file(args.sample_ids_file)
    else:
        sample_ids = sorted(base_cache)
    merged = merge_quality_and_rdt(base_cache, rdt_stats, sample_ids=sample_ids)
    write_quality_feature_cache(args.output, merged, feature_set="base_rdt")
    summary = {
        "output": str(args.output),
        "samples": len(merged),
        "base_features": len(QUALITY_FEATURE_NAMES),
        "rdt_features": len(RDT_QUALITY_FEATURE_NAMES),
        "feature_set": "base_rdt",
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


def merge_quality_and_rdt(
    base_cache: dict[str, dict[str, float]],
    rdt_stats: dict[str, dict[str, float]],
    *,
    sample_ids: list[str] | tuple[str, ...] | None = None,
) -> dict[str, dict[str, float]]:
    ids = list(sample_ids) if sample_ids is not None else sorted(base_cache)
    missing_base = [sample_id for sample_id in ids if sample_id not in base_cache]
    missing_rdt = [sample_id for sample_id in ids if sample_id not in rdt_stats]
    if missing_base or missing_rdt:
        raise ValueError(f"missing sample IDs base={missing_base[:5]} rdt={missing_rdt[:5]}")
    merged: dict[str, dict[str, float]] = {}
    for sample_id in ids:
        row = {name: float(base_cache[sample_id][name]) for name in QUALITY_FEATURE_NAMES}
        for name in RDT_QUALITY_FEATURE_NAMES:
            if name not in rdt_stats[sample_id]:
                raise ValueError(f"{sample_id}: missing RDT feature {name}")
            row[name] = float(rdt_stats[sample_id][name])
        merged[sample_id] = row
    return merged


def _load_rdt_stats(path: Path) -> dict[str, dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("RDT stats must be a sample_id -> stats mapping")
    return {str(sample_id): {str(key): float(value) for key, value in stats.items()} for sample_id, stats in payload.items()}


if __name__ == "__main__":
    raise SystemExit(main())
