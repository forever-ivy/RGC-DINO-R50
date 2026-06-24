#!/usr/bin/env python
"""Generate RGB-guided-RDT saliency previews and stats for aligned samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rgc_dino.dataset import discover_aligned_samples  # noqa: E402
from rgc_dino.rdt import load_rdt_result, write_rdt_preview, write_rdt_stats  # noqa: E402
from rgc_dino.sample_ids import load_sample_ids_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "source" / "训练集")
    parser.add_argument("--labels", type=Path, default=ROOT / "source" / "训练集" / "labels")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-ids-file", type=Path, help="optional newline-delimited sample IDs")
    parser.add_argument("--limit", type=int, default=24, help="maximum samples to process; use 0 for all selected samples")
    parser.add_argument("--max-side", type=int, default=640)
    parser.add_argument("--ir-weight", type=float, default=0.55)
    parser.add_argument("--depth-weight", type=float, default=0.45)
    parser.add_argument("--base-gate", type=float, default=0.85)
    parser.add_argument("--gain", type=float, default=0.30)
    parser.add_argument("--write-previews", action="store_true", default=True)
    parser.add_argument("--no-write-previews", dest="write_previews", action="store_false")
    parser.add_argument("--require-labels", action="store_true", default=True)
    parser.add_argument("--no-require-labels", dest="require_labels", action="store_false")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    samples = discover_aligned_samples(
        args.dataset_root,
        labels_dir=args.labels if args.labels.exists() else None,
        require_labels=args.require_labels,
    )
    if args.sample_ids_file is not None:
        requested = load_sample_ids_file(args.sample_ids_file)
        by_id = {sample.sample_id: sample for sample in samples}
        missing = [sample_id for sample_id in requested if sample_id not in by_id]
        if missing:
            print(f"missing requested sample IDs: {missing[:10]}", file=sys.stderr)
            return 2
        samples = [by_id[sample_id] for sample_id in requested]
    if args.limit is not None and args.limit > 0:
        samples = samples[: args.limit]
    if not samples:
        print("no samples selected", file=sys.stderr)
        return 2

    preview_dir = args.output_dir / "previews"
    stats_by_sample_id: dict[str, dict[str, float]] = {}
    manifest_rows = []
    for sample in samples:
        result = load_rdt_result(
            sample.visible_path,
            sample.infrared_path,
            sample.depth_path,
            max_side=args.max_side,
            ir_weight=args.ir_weight,
            depth_weight=args.depth_weight,
            base_gate=args.base_gate,
            gain=args.gain,
        )
        stats_by_sample_id[sample.sample_id] = result.stats
        preview_path = write_rdt_preview(result, preview_dir / f"{sample.sample_id}.jpg") if args.write_previews else None
        manifest_rows.append(
            {
                "sample_id": sample.sample_id,
                "visible_path": str(sample.visible_path),
                "infrared_path": str(sample.infrared_path),
                "depth_path": str(sample.depth_path),
                "label_path": str(sample.label_path) if sample.label_path is not None else None,
                "preview_path": str(preview_path) if preview_path is not None else None,
                "stats": result.stats,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rdt_stats(args.output_dir / "rdt_stats.json", stats_by_sample_id)
    (args.output_dir / "manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(_summarize(stats_by_sample_id), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"samples": len(samples), "output_dir": str(args.output_dir)}, sort_keys=True))
    return 0


def _summarize(stats_by_sample_id: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    if not stats_by_sample_id:
        return {}
    keys = sorted(next(iter(stats_by_sample_id.values())).keys())
    summary: dict[str, dict[str, float]] = {}
    for key in keys:
        values = [float(stats[key]) for stats in stats_by_sample_id.values()]
        values_sorted = sorted(values)
        n = len(values_sorted)
        summary[key] = {
            "min": values_sorted[0],
            "mean": sum(values_sorted) / n,
            "median": values_sorted[n // 2],
            "max": values_sorted[-1],
        }
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
