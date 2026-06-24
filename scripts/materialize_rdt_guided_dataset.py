#!/usr/bin/env python
"""Materialize a dataset view with RGB-guided-RDT visible images.

The output keeps the competition directory contract (visible/infrared/depth and
labels). Only visible images are rewritten; infrared/depth/labels are symlinked
by default. This is intended for lightweight validation ablations before any
training decision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rgc_dino.dataset import discover_aligned_samples  # noqa: E402
from rgc_dino.rdt import load_rdt_result  # noqa: E402
from rgc_dino.sample_ids import load_sample_ids_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "source" / "训练集")
    parser.add_argument("--labels", type=Path, default=ROOT / "source" / "训练集" / "labels")
    parser.add_argument("--sample-ids-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-side", type=int, help="optional resize before RDT computation; omit to keep original size")
    parser.add_argument("--ir-weight", type=float, default=0.55)
    parser.add_argument("--depth-weight", type=float, default=0.45)
    parser.add_argument("--base-gate", type=float, default=0.85)
    parser.add_argument("--gain", type=float, default=0.30)
    parser.add_argument("--copy-linked", action="store_true", help="copy infrared/depth/labels instead of symlinking")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    requested_ids = load_sample_ids_file(args.sample_ids_file)
    requested_set = set(requested_ids)
    samples = discover_aligned_samples(args.dataset_root, labels_dir=args.labels, require_labels=True)
    by_id = {sample.sample_id: sample for sample in samples}
    missing = sorted(requested_set - set(by_id))
    if missing:
        print(f"missing sample IDs: {missing[:10]}", file=sys.stderr)
        return 2

    for subdir in ("visible", "infrared", "depth", "labels"):
        (args.output_root / subdir).mkdir(parents=True, exist_ok=True)

    stats: dict[str, dict[str, float | str]] = {}
    manifest_rows: list[dict[str, str]] = []
    for sample_id in requested_ids:
        sample = by_id[sample_id]
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
        visible_out = args.output_root / "visible" / sample.visible_path.name
        Image.fromarray(result.guided_rgb).save(visible_out)
        infrared_out = args.output_root / "infrared" / sample.infrared_path.name
        depth_out = args.output_root / "depth" / sample.depth_path.name
        label_out = args.output_root / "labels" / f"{sample_id}.txt"
        _link_or_copy(sample.infrared_path, infrared_out, copy=args.copy_linked)
        _link_or_copy(sample.depth_path, depth_out, copy=args.copy_linked)
        if sample.label_path is None:
            raise FileNotFoundError(f"label missing for {sample_id}")
        _link_or_copy(sample.label_path, label_out, copy=args.copy_linked)
        row = {
            "sample_id": sample_id,
            "visible_path": str(visible_out),
            "infrared_path": str(infrared_out),
            "depth_path": str(depth_out),
            "label_path": str(label_out),
        }
        manifest_rows.append(row)
        stats[sample_id] = {**result.stats, **row}

    (args.output_root / "rdt_materialized_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_root / "manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    summary = {
        "output_root": str(args.output_root),
        "samples": len(requested_ids),
        "max_side": args.max_side,
        "ir_weight": args.ir_weight,
        "depth_weight": args.depth_weight,
        "base_gate": args.base_gate,
        "gain": args.gain,
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _link_or_copy(source: Path, destination: Path, *, copy: bool) -> None:
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if copy:
        shutil.copy2(source, destination)
    else:
        destination.symlink_to(source.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
