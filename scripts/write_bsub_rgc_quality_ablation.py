#!/usr/bin/env python
"""Write LSF scripts for RGC quality-feature gate ablations.

The ablation compares the existing 24-D base quality prior against the extended
base+RDT quality prior. It only generates scripts; it does not submit jobs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRETRAIN_DINO_WEIGHTS = Path("/data1/liuxuan/checkpoints/dino/checkpoint0011_4scale.pth")
DEFAULT_BASE_CACHE = ROOT / "outputs" / "cache" / "quality_features_train.json"
DEFAULT_RDT_DIR = ROOT / "outputs" / "cache" / "rdt_stats_train_640"
DEFAULT_BASE_RDT_CACHE = ROOT / "outputs" / "cache" / "quality_features_train_base_rdt.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "jobs" / "rgc_quality_base_rdt_ablation.lsf")
    parser.add_argument("--job-name", default="rgc-quality-rdt-ablate")
    parser.add_argument("--queue", default="normal")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr-drop", type=int, default=1)
    parser.add_argument("--image-max-side", type=int, default=640)
    parser.add_argument("--train-image-max-sides", type=int, nargs="+", default=[480, 560, 640])
    parser.add_argument("--limit-train", type=int, default=256)
    parser.add_argument("--limit-val", type=int, default=128)
    parser.add_argument("--val-batches", type=int, default=30)
    parser.add_argument("--log-gates-batches", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--pretrain-dino-weights", type=Path, default=DEFAULT_PRETRAIN_DINO_WEIGHTS)
    parser.add_argument("--base-quality-cache", type=Path, default=DEFAULT_BASE_CACHE)
    parser.add_argument("--rdt-output-dir", type=Path, default=DEFAULT_RDT_DIR)
    parser.add_argument("--base-rdt-quality-cache", type=Path, default=DEFAULT_BASE_RDT_CACHE)
    parser.add_argument("--base-output-dir", type=Path, default=ROOT / "outputs" / "rgc_dino" / "quality_base_ablation")
    parser.add_argument("--base-rdt-output-dir", type=Path, default=ROOT / "outputs" / "rgc_dino" / "quality_base_rdt_ablation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script = render_bsub_script(
        job_name=args.job_name,
        queue=args.queue,
        gpu=args.gpu,
        fold=args.fold,
        epochs=args.epochs,
        lr_drop=args.lr_drop,
        image_max_side=args.image_max_side,
        train_image_max_sides=tuple(args.train_image_max_sides),
        limit_train=args.limit_train,
        limit_val=args.limit_val,
        val_batches=args.val_batches,
        log_gates_batches=args.log_gates_batches,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pretrain_dino_weights=args.pretrain_dino_weights,
        base_quality_cache=args.base_quality_cache,
        rdt_output_dir=args.rdt_output_dir,
        base_rdt_quality_cache=args.base_rdt_quality_cache,
        base_output_dir=args.base_output_dir,
        base_rdt_output_dir=args.base_rdt_output_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(script, encoding="utf-8")
    print(f"wrote: {args.output}")
    print("submit manually with:")
    print(f"  bsub < {args.output}")
    return 0


def render_bsub_script(
    *,
    job_name: str,
    queue: str,
    gpu: int,
    fold: int,
    epochs: int,
    lr_drop: int,
    image_max_side: int,
    train_image_max_sides: tuple[int, ...],
    limit_train: int,
    limit_val: int,
    val_batches: int,
    log_gates_batches: int,
    batch_size: int,
    num_workers: int,
    pretrain_dino_weights: Path,
    base_quality_cache: Path,
    rdt_output_dir: Path,
    base_rdt_quality_cache: Path,
    base_output_dir: Path,
    base_rdt_output_dir: Path,
) -> str:
    train_scales = " ".join(str(side) for side in train_image_max_sides)
    return f"""#!/bin/sh
#BSUB -J {job_name}
#BSUB -q {queue}
#BSUB -n 4
#BSUB -gpu "num={gpu}:mode=exclusive_process"
#BSUB -R "rusage[mem=24000]"
#BSUB -o /data1/liuxuan/logs/{job_name}.%J.out
#BSUB -e /data1/liuxuan/logs/{job_name}.%J.err

set -eu
cd {ROOT}
. /data1/liuxuan/activate-py310.sh
mkdir -p /data1/liuxuan/logs

PRETRAIN_DINO_WEIGHTS="{pretrain_dino_weights}"
if [ ! -f "$PRETRAIN_DINO_WEIGHTS" ]; then
  echo "missing official DINO COCO pretrained weights: $PRETRAIN_DINO_WEIGHTS" >&2
  exit 2
fi

python scripts/cache_quality_features.py \
  --output {base_quality_cache} \
  --max-side-for-quality {image_max_side} \
  --num-workers {num_workers}

PYTHONPATH=src python scripts/diagnose_rdt_saliency.py \
  --dataset-root source/训练集 \
  --labels source/训练集/labels \
  --output-dir {rdt_output_dir} \
  --limit 0 \
  --max-side {image_max_side} \
  --no-write-previews

PYTHONPATH=src python scripts/merge_rdt_quality_cache.py \
  --base-quality-cache {base_quality_cache} \
  --rdt-stats {rdt_output_dir}/rdt_stats.json \
  --output {base_rdt_quality_cache}

run_ablation() {{
  FEATURE_SET="$1"
  QUALITY_CACHE="$2"
  OUTPUT_DIR="$3"
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/train_rgc_dino.py \
    --output-dir "$OUTPUT_DIR" \
    --fold {fold} \
    --epochs {epochs} \
    --lr-drop {lr_drop} \
    --batch-size {batch_size} \
    --num-workers {num_workers} \
    --image-max-side {image_max_side} \
    --train-image-max-sides {train_scales} \
    --pretrain-dino-weights "$PRETRAIN_DINO_WEIGHTS" \
    --quality-cache "$QUALITY_CACHE" \
    --quality-feature-set "$FEATURE_SET" \
    --random-horizontal-flip-prob 0.5 \
    --limit-train {limit_train} \
    --limit-val {limit_val} \
    --val-batches {val_batches} \
    --log-gates-batches {log_gates_batches} \
    --amp
}}

run_ablation base {base_quality_cache} {base_output_dir}
run_ablation base_rdt {base_rdt_quality_cache} {base_rdt_output_dir}
"""


if __name__ == "__main__":
    raise SystemExit(main())
