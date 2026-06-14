#!/usr/bin/env python
"""Write an LSF bsub script for an RGC-DINO short validation run."""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "jobs" / "train_rgc_dino_fold0_short.lsf"
DEFAULT_TRAIN_OUTPUT = ROOT / "outputs" / "rgc_dino" / "rgc_dino_fold0_short_identity_depth_aug"
DEFAULT_QUALITY_CACHE = ROOT / "outputs" / "cache" / "quality_features_train.json"
DEFAULT_OFFICIAL_DINO_CHECKPOINT = ROOT / "outputs" / "checkpoints" / "checkpoint0011_4scale.pth"
DEFAULT_FALLBACK_INIT_CHECKPOINT = ROOT / "outputs" / "checkpoints" / "a0_fold0_best_regular_snapshot_20260614_0131.pth"
DEFAULT_TRAIN_IMAGE_MAX_SIDES = (480, 560, 640, 720)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--job-name", default="rgc-dino-fold0-short")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--queue", default="normal")
    parser.add_argument("--train-output-dir", type=Path, default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr-drop", type=int, default=3)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--image-max-side", type=int, default=640)
    parser.add_argument("--train-image-max-sides", type=int, nargs="+", default=list(DEFAULT_TRAIN_IMAGE_MAX_SIDES))
    parser.add_argument("--official-dino-checkpoint", type=Path, default=DEFAULT_OFFICIAL_DINO_CHECKPOINT)
    parser.add_argument("--fallback-init-checkpoint", type=Path, default=DEFAULT_FALLBACK_INIT_CHECKPOINT)
    parser.add_argument("--quality-cache", type=Path, default=DEFAULT_QUALITY_CACHE)
    parser.add_argument("--random-horizontal-flip-prob", type=float, default=0.5)
    parser.add_argument("--log-gates-batches", type=int, default=2)
    parser.add_argument("--val-batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script = render_bsub_script(
        job_name=args.job_name,
        queue=args.queue,
        gpu=args.gpu,
        output_dir=args.train_output_dir,
        epochs=args.epochs,
        lr_drop=args.lr_drop,
        fold=args.fold,
        image_max_side=args.image_max_side,
        train_image_max_sides=tuple(args.train_image_max_sides),
        official_dino_checkpoint=args.official_dino_checkpoint,
        fallback_init_checkpoint=args.fallback_init_checkpoint,
        quality_cache=args.quality_cache,
        random_horizontal_flip_prob=args.random_horizontal_flip_prob,
        log_gates_batches=args.log_gates_batches,
        val_batches=args.val_batches,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
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
    output_dir: Path,
    epochs: int,
    lr_drop: int,
    fold: int,
    image_max_side: int,
    train_image_max_sides: tuple[int, ...],
    official_dino_checkpoint: Path,
    fallback_init_checkpoint: Path,
    quality_cache: Path,
    random_horizontal_flip_prob: float,
    log_gates_batches: int,
    val_batches: int = 50,
    batch_size: int = 1,
    num_workers: int = 2,
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

python scripts/cache_quality_features.py \\
  --output {quality_cache} \\
  --max-side-for-quality {image_max_side} \\
  --num-workers {num_workers}

INIT_DINO_CHECKPOINT="{official_dino_checkpoint}"
if [ ! -f "$INIT_DINO_CHECKPOINT" ]; then
  INIT_DINO_CHECKPOINT="{fallback_init_checkpoint}"
fi
if [ ! -f "$INIT_DINO_CHECKPOINT" ]; then
  echo "missing init checkpoint: $INIT_DINO_CHECKPOINT" >&2
  exit 2
fi

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/train_rgc_dino.py \\
  --output-dir {output_dir} \\
  --fold {fold} \\
  --epochs {epochs} \\
  --lr-drop {lr_drop} \\
  --batch-size {batch_size} \\
  --num-workers {num_workers} \\
  --image-max-side {image_max_side} \\
  --train-image-max-sides {train_scales} \\
  --init-dino-checkpoint "$INIT_DINO_CHECKPOINT" \\
  --quality-cache {quality_cache} \\
  --random-horizontal-flip-prob {random_horizontal_flip_prob} \\
  --val-batches {val_batches} \\
  --log-gates-batches {log_gates_batches} \\
  --amp
"""


if __name__ == "__main__":
    raise SystemExit(main())
