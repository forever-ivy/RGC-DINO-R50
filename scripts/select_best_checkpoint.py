#!/usr/bin/env python
"""Evaluate every RGC-DINO checkpoint in a training run on the fold validation set
and rank them by mAP@50:95.

This orchestrates the existing ``infer_rgc_dino.py`` (prediction) and
``evaluate_predictions.py`` (metric) scripts per checkpoint, so detection and
metric logic are never duplicated here. Use it after a training run finishes to
pick the checkpoint to submit, rather than blindly trusting the last epoch.

Example:
    python scripts/select_best_checkpoint.py \\
      --train-dir outputs/rgc_dino/rgc_dino_fold0_cocopretrain_12ep_20260615_1311 \\
      --val-ids outputs/splits/fold0_val_ids.txt \\
      --quality-cache outputs/cache/quality_features_train.json \\
      --device cuda
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
INFER = ROOT / "scripts" / "infer_rgc_dino.py"
EVAL = ROOT / "scripts" / "evaluate_predictions.py"

# Validation predictions are produced from the labeled training tree, restricted
# to the fold's held-out ids, then scored against the same ids' labels.
DEFAULT_DATASET_ROOT = ROOT / "source" / "训练集"
DEFAULT_LABELS = ROOT / "source" / "训练集" / "labels"

_MAP_RE = re.compile(r"^mAP@50:95:\s*([0-9.]+)\s*$", re.MULTILINE)
_MAP50_RE = re.compile(r"^mAP@50:\s*([0-9.]+)\s*$", re.MULTILINE)
_PREDOBJ_RE = re.compile(r'"prediction_objects":\s*(\d+)')


@dataclass
class CheckpointResult:
    checkpoint: str
    epoch: Optional[int]
    map_50_95: Optional[float]
    map_50: Optional[float]
    prediction_objects: Optional[int]
    error: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-dir", type=Path, required=True, help="training run dir containing checkpoint*.pth")
    parser.add_argument("--val-ids", type=Path, default=ROOT / "outputs" / "splits" / "fold0_val_ids.txt")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--config-file", type=Path, default=ROOT / "configs" / "dino_a0_rgb_4scale.py")
    parser.add_argument("--quality-cache", type=Path, help="sample_id -> quality feature cache JSON")
    parser.add_argument("--output-dir", type=Path, help="where to write per-checkpoint predictions + ranking (default: <train-dir>/val_eval)")
    parser.add_argument("--epochs", type=int, nargs="+", help="only evaluate these epoch numbers (default: all found)")
    parser.add_argument("--min-epoch", type=int, help="skip checkpoints with epoch < this (e.g. skip warmup epochs)")
    parser.add_argument("--include-latest", action="store_true", help="also evaluate bare checkpoint.pth (no epoch tag)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-max-side", type=int, default=640)
    parser.add_argument("--side-base-channels", type=int, default=32)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--nms-iou-threshold", type=float, help="optional classwise NMS IoU threshold")
    parser.add_argument("--clip-labels", action="store_true", default=True, help="clip minor label drift (sample data has 4 OOB boxes)")
    parser.add_argument("--no-clip-labels", dest="clip_labels", action="store_false")
    parser.add_argument("--limit", type=int, help="limit val images per checkpoint (debug)")
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def discover_checkpoints(train_dir: Path, *, epochs: Optional[list[int]], min_epoch: Optional[int], include_latest: bool) -> list[tuple[Optional[int], Path]]:
    """Return [(epoch, path)] sorted by epoch; epoch is None for bare checkpoint.pth."""
    found: list[tuple[Optional[int], Path]] = []
    for path in sorted(train_dir.glob("checkpoint[0-9][0-9][0-9][0-9].pth")):
        m = re.search(r"checkpoint(\d+)\.pth$", path.name)
        if not m:
            continue
        ep = int(m.group(1))
        if epochs is not None and ep not in epochs:
            continue
        if min_epoch is not None and ep < min_epoch:
            continue
        found.append((ep, path))
    if include_latest:
        bare = train_dir / "checkpoint.pth"
        if bare.exists():
            found.append((None, bare))
    return found


def run_inference(args: argparse.Namespace, checkpoint: Path, pred_dir: Path) -> None:
    cmd = [
        sys.executable, str(INFER),
        "--config-file", str(args.config_file),
        "--dataset-root", str(args.dataset_root),
        "--checkpoint", str(checkpoint),
        "--model-mode", "rgc",
        "--checkpoint-scope", "auto",
        "--output-dir", str(pred_dir),
        "--device", args.device,
        "--image-max-side", str(args.image_max_side),
        "--side-base-channels", str(args.side_base_channels),
        "--score-threshold", str(args.score_threshold),
        "--max-detections", str(args.max_detections),
        "--sample-ids-file", str(args.val_ids),
    ]
    if args.quality_cache is not None:
        cmd += ["--quality-cache", str(args.quality_cache)]
    if args.nms_iou_threshold is not None:
        cmd += ["--nms-iou-threshold", str(args.nms_iou_threshold)]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    if args.amp:
        cmd += ["--amp"]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"inference failed (rc={out.returncode}): {out.stderr.strip()[-500:]}")
    return out.stdout


def run_eval(args: argparse.Namespace, pred_dir: Path) -> tuple[float, float]:
    cmd = [
        sys.executable, str(EVAL),
        "--labels", str(args.labels),
        "--predictions", str(pred_dir),
        "--sample-ids-file", str(args.val_ids),
    ]
    if args.clip_labels:
        cmd += ["--clip-labels"]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"eval failed (rc={out.returncode}): {out.stderr.strip()[-500:]}")
    text = out.stdout
    m = _MAP_RE.search(text)
    m50 = _MAP50_RE.search(text)
    if not m:
        raise RuntimeError(f"could not parse mAP from eval output: {text[-300:]}")
    return float(m.group(1)), (float(m50.group(1)) if m50 else float("nan"))


def main() -> int:
    args = parse_args()
    if not args.train_dir.exists():
        print(f"train dir not found: {args.train_dir}", file=sys.stderr)
        return 2
    if not args.val_ids.exists():
        print(f"val ids file not found: {args.val_ids}", file=sys.stderr)
        return 2

    checkpoints = discover_checkpoints(
        args.train_dir, epochs=args.epochs, min_epoch=args.min_epoch, include_latest=args.include_latest
    )
    if not checkpoints:
        print(f"no checkpoints found in {args.train_dir}", file=sys.stderr)
        return 2

    output_dir = args.output_dir or (args.train_dir / "val_eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating {len(checkpoints)} checkpoint(s) on {args.val_ids.name} "
          f"(score>={args.score_threshold}, nms={args.nms_iou_threshold})\n", flush=True)

    results: list[CheckpointResult] = []
    for ep, ckpt in checkpoints:
        tag = f"ep{ep:04d}" if ep is not None else "latest"
        pred_dir = output_dir / f"{tag}_predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{tag}] {ckpt.name} ... ", end="", flush=True)
        try:
            infer_out = run_inference(args, ckpt, pred_dir)
            pred_obj_m = _PREDOBJ_RE.search(infer_out or "")
            pred_obj = int(pred_obj_m.group(1)) if pred_obj_m else None
            map_val, map50 = run_eval(args, pred_dir)
            results.append(CheckpointResult(ckpt.name, ep, map_val, map50, pred_obj))
            print(f"mAP@50:95={map_val:.4f}  mAP@50={map50:.4f}  preds={pred_obj}", flush=True)
        except Exception as exc:  # noqa: BLE001 - record and continue to next checkpoint
            results.append(CheckpointResult(ckpt.name, ep, None, None, None, error=str(exc)))
            print(f"FAILED: {exc}", flush=True)

    ranked = sorted(
        results,
        key=lambda r: (r.map_50_95 is not None, r.map_50_95 or -1.0),
        reverse=True,
    )

    ranking_path = output_dir / "ranking.json"
    ranking_path.write_text(
        json.dumps([asdict(r) for r in ranked], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 64)
    print(f"RANKING (by mAP@50:95) — {args.val_ids.name}")
    print("=" * 64)
    print(f"{'rank':<5}{'epoch':<7}{'mAP@50:95':<12}{'mAP@50':<10}{'preds':<8}")
    for i, r in enumerate(ranked, 1):
        if r.map_50_95 is None:
            print(f"{i:<5}{str(r.epoch):<7}{'FAILED':<12}{'-':<10}{'-':<8}  ({r.error[:60] if r.error else ''})")
        else:
            print(f"{i:<5}{str(r.epoch):<7}{r.map_50_95:<12.4f}{r.map_50:<10.4f}{str(r.prediction_objects):<8}")
    best = next((r for r in ranked if r.map_50_95 is not None), None)
    if best is not None:
        print("\n>>> BEST:", best.checkpoint, f"(epoch {best.epoch}, mAP@50:95={best.map_50_95:.4f})")
        print(f">>> predictions dir: {output_dir / (('ep%04d' % best.epoch) if best.epoch is not None else 'latest')}_predictions")
    print(f">>> ranking saved: {ranking_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
