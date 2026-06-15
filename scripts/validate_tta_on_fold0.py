#!/usr/bin/env python
"""在 fold0 验证集上实测多尺度 TTA 融合的 mAP，避免盲目占用提交次数。

复用 infer_rgc_dino_tta.py 的融合逻辑(fuse_boxes_avg)，但在验证集上跑(带 --sample-ids-file)，
最后用 evaluate_predictions.py 对比单尺度 baseline 与 TTA 融合的真实 mAP。
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INFER = ROOT / "scripts" / "infer_rgc_dino.py"
EVAL = ROOT / "scripts" / "evaluate_predictions.py"

sys.path.insert(0, str(ROOT / "scripts"))
from infer_rgc_dino_tta import load_predictions, fuse_boxes_avg, save_predictions  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--val-ids", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, default=ROOT / "source" / "训练集")
    p.add_argument("--labels", type=Path, default=ROOT / "source" / "训练集" / "labels")
    p.add_argument("--quality-cache", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--scales", type=int, nargs="+", default=[640, 720, 800])
    p.add_argument("--score-threshold", type=float, default=0.001)
    p.add_argument("--fusion-iou-threshold", type=float, default=0.5)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action="store_true", default=True)
    return p.parse_args()


def run_inference(args, scale: int, pred_dir: Path) -> None:
    pred_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(INFER),
        "--checkpoint", str(args.checkpoint),
        "--dataset-root", str(args.dataset_root),
        "--sample-ids-file", str(args.val_ids),
        "--output-dir", str(pred_dir),
        "--device", args.device,
        "--image-max-side", str(scale),
        "--score-threshold", str(args.score_threshold),
    ]
    if args.quality_cache:
        cmd += ["--quality-cache", str(args.quality_cache)]
    if args.amp:
        cmd += ["--amp"]
    print(f"  inference s{scale} ...", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"inference s{scale} failed: {r.stderr[-500:]}")


def evaluate(args, pred_dir: Path) -> tuple[float, float]:
    cmd = [
        sys.executable, str(EVAL),
        "--labels", str(args.labels),
        "--predictions", str(pred_dir),
        "--sample-ids-file", str(args.val_ids),
        "--clip-labels",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"eval failed: {r.stderr[-500:]}")
    m = re.search(r"mAP@50:95:\s*([0-9.]+)", r.stdout)
    m50 = re.search(r"mAP@50:\s*([0-9.]+)", r.stdout)
    if not m:
        raise RuntimeError(f"cannot parse mAP from: {r.stdout[-300:]}")
    return float(m.group(1)), float(m50.group(1)) if m50 else 0.0


def cap_top_k(preds: dict[str, list], k: int) -> dict[str, list]:
    capped = {}
    for image_id, boxes in preds.items():
        boxes_sorted = sorted(boxes, key=lambda b: b[5], reverse=True)
        capped[image_id] = boxes_sorted[:k]
    return capped


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1) 各尺度推理
    scale_dirs = {}
    for scale in args.scales:
        d = args.output_dir / f"s{scale}"
        run_inference(args, scale, d)
        scale_dirs[scale] = d

    # 2) 单尺度 baseline 评估(取首个尺度,通常640,对应已提交基线)
    print("\n=== 单尺度 baseline ===", flush=True)
    baseline_results = {}
    for scale in args.scales:
        mp, mp50 = evaluate(args, scale_dirs[scale])
        baseline_results[scale] = (mp, mp50)
        print(f"  s{scale}: mAP={mp:.4f}  mAP50={mp50:.4f}")

    # 3) TTA 融合
    print("\n=== TTA 融合 (avg) ===", flush=True)
    all_preds_by_image: dict[str, list] = {}
    for scale in args.scales:
        preds = load_predictions(scale_dirs[scale])
        for image_id, boxes in preds.items():
            all_preds_by_image.setdefault(image_id, []).append(boxes)

    fused = {}
    for image_id, all_boxes in all_preds_by_image.items():
        fused[image_id] = fuse_boxes_avg(all_boxes, args.fusion_iou_threshold)
    fused = cap_top_k(fused, args.top_k)

    fused_dir = args.output_dir / "fused"
    save_predictions(fused, fused_dir)
    tta_map, tta_map50 = evaluate(args, fused_dir)

    # 4) 对比报告
    best_single = max(baseline_results.values(), key=lambda x: x[0])
    print("\n" + "=" * 60)
    print("TTA 验证结果对比")
    print("=" * 60)
    for scale in args.scales:
        mp, mp50 = baseline_results[scale]
        print(f"  单尺度 s{scale:<4}: mAP={mp:.4f}  mAP50={mp50:.4f}")
    print(f"  最佳单尺度    : mAP={best_single[0]:.4f}")
    print(f"  TTA 融合      : mAP={tta_map:.4f}  mAP50={tta_map50:.4f}")
    delta = tta_map - best_single[0]
    print("-" * 60)
    print(f"  TTA 相对最佳单尺度: {'+' if delta >= 0 else ''}{delta:.4f}")
    if delta > 0:
        print("  => TTA 有提升, 值得提交")
    else:
        print("  => TTA 无提升, 不建议占用提交次数")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
