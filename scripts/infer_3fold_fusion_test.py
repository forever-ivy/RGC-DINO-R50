#!/usr/bin/env python3
"""对测试集进行3-fold融合推理并生成提交ZIP（使用NMS策略）。

要求：
1. 3个fold的checkpoint已准备好
2. 融合策略使用NMS（保留最高分，不用朴素平均）
3. 生成的预测自动打包成提交ZIP
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INFER = ROOT / "scripts" / "infer_rgc_dino.py"
FUSE = ROOT / "scripts" / "fuse_multifold_predictions.py"

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints", nargs=3, type=Path, required=True,
                   help="3个fold的checkpoint路径(fold0, fold1, fold2)")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--zip-path", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, default=ROOT / "source" / "AIC2026_PHASE_1_1000")
    p.add_argument("--quality-cache", type=Path, default=ROOT / "outputs" / "cache" / "quality_features_test.json")
    p.add_argument("--fusion-method", choices=["avg", "vote"], default="avg",
                   help="avg用NMS, vote需要至少min-votes个fold检出")
    p.add_argument("--min-votes", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action="store_true", default=True)
    return p.parse_args()

def infer_fold(ckpt: Path, fold_name: str, output_dir: Path, args) -> Path:
    """对测试集推理"""
    pred_dir = output_dir / fold_name
    pred_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(INFER),
        "--checkpoint", str(ckpt),
        "--dataset-root", str(args.dataset_root),
        "--output-dir", str(pred_dir),
        "--device", args.device,
        "--quality-cache", str(args.quality_cache),
    ]
    if args.amp:
        cmd += ["--amp"]

    print(f"  推理 {fold_name} ...", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{fold_name} 推理失败: {r.stderr[-500:]}")

    return pred_dir

def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("3-fold融合测试集推理（NMS策略）")
    print("=" * 60)

    # 1) 各fold推理
    print("\n1. 各fold测试集推理:")
    fold_names = ["fold0", "fold1", "fold2"]
    pred_dirs = []
    for ckpt, fold_name in zip(args.checkpoints, fold_names):
        pred_dir = infer_fold(ckpt, fold_name, args.output_dir, args)
        pred_dirs.append(pred_dir)

    # 2) NMS融合
    print(f"\n2. 融合预测 (方法={args.fusion_method}, NMS策略):")
    cmd = [
        sys.executable, str(FUSE),
        "--pred-dirs", *[str(d) for d in pred_dirs],
        "--output-dir", str(args.output_dir / "fused"),
        "--zip-path", str(args.zip_path),
        "--dataset-root", str(args.dataset_root),
        "--fusion-method", args.fusion_method,
        "--fusion-iou-threshold", "0.5",
    ]
    if args.fusion_method == "vote":
        cmd += ["--min-votes", str(args.min_votes)]

    print(f"  执行融合: {' '.join(str(c) for c in cmd[-6:])}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"融合失败: {r.stderr[-500:]}")

    print(r.stdout)

    print("\n" + "=" * 60)
    print(f"✓ 3-fold融合完成")
    print(f"  ZIP: {args.zip_path}")
    print(f"  策略: {args.fusion_method} (NMS - 保留最高分)")
    print("=" * 60)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
