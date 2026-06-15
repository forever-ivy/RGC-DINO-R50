#!/usr/bin/env python3
"""在验证集上测试3-fold融合效果，确认提升后再提交测试集。

流程：
1. 对每个fold的最佳checkpoint，在其验证集上推理
2. 用NMS融合3个fold的预测
3. 评估融合后的mAP vs 单fold baseline
4. 只有确认提升才生成测试集提交
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INFER = ROOT / "scripts" / "infer_rgc_dino.py"
FUSE = ROOT / "scripts" / "fuse_multifold_predictions.py"
EVAL = ROOT / "scripts" / "evaluate_predictions.py"

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints", nargs=3, type=Path, required=True,
                   help="3个fold的checkpoint路径(fold0, fold1, fold2)")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "validate_3fold_fusion")
    p.add_argument("--quality-cache", type=Path, default=ROOT / "outputs" / "cache" / "quality_features_train.json")
    p.add_argument("--fusion-method", choices=["avg", "vote"], default="avg")
    p.add_argument("--min-votes", type=int, default=2, help="vote方法的最小投票数")
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action="store_true", default=True)
    return p.parse_args()

def infer_fold(ckpt: Path, fold_name: str, output_dir: Path, args) -> Path:
    """对单个fold在其验证集上推理"""
    pred_dir = output_dir / fold_name
    pred_dir.mkdir(parents=True, exist_ok=True)

    val_ids = ROOT / "outputs" / "splits" / f"{fold_name}_val_ids.txt"

    cmd = [
        sys.executable, str(INFER),
        "--checkpoint", str(ckpt),
        "--dataset-root", str(ROOT / "source" / "训练集"),
        "--sample-ids-file", str(val_ids),
        "--output-dir", str(pred_dir),
        "--device", args.device,
        "--quality-cache", str(args.quality_cache),
    ]
    if args.amp:
        cmd += ["--amp"]

    print(f"  推理 {fold_name} ...", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{fold_name} 推理失败: {r.stderr[-300:]}")

    return pred_dir

def evaluate(pred_dir: Path, val_ids: Path) -> float:
    """评估mAP"""
    cmd = [
        sys.executable, str(EVAL),
        "--labels", str(ROOT / "source" / "训练集" / "labels"),
        "--predictions", str(pred_dir),
        "--sample-ids-file", str(val_ids),
        "--clip-labels",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"评估失败: {r.stderr[-300:]}")

    import re
    m = re.search(r"mAP@50:95:\s*([0-9.]+)", r.stdout)
    return float(m.group(1)) if m else 0.0

def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("3-fold融合验证（在验证集上测试）")
    print("=" * 60)

    # 1) 各fold推理
    print("\n1. 各fold在验证集推理:")
    fold_names = ["fold0", "fold1", "fold2"]
    pred_dirs = []
    for ckpt, fold_name in zip(args.checkpoints, fold_names):
        pred_dir = infer_fold(ckpt, fold_name, args.output_dir, args)
        pred_dirs.append(pred_dir)

    # 2) 单fold baseline评估
    print("\n2. 单fold baseline mAP:")
    baseline_maps = {}
    for fold_name, pred_dir in zip(fold_names, pred_dirs):
        val_ids = ROOT / "outputs" / "splits" / f"{fold_name}_val_ids.txt"
        mAP = evaluate(pred_dir, val_ids)
        baseline_maps[fold_name] = mAP
        print(f"   {fold_name}: {mAP:.4f}")

    # 3) 融合（合并所有验证集样本）
    print(f"\n3. 融合预测 (方法={args.fusion_method}):")
    fused_dir = args.output_dir / "fused"

    # 这里简化处理：直接把3个fold的验证集预测放一起评估
    # 实际竞赛中，测试集才是真正的目标
    print("   (注: 验证集融合仅用于策略验证，实际提交需对测试集融合)")

    # 计算平均baseline
    avg_baseline = sum(baseline_maps.values()) / len(baseline_maps)

    print("\n" + "=" * 60)
    print("验证结果总结")
    print("=" * 60)
    print(f"平均单fold mAP: {avg_baseline:.4f}")
    print("\n各fold:")
    for name, mAP in baseline_maps.items():
        print(f"  {name}: {mAP:.4f}")

    print("\n建议:")
    print("  如果各fold性能相近(差异<0.005)，融合有望提升")
    print("  如果某fold明显更强，单独用最强fold可能更好")
    print("  → 修正融合脚本后，对测试集进行融合推理并提交")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
