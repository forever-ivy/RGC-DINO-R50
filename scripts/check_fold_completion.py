#!/usr/bin/env python3
"""Fold训练完成后自动评估最佳checkpoint并准备融合。

当fold1/fold2训练完成时，自动：
1. 运行 select_best_checkpoint.py 找最佳epoch
2. 报告验证集mAP
3. 准备融合所需的3个checkpoint路径
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def check_fold_complete(fold_dir: Path, expected_epochs: int = 12) -> tuple[bool, int]:
    """检查fold是否训练完成，返回(是否完成, 当前epoch数)"""
    checkpoints = list(fold_dir.glob("checkpoint[0-9]*.pth"))
    if not checkpoints:
        return False, 0

    epochs = [int(p.stem.replace("checkpoint", "").lstrip("0") or "0") for p in checkpoints]
    max_epoch = max(epochs) if epochs else 0
    return max_epoch >= expected_epochs - 1, max_epoch

def evaluate_best_checkpoint(fold_dir: Path, fold_name: str) -> dict:
    """评估fold的最佳checkpoint"""
    print(f"\n{'='*60}")
    print(f"评估 {fold_name}")
    print('='*60)

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "select_best_checkpoint.py"),
        "--checkpoint-dir", str(fold_dir),
        "--val-ids", str(ROOT / "outputs" / "splits" / f"{fold_name}_val_ids.txt"),
        "--quality-cache", str(ROOT / "outputs" / "cache" / "quality_features_train.json"),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"⚠ 评估失败: {result.stderr[-300:]}")
        return {"error": True}

    print(result.stdout)

    # 解析最佳checkpoint（简化版，实际需要更健壮的解析）
    lines = result.stdout.splitlines()
    best_ckpt = None
    best_map = None
    for line in lines:
        if "Best checkpoint" in line or "最佳" in line:
            best_ckpt = line
        if "mAP" in line and ":" in line:
            try:
                best_map = float(line.split(":")[-1].strip())
            except:
                pass

    return {
        "error": False,
        "best_checkpoint": best_ckpt,
        "best_map": best_map,
        "fold_dir": str(fold_dir),
    }

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-epochs", type=int, default=12)
    args = parser.parse_args()

    fold_dirs = {
        "fold0": ROOT / "outputs" / "rgc_dino" / "rgc_dino_fold0_cocopretrain_12ep_20260615_1311",
        "fold1": ROOT / "outputs" / "rgc_dino" / "rgc_dino_fold1_cocopretrain_12ep_20260615_1850",
        "fold2": ROOT / "outputs" / "rgc_dino" / "rgc_dino_fold2_cocopretrain_12ep_20260615_1850",
    }

    print("检查fold训练状态...")
    results = {}

    for fold_name, fold_dir in fold_dirs.items():
        if not fold_dir.exists():
            print(f"  {fold_name}: 目录不存在")
            continue

        complete, epoch = check_fold_complete(fold_dir, args.expected_epochs)
        status = "✓ 完成" if complete else f"训练中 (epoch {epoch}/{args.expected_epochs})"
        print(f"  {fold_name}: {status}")

        if complete:
            results[fold_name] = evaluate_best_checkpoint(fold_dir, fold_name)
        else:
            results[fold_name] = {"complete": False, "epoch": epoch}

    # 汇总报告
    print("\n" + "="*60)
    print("汇总报告")
    print("="*60)

    completed = [k for k, v in results.items() if not v.get("error") and v.get("best_map")]
    print(f"\n完成训练的fold: {len(completed)}/3")

    for fold_name in completed:
        r = results[fold_name]
        print(f"\n{fold_name}:")
        print(f"  最佳mAP: {r.get('best_map', '?'):.4f}")
        print(f"  {r.get('best_checkpoint', 'N/A')}")

    if len(completed) >= 2:
        print("\n✓ 至少2个fold完成，可以进行多fold融合")
        print("→ 使用 scripts/fuse_multifold_predictions.py")
    else:
        print(f"\n→ 等待更多fold完成 ({len(completed)}/2)")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
