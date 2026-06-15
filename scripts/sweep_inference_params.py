#!/usr/bin/env python
"""Sweep inference hyperparameters (threshold, NMS, TTA) on validation set to find optimal config.

快速扫描推理超参，用验证集 mAP 筛选最优配置，然后对测试集推理提交。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
INFER = ROOT / "scripts" / "infer_rgc_dino.py"
EVAL = ROOT / "scripts" / "evaluate_predictions.py"

@dataclass
class SweepConfig:
    """一组推理超参配置"""
    score_threshold: float
    nms_iou_threshold: Optional[float] = None
    tta_hflip: bool = False
    tta_scales: Optional[list[int]] = None

    def tag(self) -> str:
        """生成唯一标签"""
        parts = [f"th{self.score_threshold:.4f}".replace(".", "")]
        if self.nms_iou_threshold is not None:
            parts.append(f"nms{int(self.nms_iou_threshold*100):02d}")
        if self.tta_hflip:
            parts.append("hflip")
        if self.tta_scales:
            parts.append(f"ms{'_'.join(map(str, self.tta_scales))}")
        return "_".join(parts)

@dataclass
class SweepResult:
    config: SweepConfig
    val_map: float
    val_map50: float
    pred_count: int
    pred_dir: str
    error: Optional[str] = None

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--val-ids", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "source" / "训练集")
    parser.add_argument("--labels", type=Path, default=ROOT / "source" / "训练集" / "labels")
    parser.add_argument("--quality-cache", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "sweep_inference")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true", default=True)

    # Sweep范围
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.001, 0.003, 0.01, 0.03, 0.05, 0.1])
    parser.add_argument("--nms-iou", type=float, nargs="+", help="NMS IoU thresholds to try (e.g. 0.5 0.6 0.7)")
    parser.add_argument("--tta-hflip", action="store_true", help="Enable horizontal flip TTA")
    parser.add_argument("--tta-scales", type=int, nargs="+", help="Multi-scale TTA (e.g. 640 720 800)")

    parser.add_argument("--top-k", type=int, default=5, help="只报告Top-K结果")
    return parser.parse_args()

def run_inference(args: argparse.Namespace, config: SweepConfig, pred_dir: Path) -> Optional[int]:
    """运行推理，返回预测目标数"""
    cmd = [
        sys.executable, str(INFER),
        "--checkpoint", str(args.checkpoint),
        "--dataset-root", str(args.dataset_root),
        "--sample-ids-file", str(args.val_ids),
        "--output-dir", str(pred_dir),
        "--device", args.device,
        "--score-threshold", str(config.score_threshold),
        "--image-max-side", "640",
    ]
    if config.nms_iou_threshold is not None:
        cmd += ["--nms-iou-threshold", str(config.nms_iou_threshold)]
    if args.quality_cache:
        cmd += ["--quality-cache", str(args.quality_cache)]
    if args.amp:
        cmd += ["--amp"]

    # TTA在infer_rgc_dino.py中未实现，这里先占位
    # 实际需要修改infer脚本或在这里做多次推理+融合

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Inference failed: {result.stderr[-500:]}")

    # 从stdout提取预测目标数
    for line in result.stdout.splitlines():
        if "prediction_objects" in line:
            try:
                return json.loads(line)["prediction_objects"]
            except:
                pass
    return None

def run_eval(args: argparse.Namespace, pred_dir: Path) -> tuple[float, float]:
    """运行评估，返回(mAP@50:95, mAP@50)"""
    cmd = [
        sys.executable, str(EVAL),
        "--labels", str(args.labels),
        "--predictions", str(pred_dir),
        "--sample-ids-file", str(args.val_ids),
        "--clip-labels",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Eval failed: {result.stderr[-500:]}")

    # 解析mAP
    import re
    map_match = re.search(r"mAP@50:95:\s*([0-9.]+)", result.stdout)
    map50_match = re.search(r"mAP@50:\s*([0-9.]+)", result.stdout)
    if not map_match:
        raise RuntimeError(f"Cannot parse mAP from: {result.stdout[-300:]}")
    return float(map_match.group(1)), float(map50_match.group(1)) if map50_match else 0.0

def generate_sweep_configs(args: argparse.Namespace) -> list[SweepConfig]:
    """生成扫描配置列表"""
    configs = []

    # 阶段1: threshold扫描（无NMS）
    for th in args.thresholds:
        configs.append(SweepConfig(score_threshold=th))

    # 阶段2: 如果指定了NMS，在最优threshold基础上扫NMS
    # 这里先全组合，后续可优化为两阶段
    if args.nms_iou:
        for th in args.thresholds:
            for nms in args.nms_iou:
                configs.append(SweepConfig(score_threshold=th, nms_iou_threshold=nms))

    # 阶段3: TTA (先占位，实际需要修改infer脚本)
    # if args.tta_hflip or args.tta_scales:
    #     configs.append(SweepConfig(..., tta_hflip=True))

    return configs

def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    configs = generate_sweep_configs(args)
    print(f"扫描 {len(configs)} 组配置...")
    print(f"Checkpoint: {args.checkpoint.name}")
    print(f"Val IDs: {args.val_ids.name}")
    print()

    results: list[SweepResult] = []

    for i, config in enumerate(configs, 1):
        tag = config.tag()
        pred_dir = args.output_dir / f"sweep_{tag}"
        pred_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{i}/{len(configs)}] {tag} ... ", end="", flush=True)

        try:
            pred_count = run_inference(args, config, pred_dir)
            val_map, val_map50 = run_eval(args, pred_dir)
            results.append(SweepResult(config, val_map, val_map50, pred_count or 0, str(pred_dir)))
            print(f"mAP={val_map:.4f}  mAP50={val_map50:.4f}  preds={pred_count}")
        except Exception as e:
            results.append(SweepResult(config, 0.0, 0.0, 0, str(pred_dir), error=str(e)))
            print(f"FAILED: {e}")

    # 排序并保存
    results_sorted = sorted(results, key=lambda r: r.val_map, reverse=True)

    ranking_path = args.output_dir / "sweep_ranking.json"
    ranking_path.write_text(
        json.dumps([asdict(r) for r in results_sorted], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print(f"TOP-{args.top_k} 配置 (by mAP@50:95)")
    print("=" * 70)
    print(f"{'rank':<5}{'mAP@50:95':<12}{'mAP@50':<10}{'preds':<8}{'config':<40}")
    for i, r in enumerate(results_sorted[:args.top_k], 1):
        if r.error:
            print(f"{i:<5}{'FAILED':<12}{'-':<10}{'-':<8}{r.config.tag():<40}")
        else:
            print(f"{i:<5}{r.val_map:<12.4f}{r.val_map50:<10.4f}{r.pred_count:<8}{r.config.tag():<40}")

    best = results_sorted[0]
    if not best.error:
        print(f"\n>>> BEST: {best.config.tag()}")
        print(f">>>   mAP@50:95={best.val_map:.4f}, mAP@50={best.val_map50:.4f}")
        print(f">>>   threshold={best.config.score_threshold}, NMS={best.config.nms_iou_threshold}")
        print(f">>>   predictions: {best.pred_dir}")

    print(f"\n>>> 完整排名: {ranking_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
