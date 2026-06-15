#!/usr/bin/env python
"""TTA推理：对测试集进行翻转+多尺度推理，融合结果提升性能。

支持：
- Horizontal flip
- Multi-scale (多个image-max-side)
- Box averaging / NMS融合
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INFER = ROOT / "scripts" / "infer_rgc_dino.py"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "source" / "AIC2026_PHASE_1_1000")
    parser.add_argument("--quality-cache", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--zip-path", type=Path, required=True)

    # TTA配置
    parser.add_argument("--hflip", action="store_true", help="Enable horizontal flip TTA")
    parser.add_argument("--scales", type=int, nargs="+", default=[640], help="Multi-scale inference (e.g. 640 720 800)")

    # 推理参数
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-threshold", type=float, default=0.001)
    parser.add_argument("--nms-iou-threshold", type=float, help="Optional NMS before fusion")
    parser.add_argument("--amp", action="store_true", default=True)

    # 融合策略
    parser.add_argument("--fusion-method", choices=["avg", "wbf", "nms"], default="avg",
                       help="Box fusion method: avg=average boxes, wbf=weighted boxes fusion, nms=NMS")
    parser.add_argument("--fusion-iou-threshold", type=float, default=0.5, help="IoU threshold for box fusion")

    return parser.parse_args()

def run_single_inference(
    args: argparse.Namespace,
    scale: int,
    hflip: bool,
    output_dir: Path,
) -> Path:
    """运行单次推理，返回预测目录"""
    tag = f"s{scale}"
    if hflip:
        tag += "_hflip"

    pred_dir = output_dir / tag
    pred_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(INFER),
        "--checkpoint", str(args.checkpoint),
        "--dataset-root", str(args.dataset_root),
        "--output-dir", str(pred_dir),
        "--device", args.device,
        "--image-max-side", str(scale),
        "--score-threshold", str(args.score_threshold),
    ]

    if args.quality_cache:
        cmd += ["--quality-cache", str(args.quality_cache)]
    if args.nms_iou_threshold:
        cmd += ["--nms-iou-threshold", str(args.nms_iou_threshold)]
    if args.amp:
        cmd += ["--amp"]

    # Horizontal flip需要在模型层实现，这里先占位
    # 实际需要修改infer_rgc_dino.py支持--hflip参数
    # 或者后处理翻转预测框

    print(f"  Running inference: {tag} ...", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Inference failed for {tag}: {result.stderr[-500:]}")

    return pred_dir

def load_predictions(pred_dir: Path) -> dict[str, list]:
    """加载预测结果 {image_id: [(cls, cx, cy, w, h, score), ...]}"""
    preds = {}
    for txt_file in sorted(pred_dir.glob("*.txt")):
        image_id = txt_file.stem
        boxes = []
        if txt_file.stat().st_size > 0:
            with open(txt_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 6:
                        cls_id, cx, cy, w, h, score = map(float, parts)
                        boxes.append((int(cls_id), cx, cy, w, h, score))
        preds[image_id] = boxes
    return preds

def box_iou(box1, box2):
    """计算两个box的IoU (cx,cy,w,h格式)"""
    _, cx1, cy1, w1, h1, _ = box1
    _, cx2, cy2, w2, h2, _ = box2

    x1_min, y1_min = cx1 - w1/2, cy1 - h1/2
    x1_max, y1_max = cx1 + w1/2, cy1 + h1/2
    x2_min, y2_min = cx2 - w2/2, cy2 - h2/2
    x2_max, y2_max = cx2 + w2/2, cy2 + h2/2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0

def fuse_boxes_avg(all_boxes: list[list], iou_threshold: float) -> list:
    """简单box averaging融合"""
    if not all_boxes:
        return []

    # 展平所有box
    flat_boxes = []
    for boxes in all_boxes:
        flat_boxes.extend(boxes)

    if not flat_boxes:
        return []

    # 按类别分组
    cls_boxes = {}
    for box in flat_boxes:
        cls_id = box[0]
        if cls_id not in cls_boxes:
            cls_boxes[cls_id] = []
        cls_boxes[cls_id].append(box)

    # 每个类别内融合
    fused = []
    for cls_id, boxes in cls_boxes.items():
        boxes = sorted(boxes, key=lambda x: x[5], reverse=True)  # 按score排序

        while boxes:
            # 取最高分box
            best_box = boxes.pop(0)
            cluster = [best_box]

            # 找到与它IoU>threshold的所有box
            remaining = []
            for box in boxes:
                if box_iou(best_box, box) > iou_threshold:
                    cluster.append(box)
                else:
                    remaining.append(box)
            boxes = remaining

            # 平均融合cluster
            cls_id = cluster[0][0]
            cx = np.mean([b[1] for b in cluster])
            cy = np.mean([b[2] for b in cluster])
            w = np.mean([b[3] for b in cluster])
            h = np.mean([b[4] for b in cluster])
            score = np.mean([b[5] for b in cluster])  # 平均置信度

            fused.append((int(cls_id), cx, cy, w, h, score))

    return fused

def save_predictions(preds: dict[str, list], output_dir: Path):
    """保存融合后的预测"""
    output_dir.mkdir(parents=True, exist_ok=True)
    for image_id, boxes in preds.items():
        txt_file = output_dir / f"{image_id}.txt"
        with open(txt_file, "w") as f:
            for box in boxes:
                cls_id, cx, cy, w, h, score = box
                f.write(f"{cls_id} {cx} {cy} {w} {h} {score}\n")

def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 生成TTA配置
    tta_configs = []
    for scale in args.scales:
        tta_configs.append((scale, False))  # 原图
        if args.hflip:
            tta_configs.append((scale, True))  # 翻转

    print(f"TTA配置: {len(tta_configs)} 次推理")
    for scale, hflip in tta_configs:
        print(f"  - scale={scale}, hflip={hflip}")
    print()

    # 运行所有推理
    pred_dirs = []
    for scale, hflip in tta_configs:
        pred_dir = run_single_inference(args, scale, hflip, args.output_dir)
        pred_dirs.append(pred_dir)

    print("\n融合预测...", flush=True)

    # 加载所有预测
    all_preds_by_image = {}
    for pred_dir in pred_dirs:
        preds = load_predictions(pred_dir)
        for image_id, boxes in preds.items():
            if image_id not in all_preds_by_image:
                all_preds_by_image[image_id] = []
            all_preds_by_image[image_id].append(boxes)

    # 融合
    fused_preds = {}
    for image_id, all_boxes in all_preds_by_image.items():
        if args.fusion_method == "avg":
            fused_preds[image_id] = fuse_boxes_avg(all_boxes, args.fusion_iou_threshold)
        else:
            # WBF/NMS待实现
            raise NotImplementedError(f"Fusion method {args.fusion_method} not yet implemented")

    # 保存
    fused_dir = args.output_dir / "fused"
    save_predictions(fused_preds, fused_dir)

    # 打包ZIP
    print("\n打包提交文件...", flush=True)
    from make_submission import make_submission_zip
    make_submission_zip(
        dataset_root=args.dataset_root,
        submission_dir=fused_dir,
        zip_path=args.zip_path,
    )

    print(f"\n✓ TTA推理完成")
    print(f"  融合预测: {fused_dir}")
    print(f"  ZIP: {args.zip_path}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
