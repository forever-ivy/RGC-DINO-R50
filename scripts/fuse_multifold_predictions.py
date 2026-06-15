#!/usr/bin/env python
"""多fold模型预测融合。

支持：
- 多个checkpoint的预测融合（box averaging / WBF / voting）
- 适用于3-fold交叉验证模型集成
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional
import numpy as np

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dirs", type=Path, nargs="+", required=True,
                       help="多个预测目录（每个fold一个）")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--zip-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("source/AIC2026_PHASE_1_1000"))

    # 融合策略
    parser.add_argument("--fusion-method", choices=["avg", "vote", "wbf"], default="avg",
                       help="avg=average boxes, vote=majority vote, wbf=weighted boxes fusion")
    parser.add_argument("--fusion-iou-threshold", type=float, default=0.5,
                       help="IoU threshold for matching boxes across folds")
    parser.add_argument("--min-votes", type=int, default=2,
                       help="For vote method: minimum folds that must predict a box")

    return parser.parse_args()

def load_predictions(pred_dir: Path) -> dict[str, list]:
    """加载预测 {image_id: [(cls, cx, cy, w, h, score), ...]}"""
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
    """计算IoU (cx,cy,w,h格式)"""
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

def fuse_boxes_avg(all_boxes_from_folds: list[list], iou_threshold: float) -> list:
    """Box averaging融合多个fold的预测"""
    if not all_boxes_from_folds:
        return []

    # 展平所有fold的box
    flat_boxes = []
    for boxes in all_boxes_from_folds:
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
        boxes = sorted(boxes, key=lambda x: x[5], reverse=True)

        while boxes:
            best_box = boxes.pop(0)
            cluster = [best_box]

            remaining = []
            for box in boxes:
                if box_iou(best_box, box) > iou_threshold:
                    cluster.append(box)
                else:
                    remaining.append(box)
            boxes = remaining

            # NMS融合：保留最高分box，坐标加权平均（权重=score）
            cls_id = cluster[0][0]
            weights = np.array([b[5] for b in cluster])
            weights = weights / weights.sum()  # 归一化
            cx = np.average([b[1] for b in cluster], weights=weights)
            cy = np.average([b[2] for b in cluster], weights=weights)
            w = np.average([b[3] for b in cluster], weights=weights)
            h = np.average([b[4] for b in cluster], weights=weights)
            score = max([b[5] for b in cluster])  # 保留最高分，不降低置信度

            fused.append((int(cls_id), cx, cy, w, h, score))

    return fused

def fuse_boxes_vote(all_boxes_from_folds: list[list], iou_threshold: float, min_votes: int) -> list:
    """投票融合：只保留至少min_votes个fold都预测的box"""
    # 简化实现：先用avg聚类，然后只保留cluster size >= min_votes的
    if not all_boxes_from_folds:
        return []

    flat_boxes = []
    for boxes in all_boxes_from_folds:
        flat_boxes.extend(boxes)

    if not flat_boxes:
        return []

    cls_boxes = {}
    for box in flat_boxes:
        cls_id = box[0]
        if cls_id not in cls_boxes:
            cls_boxes[cls_id] = []
        cls_boxes[cls_id].append(box)

    fused = []
    for cls_id, boxes in cls_boxes.items():
        boxes = sorted(boxes, key=lambda x: x[5], reverse=True)

        while boxes:
            best_box = boxes.pop(0)
            cluster = [best_box]

            remaining = []
            for box in boxes:
                if box_iou(best_box, box) > iou_threshold:
                    cluster.append(box)
                else:
                    remaining.append(box)
            boxes = remaining

            # 只保留得票数>=min_votes的cluster
            if len(cluster) >= min_votes:
                cls_id = cluster[0][0]
                weights = np.array([b[5] for b in cluster])
                weights = weights / weights.sum()
                cx = np.average([b[1] for b in cluster], weights=weights)
                cy = np.average([b[2] for b in cluster], weights=weights)
                w = np.average([b[3] for b in cluster], weights=weights)
                h = np.average([b[4] for b in cluster], weights=weights)
                score = max([b[5] for b in cluster])  # 保留最高分
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

    print(f"融合 {len(args.pred_dirs)} 个fold的预测")
    print(f"方法: {args.fusion_method}")
    print(f"IoU阈值: {args.fusion_iou_threshold}")
    if args.fusion_method == "vote":
        print(f"最小投票数: {args.min_votes}")
    print()

    # 加载所有fold的预测
    all_fold_preds = []
    for i, pred_dir in enumerate(args.pred_dirs):
        print(f"加载 fold{i}: {pred_dir.name} ...", flush=True)
        preds = load_predictions(pred_dir)
        all_fold_preds.append(preds)

    # 获取所有image_id
    all_image_ids = set()
    for preds in all_fold_preds:
        all_image_ids.update(preds.keys())

    print(f"\n融合 {len(all_image_ids)} 张图片的预测...", flush=True)

    # 逐图融合
    fused_preds = {}
    for image_id in sorted(all_image_ids):
        # 收集该图片在所有fold的预测
        boxes_from_folds = []
        for preds in all_fold_preds:
            if image_id in preds:
                boxes_from_folds.append(preds[image_id])
            else:
                boxes_from_folds.append([])

        # 融合
        if args.fusion_method == "avg":
            fused_boxes = fuse_boxes_avg(boxes_from_folds, args.fusion_iou_threshold)
        elif args.fusion_method == "vote":
            fused_boxes = fuse_boxes_vote(boxes_from_folds, args.fusion_iou_threshold, args.min_votes)
        else:
            raise NotImplementedError(f"Fusion method {args.fusion_method}")

        fused_preds[image_id] = fused_boxes

    # 保存
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_predictions(fused_preds, args.output_dir)

    # 打包
    print("\n打包提交文件...", flush=True)
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from make_submission import make_submission_zip

    make_submission_zip(
        dataset_root=args.dataset_root,
        submission_dir=args.output_dir,
        zip_path=args.zip_path,
    )

    total_boxes = sum(len(boxes) for boxes in fused_preds.values())
    print(f"\n✓ 多fold融合完成")
    print(f"  融合预测: {args.output_dir}")
    print(f"  总预测框: {total_boxes}")
    print(f"  ZIP: {args.zip_path}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
