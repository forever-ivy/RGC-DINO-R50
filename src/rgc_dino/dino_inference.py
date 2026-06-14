"""Inference conversion helpers for DINO outputs."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor

from .constants import MAX_PREDICTIONS_PER_IMAGE, NUM_CLASSES
from .labels import DetectionLabel
from .metrics import box_iou_xyxy, xywh_to_xyxy


def dino_result_to_detection_labels(
    result: Mapping[str, Tensor],
    *,
    orig_height: int,
    orig_width: int,
    score_threshold: float = 0.05,
    max_detections: int = MAX_PREDICTIONS_PER_IMAGE,
    nms_iou_threshold: float | None = None,
) -> list[DetectionLabel]:
    """Convert one DINO postprocessor result into normalized submission labels."""
    if orig_height <= 0 or orig_width <= 0:
        raise ValueError("original image size must be positive")
    if nms_iou_threshold is not None and not 0.0 < nms_iou_threshold <= 1.0:
        raise ValueError("nms_iou_threshold must be in (0, 1]")
    records: list[DetectionLabel] = []
    scores = result["scores"].detach().cpu()
    labels = result["labels"].detach().cpu()
    boxes = result["boxes"].detach().cpu()

    for score_tensor, label_tensor, box_tensor in zip(scores, labels, boxes):
        confidence = float(score_tensor)
        if confidence < score_threshold:
            continue
        class_id = int(label_tensor)
        if not 0 <= class_id < NUM_CLASSES:
            continue

        x1, y1, x2, y2 = [float(value) for value in box_tensor.tolist()]
        x1 = _clip(x1, 0.0, float(orig_width))
        x2 = _clip(x2, 0.0, float(orig_width))
        y1 = _clip(y1, 0.0, float(orig_height))
        y2 = _clip(y2, 0.0, float(orig_height))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        box_w = x2 - x1
        box_h = y2 - y1
        if box_w <= 0.0 or box_h <= 0.0:
            continue

        records.append(
            DetectionLabel(
                class_id=class_id,
                norm_center_x=_clip((x1 + box_w / 2.0) / float(orig_width), 0.0, 1.0),
                norm_center_y=_clip((y1 + box_h / 2.0) / float(orig_height), 0.0, 1.0),
                norm_w=_clip(box_w / float(orig_width), 1e-12, 1.0),
                norm_h=_clip(box_h / float(orig_height), 1e-12, 1.0),
                confidence=_clip(confidence, 0.0, 1.0),
            )
        )

    if nms_iou_threshold is not None:
        records = _classwise_nms(records, iou_threshold=nms_iou_threshold)
    records.sort(key=lambda record: record.confidence if record.confidence is not None else -1.0, reverse=True)
    return records[:max_detections]


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _classwise_nms(records: list[DetectionLabel], *, iou_threshold: float) -> list[DetectionLabel]:
    kept: list[DetectionLabel] = []
    by_class: dict[int, list[DetectionLabel]] = {}
    for record in records:
        by_class.setdefault(record.class_id, []).append(record)

    for class_records in by_class.values():
        selected: list[DetectionLabel] = []
        for record in sorted(class_records, key=lambda item: item.confidence or 0.0, reverse=True):
            box = xywh_to_xyxy(record)
            if all(box_iou_xyxy(box, xywh_to_xyxy(previous)) < iou_threshold for previous in selected):
                selected.append(record)
        kept.extend(selected)
    return kept
