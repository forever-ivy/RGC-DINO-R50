"""CPU-safe object detection metrics for YOLO-style normalized boxes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .constants import CLASS_NAMES, NUM_CLASSES
from .labels import DetectionLabel

BoxXYXY = tuple[float, float, float, float]


@dataclass(frozen=True)
class ClassAP:
    class_id: int
    iou_threshold: float
    ap: float
    ground_truth_count: int
    prediction_count: int


@dataclass(frozen=True)
class MapResult:
    map: float
    map50: float
    ground_truth_count: int
    prediction_count: int
    per_class: tuple[ClassAP, ...]


@dataclass(frozen=True)
class PredictionSafetyIssue:
    image_id: str
    reason: str
    detail: str


@dataclass(frozen=True)
class PredictionSafetyReport:
    valid: bool
    image_count: int
    missing_image_count: int
    empty_image_count: int
    prediction_count: int
    max_predictions_per_image: int
    class_counts: dict[int, int]
    confidence_min: float | None
    confidence_max: float | None
    issues: tuple[PredictionSafetyIssue, ...]


def coco_iou_thresholds() -> tuple[float, ...]:
    return tuple(round(0.5 + index * 0.05, 2) for index in range(10))


def map_at_iou(result: MapResult, iou_threshold: float) -> float:
    """Return mean AP for one IoU threshold over classes with ground truth."""
    threshold = round(float(iou_threshold), 2)
    values = [
        item.ap
        for item in result.per_class
        if round(item.iou_threshold, 2) == threshold and item.ground_truth_count > 0
    ]
    return sum(values) / len(values) if values else 0.0


def per_class_ap_summary(result: MapResult, *, num_classes: int = NUM_CLASSES) -> list[dict[str, float | int | str]]:
    """Summarize mAP@50:95/AP50/AP75/AP90 for each class."""
    by_class: dict[int, list[ClassAP]] = {class_id: [] for class_id in range(num_classes)}
    for item in result.per_class:
        if 0 <= item.class_id < num_classes:
            by_class[item.class_id].append(item)

    summary: list[dict[str, float | int | str]] = []
    for class_id in range(num_classes):
        rows = by_class[class_id]
        gt_count = max((row.ground_truth_count for row in rows), default=0)
        pred_count = max((row.prediction_count for row in rows), default=0)
        valid_rows = [row for row in rows if row.ground_truth_count > 0]
        ap_values = [row.ap for row in valid_rows]
        by_iou = {round(row.iou_threshold, 2): row.ap for row in valid_rows}
        summary.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else str(class_id),
                "ground_truth_count": gt_count,
                "prediction_count": pred_count,
                "ap_50_95": sum(ap_values) / len(ap_values) if ap_values else 0.0,
                "ap_50": by_iou.get(0.5, 0.0),
                "ap_75": by_iou.get(0.75, 0.0),
                "ap_90": by_iou.get(0.9, 0.0),
            }
        )
    return summary


def map_summary(result: MapResult, *, num_classes: int = NUM_CLASSES) -> dict[str, object]:
    """Return common aggregate and per-class AP diagnostics."""
    return {
        "map_50_95": result.map,
        "map_50": result.map50,
        "map_75": map_at_iou(result, 0.75),
        "map_90": map_at_iou(result, 0.9),
        "ground_truth_count": result.ground_truth_count,
        "prediction_count": result.prediction_count,
        "per_class_ap": per_class_ap_summary(result, num_classes=num_classes),
    }


def validate_prediction_safety(
    predictions: Mapping[str, Sequence[DetectionLabel]],
    *,
    image_ids: Sequence[str],
    max_predictions_per_image: int,
    num_classes: int = NUM_CLASSES,
) -> PredictionSafetyReport:
    """Check prediction distributions for submission-safety diagnostics."""
    issues: list[PredictionSafetyIssue] = []
    class_counts = {class_id: 0 for class_id in range(num_classes)}
    confidences: list[float] = []
    missing = 0
    empty = 0
    total = 0
    max_count = 0

    for image_id in image_ids:
        records = list(predictions.get(image_id, []))
        if image_id not in predictions:
            missing += 1
        if not records:
            empty += 1
        if len(records) > max_predictions_per_image:
            issues.append(
                PredictionSafetyIssue(
                    image_id=image_id,
                    reason="too_many_predictions",
                    detail=f"{len(records)} > {max_predictions_per_image}",
                )
            )
        max_count = max(max_count, len(records))
        total += len(records)
        for record in records:
            if not 0 <= record.class_id < num_classes:
                issues.append(
                    PredictionSafetyIssue(
                        image_id=image_id,
                        reason="invalid_class_id",
                        detail=str(record.class_id),
                    )
                )
                continue
            class_counts[record.class_id] += 1
            if record.confidence is None:
                issues.append(
                    PredictionSafetyIssue(
                        image_id=image_id,
                        reason="missing_confidence",
                        detail=str(record.class_id),
                    )
                )
            else:
                confidences.append(record.confidence)
                if not 0.0 <= record.confidence <= 1.0:
                    issues.append(
                        PredictionSafetyIssue(
                            image_id=image_id,
                            reason="confidence_out_of_range",
                            detail=str(record.confidence),
                        )
                    )

    return PredictionSafetyReport(
        valid=not issues,
        image_count=len(image_ids),
        missing_image_count=missing,
        empty_image_count=empty,
        prediction_count=total,
        max_predictions_per_image=max_count,
        class_counts=class_counts,
        confidence_min=min(confidences) if confidences else None,
        confidence_max=max(confidences) if confidences else None,
        issues=tuple(issues),
    )


def xywh_to_xyxy(label: DetectionLabel) -> BoxXYXY:
    half_w = label.norm_w / 2.0
    half_h = label.norm_h / 2.0
    return (
        label.norm_center_x - half_w,
        label.norm_center_y - half_h,
        label.norm_center_x + half_w,
        label.norm_center_y + half_h,
    )


def box_iou_xyxy(left: BoxXYXY, right: BoxXYXY) -> float:
    inter_x1 = max(left[0], right[0])
    inter_y1 = max(left[1], right[1])
    inter_x2 = min(left[2], right[2])
    inter_y2 = min(left[3], right[3])
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def evaluate_detection_map(
    ground_truths: Mapping[str, Sequence[DetectionLabel]],
    predictions: Mapping[str, Sequence[DetectionLabel]],
    *,
    iou_thresholds: Iterable[float] | None = None,
    num_classes: int = NUM_CLASSES,
) -> MapResult:
    """Evaluate COCO-style mAP@50:95 over normalized YOLO boxes."""
    thresholds = tuple(iou_thresholds if iou_thresholds is not None else coco_iou_thresholds())
    if not thresholds:
        raise ValueError("at least one IoU threshold is required")

    _require_prediction_confidences(predictions)
    per_class: list[ClassAP] = []
    for threshold in thresholds:
        for class_id in range(num_classes):
            ap = _average_precision_for_class(ground_truths, predictions, class_id, threshold)
            per_class.append(ap)

    scored = [item.ap for item in per_class if item.ground_truth_count > 0]
    map_value = sum(scored) / len(scored) if scored else 0.0
    map50_values = [
        item.ap
        for item in per_class
        if item.iou_threshold == 0.5 and item.ground_truth_count > 0
    ]
    map50 = sum(map50_values) / len(map50_values) if map50_values else 0.0
    return MapResult(
        map=map_value,
        map50=map50,
        ground_truth_count=sum(len(records) for records in ground_truths.values()),
        prediction_count=sum(len(records) for records in predictions.values()),
        per_class=tuple(per_class),
    )


def _average_precision_for_class(
    ground_truths: Mapping[str, Sequence[DetectionLabel]],
    predictions: Mapping[str, Sequence[DetectionLabel]],
    class_id: int,
    iou_threshold: float,
) -> ClassAP:
    gt_by_image = {
        image_id: [record for record in records if record.class_id == class_id]
        for image_id, records in ground_truths.items()
    }
    gt_count = sum(len(records) for records in gt_by_image.values())
    class_predictions = [
        (image_id, record)
        for image_id, records in predictions.items()
        for record in records
        if record.class_id == class_id
    ]
    class_predictions.sort(key=lambda item: item[1].confidence or 0.0, reverse=True)

    if gt_count == 0:
        return ClassAP(class_id, iou_threshold, 0.0, 0, len(class_predictions))

    matched: dict[str, set[int]] = {image_id: set() for image_id in gt_by_image}
    true_positives: list[float] = []
    false_positives: list[float] = []

    for image_id, prediction in class_predictions:
        image_ground_truths = gt_by_image.get(image_id, [])
        pred_box = xywh_to_xyxy(prediction)
        best_iou = 0.0
        best_index = -1
        for index, target in enumerate(image_ground_truths):
            if index in matched.setdefault(image_id, set()):
                continue
            iou = box_iou_xyxy(pred_box, xywh_to_xyxy(target))
            if iou > best_iou:
                best_iou = iou
                best_index = index

        if best_iou >= iou_threshold and best_index >= 0:
            matched[image_id].add(best_index)
            true_positives.append(1.0)
            false_positives.append(0.0)
        else:
            true_positives.append(0.0)
            false_positives.append(1.0)

    ap = _interpolated_average_precision(true_positives, false_positives, gt_count)
    return ClassAP(class_id, iou_threshold, ap, gt_count, len(class_predictions))


def _interpolated_average_precision(
    true_positives: Sequence[float],
    false_positives: Sequence[float],
    ground_truth_count: int,
) -> float:
    if not true_positives:
        return 0.0

    precisions: list[float] = []
    recalls: list[float] = []
    cum_tp = 0.0
    cum_fp = 0.0
    for tp, fp in zip(true_positives, false_positives):
        cum_tp += tp
        cum_fp += fp
        precisions.append(cum_tp / max(cum_tp + cum_fp, 1e-12))
        recalls.append(cum_tp / ground_truth_count)

    ap = 0.0
    for recall_threshold in (index / 100 for index in range(101)):
        precision_at_recall = [
            precision
            for precision, recall in zip(precisions, recalls)
            if recall >= recall_threshold
        ]
        ap += max(precision_at_recall) if precision_at_recall else 0.0
    return ap / 101.0


def _require_prediction_confidences(
    predictions: Mapping[str, Sequence[DetectionLabel]],
) -> None:
    for image_id, records in predictions.items():
        for index, record in enumerate(records, start=1):
            if record.confidence is None:
                raise ValueError(f"{image_id}:{index}: prediction is missing confidence")
