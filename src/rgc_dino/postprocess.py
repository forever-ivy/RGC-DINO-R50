"""Post-processing helpers for competition detection predictions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any, NamedTuple

from .constants import CLASS_NAMES, NUM_CLASSES
from .labels import DetectionLabel
from .metrics import box_iou_xyxy, xywh_to_xyxy


ClassThresholds = tuple[float, ...]


class ClassAllocationConfig(NamedTuple):
    """Configuration for class-aware final top-k allocation."""

    score_weights: tuple[float, ...] | None = None
    score_biases: tuple[float, ...] | None = None
    soft_caps: tuple[int | None, ...] | None = None
    soft_cap_decay: float = 1.0
    reserved_quotas: tuple[int, ...] | None = None


class _ScoredDetection(NamedTuple):
    record: DetectionLabel
    score: float
    original_index: int


def load_class_score_thresholds(path: str | Path, *, num_classes: int = NUM_CLASSES) -> ClassThresholds:
    """Load per-class score thresholds from JSON.

    Supported JSON shapes:
    - ``[0.001, ...]`` with one value per class
    - ``{"class_conf": [...]}`` or ``{"class_score_thresholds": [...]}``
    - ``{"class_conf": {"0": 0.003, "person": 0.003}}``
    - ``{"0": 0.003, "person": 0.003}``

    Missing classes in mapping form default to 0.0, which keeps all candidates
    for those classes after the global candidate threshold has been applied.
    """
    threshold_path = Path(path)
    payload = json.loads(threshold_path.read_text(encoding="utf-8"))
    return coerce_class_score_thresholds(payload, num_classes=num_classes)


def coerce_class_score_thresholds(payload: Any, *, num_classes: int = NUM_CLASSES) -> ClassThresholds:
    """Normalize a JSON-like threshold payload into a fixed-length tuple."""
    if isinstance(payload, Mapping):
        for key in ("class_conf", "class_score_thresholds", "thresholds"):
            if key in payload:
                return coerce_class_score_thresholds(payload[key], num_classes=num_classes)
        thresholds = [0.0] * num_classes
        for raw_key, raw_value in payload.items():
            class_id = _class_id_from_key(raw_key, num_classes=num_classes)
            thresholds[class_id] = _validated_threshold(raw_value, source=f"class {raw_key}")
        return tuple(thresholds)

    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        if len(payload) != num_classes:
            raise ValueError(f"expected {num_classes} class thresholds, got {len(payload)}")
        return tuple(_validated_threshold(value, source=f"class {index}") for index, value in enumerate(payload))

    raise ValueError("class thresholds must be a list or mapping")


def apply_class_score_thresholds(
    predictions: Mapping[str, Sequence[DetectionLabel]],
    thresholds: Sequence[float] | None,
) -> dict[str, list[DetectionLabel]]:
    """Filter predictions by per-class confidence thresholds."""
    if thresholds is None:
        return {sample_id: list(records) for sample_id, records in predictions.items()}
    if len(thresholds) != NUM_CLASSES:
        raise ValueError(f"expected {NUM_CLASSES} class thresholds, got {len(thresholds)}")

    filtered: dict[str, list[DetectionLabel]] = {}
    for sample_id, records in predictions.items():
        kept: list[DetectionLabel] = []
        for record in records:
            confidence = record.confidence if record.confidence is not None else 0.0
            if confidence >= float(thresholds[record.class_id]):
                kept.append(record)
        if kept:
            filtered[sample_id] = kept
    return filtered


def apply_classwise_nms(
    predictions: Mapping[str, Sequence[DetectionLabel]],
    *,
    iou_threshold: float | None,
) -> dict[str, list[DetectionLabel]]:
    """Apply greedy class-wise NMS independently per image."""
    if iou_threshold is None:
        return {sample_id: list(records) for sample_id, records in predictions.items()}
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1]")
    return {
        sample_id: _classwise_nms_one_image(list(records), iou_threshold=iou_threshold)
        for sample_id, records in predictions.items()
    }


def cap_predictions_per_image(
    predictions: Mapping[str, Sequence[DetectionLabel]],
    *,
    max_detections: int,
) -> dict[str, list[DetectionLabel]]:
    """Sort by confidence and retain at most ``max_detections`` per image."""
    if max_detections <= 0:
        raise ValueError("max_detections must be positive")
    capped: dict[str, list[DetectionLabel]] = {}
    for sample_id, records in predictions.items():
        sorted_records = sorted(records, key=lambda item: item.confidence or 0.0, reverse=True)
        capped[sample_id] = sorted_records[:max_detections]
    return capped


def load_class_allocation_config(path: str | Path, *, num_classes: int = NUM_CLASSES) -> ClassAllocationConfig:
    """Load class-aware top-k allocation config from JSON.

    Supported keys:
    - ``score_weights`` / ``class_score_weights``: list or class-name/id mapping.
    - ``score_biases`` / ``class_score_biases``: list or class-name/id mapping.
    - ``soft_caps`` / ``class_soft_caps``: list or class-name/id mapping; missing classes disable caps.
    - ``soft_cap_decay``: multiplicative factor for same-class records after the cap.
    - ``reserved_quotas`` / ``class_reserved_quotas``: list or class-name/id mapping.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return coerce_class_allocation_config(payload, num_classes=num_classes)


def coerce_class_allocation_config(payload: Any, *, num_classes: int = NUM_CLASSES) -> ClassAllocationConfig:
    """Normalize a JSON-like payload into :class:`ClassAllocationConfig`."""
    if payload is None:
        return ClassAllocationConfig()
    if not isinstance(payload, Mapping):
        raise ValueError("class allocation config must be a mapping")
    score_weights = _coerce_optional_class_floats(
        _first_present(payload, "score_weights", "class_score_weights", "weights"),
        num_classes=num_classes,
        default=1.0,
        min_value=0.0,
        name="score weight",
    )
    score_biases = _coerce_optional_class_floats(
        _first_present(payload, "score_biases", "class_score_biases", "biases"),
        num_classes=num_classes,
        default=0.0,
        min_value=None,
        name="score bias",
    )
    soft_caps = _coerce_optional_class_ints(
        _first_present(payload, "soft_caps", "class_soft_caps"),
        num_classes=num_classes,
        default=None,
        name="soft cap",
    )
    reserved_quotas = _coerce_optional_class_ints(
        _first_present(payload, "reserved_quotas", "class_reserved_quotas", "quotas"),
        num_classes=num_classes,
        default=0,
        name="reserved quota",
    )
    soft_cap_decay = float(payload.get("soft_cap_decay", 1.0))
    if not 0.0 <= soft_cap_decay <= 1.0:
        raise ValueError("soft_cap_decay must be in [0, 1]")
    return ClassAllocationConfig(
        score_weights=score_weights,
        score_biases=score_biases,
        soft_caps=soft_caps,
        soft_cap_decay=soft_cap_decay,
        reserved_quotas=reserved_quotas,
    )


def apply_class_score_weights(
    predictions: Mapping[str, Sequence[DetectionLabel]],
    *,
    weights: Sequence[float] | None = None,
    biases: Sequence[float] | None = None,
) -> dict[str, list[DetectionLabel]]:
    """Return predictions with confidence replaced by class-calibrated scores."""
    _validate_optional_class_values(weights, name="weights")
    _validate_optional_class_values(biases, name="biases")
    calibrated: dict[str, list[DetectionLabel]] = {}
    for sample_id, records in predictions.items():
        calibrated[sample_id] = [
            _replace_confidence(record, _class_aware_score(record, weights=weights, biases=biases))
            for record in records
        ]
    return calibrated


def cap_predictions_per_image_class_aware(
    predictions: Mapping[str, Sequence[DetectionLabel]],
    *,
    max_detections: int,
    allocation: ClassAllocationConfig | Mapping[str, Any] | None = None,
) -> dict[str, list[DetectionLabel]]:
    """Apply class-aware score calibration/quotas and retain at most ``max_detections``.

    With an empty/default allocation this is equivalent to global confidence top-k.
    Reserved quotas are *opportunistic*: they reserve up to K existing high-scoring
    records for a class, but never synthesize low-quality records or exceed top-k.
    """
    if max_detections <= 0:
        raise ValueError("max_detections must be positive")
    config = coerce_class_allocation_config(allocation) if isinstance(allocation, Mapping) else (allocation or ClassAllocationConfig())
    _validate_optional_class_values(config.score_weights, name="score_weights")
    _validate_optional_class_values(config.score_biases, name="score_biases")
    capped: dict[str, list[DetectionLabel]] = {}
    for sample_id, records in predictions.items():
        capped[sample_id] = _cap_one_image_class_aware(list(records), max_detections=max_detections, config=config)
    return capped


def summarize_predictions(
    predictions: Mapping[str, Sequence[DetectionLabel]],
    *,
    image_ids: Sequence[str] | None = None,
    num_classes: int = NUM_CLASSES,
) -> dict[str, Any]:
    """Return lightweight count diagnostics for prediction distributions."""
    per_class = {str(class_id): 0 for class_id in range(num_classes)}
    per_image_counts: dict[str, int] = {}
    total = 0
    ids = list(image_ids) if image_ids is not None else sorted(predictions)
    for sample_id in ids:
        records = list(predictions.get(sample_id, []))
        per_image_counts[sample_id] = len(records)
        total += len(records)
        for record in records:
            if 0 <= record.class_id < num_classes:
                per_class[str(record.class_id)] += 1
    image_count = len(ids)
    counts = list(per_image_counts.values())
    return {
        "images": image_count,
        "prediction_objects": total,
        "non_empty_images": sum(1 for count in counts if count > 0),
        "mean_predictions_per_image": total / max(1, image_count),
        "max_predictions_per_image": max(counts) if counts else 0,
        "per_image_count_quantiles": count_quantiles(counts),
        "per_class_counts": per_class,
    }


def count_quantiles(values: Sequence[int | float]) -> dict[str, float]:
    """Return stable quantiles for count-like diagnostics."""
    if not values:
        return {"min": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    sorted_values = sorted(float(value) for value in values)
    return {
        "min": sorted_values[0],
        "p50": _quantile(sorted_values, 0.5),
        "p90": _quantile(sorted_values, 0.9),
        "p95": _quantile(sorted_values, 0.95),
        "max": sorted_values[-1],
    }


def score_histograms(
    predictions: Mapping[str, Sequence[DetectionLabel]],
    *,
    bins: Sequence[float] | None = None,
    num_classes: int = NUM_CLASSES,
) -> dict[str, Any]:
    """Count prediction confidences by class and score bucket."""
    edges = list(bins) if bins is not None else [0.0, 0.0005, 0.001, 0.0015, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    if len(edges) < 2 or any(edges[index] > edges[index + 1] for index in range(len(edges) - 1)):
        raise ValueError("score histogram bins must be sorted ascending")
    bucket_labels = [f"[{edges[index]:g},{edges[index + 1]:g}{']' if index == len(edges) - 2 else ')'}" for index in range(len(edges) - 1)]
    by_class = {
        str(class_id): {bucket: 0 for bucket in bucket_labels}
        for class_id in range(num_classes)
    }
    all_scores = {bucket: 0 for bucket in bucket_labels}
    for records in predictions.values():
        for record in records:
            if record.confidence is None:
                continue
            bucket_index = _bucket_index(float(record.confidence), edges)
            bucket = bucket_labels[bucket_index]
            all_scores[bucket] += 1
            if 0 <= record.class_id < num_classes:
                by_class[str(record.class_id)][bucket] += 1
    return {"bins": edges, "all": all_scores, "by_class": by_class}


def topk_truncation_report(
    before: Mapping[str, Sequence[DetectionLabel]],
    after: Mapping[str, Sequence[DetectionLabel]],
    *,
    image_ids: Sequence[str] | None = None,
    max_detections: int,
    num_classes: int = NUM_CLASSES,
) -> dict[str, Any]:
    """Summarize predictions dropped by final per-image top-k capping."""
    ids = list(image_ids) if image_ids is not None else sorted(set(before) | set(after))
    dropped_by_class = {str(class_id): 0 for class_id in range(num_classes)}
    dropped_scores: list[DetectionLabel] = []
    saturated_images: list[str] = []
    total_before = 0
    total_after = 0
    for sample_id in ids:
        before_records = sorted(before.get(sample_id, []), key=lambda item: item.confidence or 0.0, reverse=True)
        after_records = list(after.get(sample_id, []))
        total_before += len(before_records)
        total_after += len(after_records)
        if len(before_records) > max_detections:
            saturated_images.append(sample_id)
        dropped = before_records[max_detections:]
        for record in dropped:
            if 0 <= record.class_id < num_classes:
                dropped_by_class[str(record.class_id)] += 1
            dropped_scores.append(record)
    return {
        "max_detections": max_detections,
        "images": len(ids),
        "saturated_image_count": len(saturated_images),
        "saturated_images_sample": saturated_images[:20],
        "total_before_topk": total_before,
        "total_after_topk": total_after,
        "dropped_prediction_objects": max(0, total_before - total_after),
        "dropped_by_class": dropped_by_class,
        "dropped_score_histogram": score_histograms({"dropped": dropped_scores}, num_classes=num_classes),
    }


def _cap_one_image_class_aware(
    records: list[DetectionLabel],
    *,
    max_detections: int,
    config: ClassAllocationConfig,
) -> list[DetectionLabel]:
    scored = [
        _ScoredDetection(
            record=record,
            score=_class_aware_score(record, weights=config.score_weights, biases=config.score_biases),
            original_index=index,
        )
        for index, record in enumerate(records)
    ]
    if config.soft_caps is not None and config.soft_cap_decay < 1.0:
        scored = _apply_soft_caps(scored, soft_caps=config.soft_caps, decay=config.soft_cap_decay)

    selected: list[_ScoredDetection] = []
    selected_indices: set[int] = set()
    quotas = config.reserved_quotas
    if quotas is not None:
        by_class: dict[int, list[_ScoredDetection]] = {}
        for item in scored:
            by_class.setdefault(item.record.class_id, []).append(item)
        for class_id, quota in enumerate(quotas):
            if quota <= 0 or len(selected) >= max_detections:
                continue
            candidates = sorted(by_class.get(class_id, []), key=_scored_sort_key)
            for item in candidates[: max(0, min(quota, max_detections - len(selected)) )]:
                selected.append(item)
                selected_indices.add(item.original_index)

    remaining = [item for item in scored if item.original_index not in selected_indices]
    selected.extend(sorted(remaining, key=_scored_sort_key)[: max(0, max_detections - len(selected))])
    selected = sorted(selected[:max_detections], key=_scored_sort_key)
    return [_replace_confidence(item.record, item.score) for item in selected]


def _apply_soft_caps(
    scored: Sequence[_ScoredDetection],
    *,
    soft_caps: Sequence[int | None],
    decay: float,
) -> list[_ScoredDetection]:
    by_class: dict[int, list[_ScoredDetection]] = {}
    for item in scored:
        by_class.setdefault(item.record.class_id, []).append(item)
    decayed_by_index: dict[int, _ScoredDetection] = {}
    for class_id, items in by_class.items():
        cap = soft_caps[class_id]
        if cap is None:
            for item in items:
                decayed_by_index[item.original_index] = item
            continue
        if cap < 0:
            raise ValueError("soft caps must be non-negative or null")
        for rank, item in enumerate(sorted(items, key=_scored_sort_key), start=1):
            score = item.score if rank <= cap else item.score * decay
            decayed_by_index[item.original_index] = _ScoredDetection(item.record, score, item.original_index)
    return [decayed_by_index[item.original_index] for item in scored]


def _class_aware_score(
    record: DetectionLabel,
    *,
    weights: Sequence[float] | None,
    biases: Sequence[float] | None,
) -> float:
    score = float(record.confidence if record.confidence is not None else 0.0)
    if weights is not None:
        score *= float(weights[record.class_id])
    if biases is not None:
        score += float(biases[record.class_id])
    return min(1.0, max(0.0, score))


def _replace_confidence(record: DetectionLabel, confidence: float) -> DetectionLabel:
    return DetectionLabel(
        class_id=record.class_id,
        norm_center_x=record.norm_center_x,
        norm_center_y=record.norm_center_y,
        norm_w=record.norm_w,
        norm_h=record.norm_h,
        confidence=confidence,
    )


def _scored_sort_key(item: _ScoredDetection) -> tuple[float, float, int]:
    original_confidence = float(item.record.confidence if item.record.confidence is not None else 0.0)
    return (-item.score, -original_confidence, item.original_index)


def _first_present(payload: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _coerce_optional_class_floats(
    payload: Any,
    *,
    num_classes: int,
    default: float,
    min_value: float | None,
    name: str,
) -> tuple[float, ...] | None:
    if payload is None:
        return None
    if isinstance(payload, Mapping):
        values = [float(default)] * num_classes
        for raw_key, raw_value in payload.items():
            class_id = _class_id_from_key(raw_key, num_classes=num_classes)
            value = float(raw_value)
            if min_value is not None and value < min_value:
                raise ValueError(f"{name} for class {raw_key} must be >= {min_value}")
            values[class_id] = value
        return tuple(values)
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        if len(payload) != num_classes:
            raise ValueError(f"expected {num_classes} {name}s, got {len(payload)}")
        values = tuple(float(value) for value in payload)
        if min_value is not None and any(value < min_value for value in values):
            raise ValueError(f"{name}s must be >= {min_value}")
        return values
    raise ValueError(f"{name}s must be a list or mapping")


def _coerce_optional_class_ints(
    payload: Any,
    *,
    num_classes: int,
    default: int | None,
    name: str,
) -> tuple[int | None, ...] | None:
    if payload is None:
        return None
    values: list[int | None] = [default] * num_classes
    if isinstance(payload, Mapping):
        for raw_key, raw_value in payload.items():
            class_id = _class_id_from_key(raw_key, num_classes=num_classes)
            values[class_id] = _validated_optional_int(raw_value, name=name, source=str(raw_key))
        return tuple(values)
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        if len(payload) != num_classes:
            raise ValueError(f"expected {num_classes} {name}s, got {len(payload)}")
        return tuple(_validated_optional_int(value, name=name, source=str(index)) for index, value in enumerate(payload))
    raise ValueError(f"{name}s must be a list or mapping")


def _validated_optional_int(value: Any, *, name: str, source: str) -> int | None:
    if value is None:
        return None
    integer = int(value)
    if integer < 0:
        raise ValueError(f"{name} for class {source} must be non-negative or null")
    return integer


def _validate_optional_class_values(values: Sequence[float] | None, *, name: str) -> None:
    if values is not None and len(values) != NUM_CLASSES:
        raise ValueError(f"expected {NUM_CLASSES} {name}, got {len(values)}")


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[int(position)]
    lower_value = sorted_values[lower]
    upper_value = sorted_values[upper]
    return lower_value + (upper_value - lower_value) * (position - lower)


def _bucket_index(value: float, edges: Sequence[float]) -> int:
    for index in range(len(edges) - 1):
        lower = edges[index]
        upper = edges[index + 1]
        if index == len(edges) - 2:
            if lower <= value <= upper:
                return index
        elif lower <= value < upper:
            return index
    if value < edges[0]:
        return 0
    return len(edges) - 2


def _classwise_nms_one_image(records: list[DetectionLabel], *, iou_threshold: float) -> list[DetectionLabel]:
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
    kept.sort(key=lambda item: item.confidence or 0.0, reverse=True)
    return kept


def _class_id_from_key(raw_key: Any, *, num_classes: int) -> int:
    if isinstance(raw_key, int):
        class_id = raw_key
    else:
        text = str(raw_key).strip()
        if text in CLASS_NAMES:
            class_id = CLASS_NAMES.index(text)
        else:
            try:
                class_id = int(text)
            except ValueError as exc:
                raise ValueError(f"unknown class threshold key: {raw_key!r}") from exc
    if not 0 <= class_id < num_classes:
        raise ValueError(f"class threshold key {raw_key!r} outside [0, {num_classes - 1}]")
    return class_id


def _validated_threshold(value: Any, *, source: str) -> float:
    threshold = float(value)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"{source}: threshold {threshold} outside [0, 1]")
    return threshold
