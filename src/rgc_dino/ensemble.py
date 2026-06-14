"""Prediction-directory ensembling helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .constants import MAX_PREDICTIONS_PER_IMAGE
from .labels import DetectionLabel, load_label_file
from .metrics import box_iou_xyxy, xywh_to_xyxy
from .submission import clear_prediction_txt_files, write_submission_files


def ensemble_prediction_dirs(
    prediction_dirs: Sequence[str | Path],
    output_dir: str | Path,
    *,
    image_ids: Sequence[str] | None = None,
    nms_iou_threshold: float = 0.8,
    max_predictions_per_image: int = MAX_PREDICTIONS_PER_IMAGE,
    min_model_votes: int = 1,
) -> dict[str, int | float | str]:
    """Merge prediction TXT directories with classwise NMS."""
    roots = [Path(path) for path in prediction_dirs]
    if len(roots) < 2:
        raise ValueError("at least two prediction directories are required")
    if not 0.0 < nms_iou_threshold <= 1.0:
        raise ValueError("nms_iou_threshold must be in (0, 1]")
    if not 1 <= min_model_votes <= len(roots):
        raise ValueError("min_model_votes must be between 1 and the number of prediction directories")

    expected_ids = sorted(image_ids) if image_ids is not None else _image_ids_from_dir(roots[0])
    predictions: dict[str, list[DetectionLabel]] = {}
    for image_id in expected_ids:
        records: list[tuple[DetectionLabel, int]] = []
        for model_index, root in enumerate(roots):
            path = root / f"{image_id}.txt"
            if path.exists():
                records.extend((record, model_index) for record in load_label_file(path, require_confidence=True))
        predictions[image_id] = _classwise_nms_with_votes(
            records,
            iou_threshold=nms_iou_threshold,
            max_predictions=max_predictions_per_image,
            min_model_votes=min_model_votes,
        )

    clear_prediction_txt_files(output_dir)
    write_submission_files(
        expected_ids,
        predictions,
        output_dir,
        max_predictions_per_image=max_predictions_per_image,
    )
    return {
        "files": len(expected_ids),
        "prediction_objects": sum(len(records) for records in predictions.values()),
        "nms_iou_threshold": nms_iou_threshold,
        "min_model_votes": min_model_votes,
        "output_dir": str(output_dir),
    }


def classwise_nms(
    records: Sequence[DetectionLabel],
    *,
    iou_threshold: float,
    max_predictions: int = MAX_PREDICTIONS_PER_IMAGE,
) -> list[DetectionLabel]:
    """Keep highest-confidence boxes after classwise NMS."""
    return _classwise_nms_with_votes(
        [(record, 0) for record in records],
        iou_threshold=iou_threshold,
        max_predictions=max_predictions,
        min_model_votes=1,
    )


def _classwise_nms_with_votes(
    records: Sequence[tuple[DetectionLabel, int]],
    *,
    iou_threshold: float,
    max_predictions: int,
    min_model_votes: int,
) -> list[DetectionLabel]:
    kept: list[DetectionLabel] = []
    by_class: dict[int, list[tuple[DetectionLabel, int]]] = {}
    for record, model_index in records:
        by_class.setdefault(record.class_id, []).append((record, model_index))

    for class_records in by_class.values():
        remaining = sorted(class_records, key=lambda item: item[0].confidence or 0.0, reverse=True)
        while remaining:
            leader, _leader_model = remaining[0]
            leader_box = xywh_to_xyxy(leader)
            cluster: list[tuple[DetectionLabel, int]] = []
            next_remaining: list[tuple[DetectionLabel, int]] = []
            for candidate, model_index in remaining:
                if box_iou_xyxy(leader_box, xywh_to_xyxy(candidate)) >= iou_threshold:
                    cluster.append((candidate, model_index))
                else:
                    next_remaining.append((candidate, model_index))
            if len({model_index for _candidate, model_index in cluster}) >= min_model_votes:
                kept.append(leader)
            remaining = next_remaining

    kept.sort(key=lambda record: record.confidence if record.confidence is not None else -1.0, reverse=True)
    return kept[:max_predictions]


def _image_ids_from_dir(prediction_dir: Path) -> list[str]:
    image_ids = sorted(path.stem for path in prediction_dir.glob("*.txt"))
    if not image_ids:
        raise ValueError(f"no TXT prediction files found in {prediction_dir}")
    return image_ids
