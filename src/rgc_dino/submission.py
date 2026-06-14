"""Submission TXT writing, validation, and packaging helpers."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Mapping, Sequence

from .constants import MAX_PREDICTIONS_PER_IMAGE
from .labels import DetectionLabel, load_label_file, parse_label_line


def format_submission_line(label: DetectionLabel) -> str:
    if label.confidence is None:
        raise ValueError("submission predictions require confidence")
    parse_label_line(
        (
            f"{label.class_id} {label.norm_center_x:g} {label.norm_center_y:g} "
            f"{label.norm_w:g} {label.norm_h:g} {label.confidence:g}"
        ),
        require_confidence=True,
    )
    return (
        f"{label.class_id} {label.norm_center_x:g} {label.norm_center_y:g} "
        f"{label.norm_w:g} {label.norm_h:g} {label.confidence:g}"
    )


def write_submission_files(
    image_ids: Sequence[str],
    predictions: Mapping[str, Sequence[DetectionLabel]],
    output_dir: str | Path,
    *,
    max_predictions_per_image: int = MAX_PREDICTIONS_PER_IMAGE,
) -> None:
    """Write one TXT file per image id, sorted by descending confidence."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    expected_ids = sorted(set(image_ids))
    for image_id in expected_ids:
        records = sorted(
            predictions.get(image_id, []),
            key=lambda record: record.confidence if record.confidence is not None else -1.0,
            reverse=True,
        )[:max_predictions_per_image]
        lines = [format_submission_line(record) for record in records]
        content = "\n".join(lines)
        if content:
            content += "\n"
        (out_path / f"{image_id}.txt").write_text(content, encoding="utf-8")


def write_empty_submission(image_ids: Sequence[str], output_dir: str | Path) -> None:
    """Write a legal no-detection submission skeleton."""
    write_submission_files(image_ids, {}, output_dir)


def run_no_detection_inference(image_ids: Sequence[str], output_dir: str | Path) -> None:
    """Create deterministic v0 no-detection predictions for every image id."""
    clear_prediction_txt_files(output_dir)
    write_empty_submission(image_ids, output_dir)


def clear_prediction_txt_files(output_dir: str | Path) -> None:
    """Remove stale TXT prediction files from an output directory."""
    root = Path(output_dir)
    if not root.exists():
        return
    for path in root.glob("*.txt"):
        path.unlink()


def validate_submission_dir(
    image_ids: Sequence[str],
    submission_dir: str | Path,
    *,
    max_predictions_per_image: int = MAX_PREDICTIONS_PER_IMAGE,
) -> list[str]:
    """Return validation errors for a submission directory."""
    errors: list[str] = []
    root = Path(submission_dir)
    expected = {f"{image_id}.txt" for image_id in image_ids}
    actual = {path.name for path in root.glob("*.txt")}

    for missing in sorted(expected - actual):
        errors.append(f"missing required submission file: {missing}")
    for extra in sorted(actual - expected):
        errors.append(f"unexpected submission file: {extra}")

    for filename in sorted(expected & actual):
        path = root / filename
        try:
            records = load_label_file(path, require_confidence=True)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if len(records) > max_predictions_per_image:
            errors.append(
                f"{path}: {len(records)} predictions exceeds limit {max_predictions_per_image}"
            )

    return errors


def zip_submission_dir(submission_dir: str | Path, zip_path: str | Path) -> Path:
    """Create a zip with submission TXT files at archive root."""
    root = Path(submission_dir)
    output = Path(zip_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.glob("*.txt")):
            archive.write(path, arcname=path.name)
    return output
