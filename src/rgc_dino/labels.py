"""Utilities for YOLO-style competition labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .constants import NUM_CLASSES


@dataclass(frozen=True)
class DetectionLabel:
    class_id: int
    norm_center_x: float
    norm_center_y: float
    norm_w: float
    norm_h: float
    confidence: float | None = None

    @property
    def has_confidence(self) -> bool:
        return self.confidence is not None


def parse_label_line(
    line: str,
    *,
    require_confidence: bool = False,
    source: str = "<line>",
) -> DetectionLabel:
    """Parse and validate one train-label or submission line."""
    parts = line.split()
    expected = 6 if require_confidence else 5
    if len(parts) != expected:
        raise ValueError(f"{source}: expected {expected} fields, got {len(parts)}")

    try:
        class_id = int(parts[0])
        values = [float(value) for value in parts[1:]]
    except ValueError as exc:
        raise ValueError(f"{source}: non-numeric label field") from exc

    if not 0 <= class_id < NUM_CLASSES:
        raise ValueError(f"{source}: class_id {class_id} outside [0, {NUM_CLASSES - 1}]")

    norm_center_x, norm_center_y, norm_w, norm_h = values[:4]
    _validate_unit_interval(norm_center_x, "norm_center_x", source)
    _validate_unit_interval(norm_center_y, "norm_center_y", source)
    _validate_positive_unit(norm_w, "norm_w", source)
    _validate_positive_unit(norm_h, "norm_h", source)

    confidence = None
    if require_confidence:
        confidence = values[4]
        _validate_unit_interval(confidence, "confidence", source)

    return DetectionLabel(
        class_id=class_id,
        norm_center_x=norm_center_x,
        norm_center_y=norm_center_y,
        norm_w=norm_w,
        norm_h=norm_h,
        confidence=confidence,
    )


def load_label_file(path: str | Path, *, require_confidence: bool = False) -> list[DetectionLabel]:
    """Load a label TXT file, skipping blank lines."""
    label_path = Path(path)
    records: list[DetectionLabel] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        records.append(
            parse_label_line(
                stripped,
                require_confidence=require_confidence,
                source=f"{label_path}:{line_number}",
            )
        )
    return records


def load_label_file_clipped(
    path: str | Path,
    *,
    require_confidence: bool = False,
) -> tuple[list[DetectionLabel], int]:
    """Load a label file and clip minor numeric drift into valid YOLO ranges."""
    label_path = Path(path)
    records: list[DetectionLabel] = []
    clipped_count = 0
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        record, was_clipped = parse_label_line_clipped(
            stripped,
            require_confidence=require_confidence,
            source=f"{label_path}:{line_number}",
        )
        records.append(record)
        clipped_count += int(was_clipped)
    return records, clipped_count


def load_label_dir(
    labels_dir: str | Path,
    *,
    require_confidence: bool = False,
    clip: bool = False,
) -> dict[str, list[DetectionLabel]]:
    """Load all TXT files in a directory keyed by sample id."""
    if clip:
        return {
            path.stem: load_label_file_clipped(path, require_confidence=require_confidence)[0]
            for path in iter_label_files(labels_dir)
        }
    return {
        path.stem: load_label_file(path, require_confidence=require_confidence)
        for path in iter_label_files(labels_dir)
    }


def parse_label_line_clipped(
    line: str,
    *,
    require_confidence: bool = False,
    source: str = "<line>",
) -> tuple[DetectionLabel, bool]:
    """Parse a label line, clipping values after numeric/type checks."""
    parts = line.split()
    expected = 6 if require_confidence else 5
    if len(parts) != expected:
        raise ValueError(f"{source}: expected {expected} fields, got {len(parts)}")

    try:
        class_id = int(parts[0])
        values = [float(value) for value in parts[1:]]
    except ValueError as exc:
        raise ValueError(f"{source}: non-numeric label field") from exc

    if not 0 <= class_id < NUM_CLASSES:
        raise ValueError(f"{source}: class_id {class_id} outside [0, {NUM_CLASSES - 1}]")

    norm_center_x, clipped_x = _clip_unit_interval(values[0])
    norm_center_y, clipped_y = _clip_unit_interval(values[1])
    norm_w, clipped_w = _clip_positive_unit(values[2])
    norm_h, clipped_h = _clip_positive_unit(values[3])
    confidence = None
    clipped_confidence = False
    if require_confidence:
        confidence, clipped_confidence = _clip_unit_interval(values[4])

    return (
        DetectionLabel(
            class_id=class_id,
            norm_center_x=norm_center_x,
            norm_center_y=norm_center_y,
            norm_w=norm_w,
            norm_h=norm_h,
            confidence=confidence,
        ),
        clipped_x or clipped_y or clipped_w or clipped_h or clipped_confidence,
    )


def iter_label_files(labels_dir: str | Path) -> Iterable[Path]:
    """Yield label TXT files in stable order."""
    yield from sorted(Path(labels_dir).glob("*.txt"))


def _validate_unit_interval(value: float, field: str, source: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{source}: {field}={value} outside [0, 1]")


def _validate_positive_unit(value: float, field: str, source: str) -> None:
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{source}: {field}={value} outside (0, 1]")


def _clip_unit_interval(value: float) -> tuple[float, bool]:
    clipped = min(1.0, max(0.0, value))
    return clipped, clipped != value


def _clip_positive_unit(value: float) -> tuple[float, bool]:
    clipped = min(1.0, max(1e-12, value))
    return clipped, clipped != value
