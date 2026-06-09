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


def iter_label_files(labels_dir: str | Path) -> Iterable[Path]:
    """Yield label TXT files in stable order."""
    yield from sorted(Path(labels_dir).glob("*.txt"))


def _validate_unit_interval(value: float, field: str, source: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{source}: {field}={value} outside [0, 1]")


def _validate_positive_unit(value: float, field: str, source: str) -> None:
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{source}: {field}={value} outside (0, 1]")

