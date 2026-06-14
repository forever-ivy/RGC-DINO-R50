"""Helpers for reading and applying sample ID lists."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")


def load_sample_ids_file(path: str | Path) -> list[str]:
    """Load sample IDs from a text file, skipping blanks and full-line comments."""
    seen: set[str] = set()
    sample_ids: list[str] = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        sample_id = raw_line.strip()
        if not sample_id or sample_id.startswith("#"):
            continue
        if sample_id in seen:
            continue
        seen.add(sample_id)
        sample_ids.append(sample_id)
    return sample_ids


def restrict_mapping_to_sample_ids(values: Mapping[str, T], sample_ids: Sequence[str]) -> dict[str, T]:
    """Return mapping entries for requested sample IDs, preserving requested order."""
    return {sample_id: values[sample_id] for sample_id in sample_ids if sample_id in values}
