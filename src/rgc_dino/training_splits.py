"""Training split selection helpers."""

from __future__ import annotations

from pathlib import Path

from .coco_export import load_split_ids
from .dataset import discover_aligned_samples


def select_train_val_ids(
    *,
    dataset_root: str | Path,
    labels_dir: str | Path,
    assignments_path: str | Path,
    fold: int,
    train_all: bool = False,
) -> tuple[list[str], list[str]]:
    """Return train/validation IDs for fold training or all-data final training."""
    if not train_all:
        return load_split_ids(assignments_path, fold=fold)

    samples = discover_aligned_samples(
        dataset_root,
        labels_dir=labels_dir,
        require_labels=True,
    )
    train_ids = [sample.sample_id for sample in samples]
    if not train_ids:
        raise ValueError("no labeled aligned samples found for all-train mode")
    return train_ids, []
