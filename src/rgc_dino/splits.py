"""Deterministic grouped split utilities for small object-detection datasets."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

from .constants import NUM_CLASSES
from .labels import load_label_dir


@dataclass(frozen=True)
class SplitFold:
    fold_index: int
    train_ids: tuple[str, ...]
    val_ids: tuple[str, ...]
    val_class_counts: dict[int, int]


def build_grouped_stratified_splits(
    labels_by_sample: Mapping[str, Sequence[int]],
    *,
    folds: int = 3,
) -> list[SplitFold]:
    """Build deterministic grouped folds with greedy class-count balancing."""
    if folds < 2:
        raise ValueError("folds must be >= 2")
    if len(labels_by_sample) < folds:
        raise ValueError("number of samples must be >= folds")

    groups = _group_samples(labels_by_sample)
    fold_groups: list[list[str]] = [[] for _ in range(folds)]
    fold_counts: list[Counter[int]] = [Counter() for _ in range(folds)]

    for group_id, group_sample_ids in sorted(
        groups.items(),
        key=lambda item: (-sum(len(labels_by_sample[sample_id]) for sample_id in item[1]), item[0]),
    ):
        group_counts = Counter()
        for sample_id in group_sample_ids:
            group_counts.update(labels_by_sample[sample_id])

        best_fold = min(
            range(folds),
            key=lambda index: (
                _imbalance_score(fold_counts[index] + group_counts),
                len(fold_groups[index]),
                index,
            ),
        )
        fold_groups[best_fold].append(group_id)
        fold_counts[best_fold].update(group_counts)

    sample_ids = set(labels_by_sample)
    output: list[SplitFold] = []
    for fold_index, group_ids in enumerate(fold_groups):
        val_ids = tuple(
            sorted(
                sample_id
                for group_id in group_ids
                for sample_id in groups[group_id]
            )
        )
        train_ids = tuple(sorted(sample_ids - set(val_ids)))
        output.append(
            SplitFold(
                fold_index=fold_index,
                train_ids=train_ids,
                val_ids=val_ids,
                val_class_counts=dict(sorted(fold_counts[fold_index].items())),
            )
        )
    return output


def load_label_class_ids(labels_dir: str | Path, *, clip: bool = False) -> dict[str, list[int]]:
    records_by_sample = load_label_dir(labels_dir, clip=clip)
    return {
        sample_id: [record.class_id for record in records]
        for sample_id, records in records_by_sample.items()
    }


def write_split_manifest(splits: Sequence[SplitFold], output_dir: str | Path) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "folds": len(splits),
        "samples": len({sample_id for fold in splits for sample_id in fold.val_ids}),
        "fold_summaries": [
            {
                "fold_index": fold.fold_index,
                "train_count": len(fold.train_ids),
                "val_count": len(fold.val_ids),
                "val_class_counts": {str(class_id): fold.val_class_counts.get(class_id, 0) for class_id in range(NUM_CLASSES)},
            }
            for fold in splits
        ],
    }
    manifest_path = root / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    rows: list[str] = []
    for fold in splits:
        for split_name, ids in (("train", fold.train_ids), ("val", fold.val_ids)):
            for sample_id in ids:
                rows.append(
                    json.dumps(
                        {
                            "fold": fold.fold_index,
                            "sample_id": sample_id,
                            "split": split_name,
                        },
                        sort_keys=True,
                    )
                )
    (root / "fold_assignments.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return manifest_path


def _group_samples(labels_by_sample: Mapping[str, Sequence[int]]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for sample_id in sorted(labels_by_sample):
        groups.setdefault(_group_id(sample_id), []).append(sample_id)
    return groups


def _group_id(sample_id: str) -> str:
    parts = sample_id.split("_")
    if len(parts) >= 3:
        return "_".join(parts[:2])
    if len(parts) >= 2:
        return parts[0]
    return sample_id[:3]


def _imbalance_score(class_counts: Counter[int]) -> tuple[int, int]:
    values = [class_counts.get(class_id, 0) for class_id in range(NUM_CLASSES)]
    return max(values) - min(values), sum(values)
