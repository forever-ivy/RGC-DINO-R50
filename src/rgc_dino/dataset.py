"""Dataset discovery helpers for aligned RGB/IR/depth competition samples."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

MODALITIES: tuple[str, ...] = ("visible", "infrared", "depth")
IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class MultimodalSample:
    sample_id: str
    visible_path: Path
    infrared_path: Path
    depth_path: Path
    label_path: Path | None = None

    @property
    def image_paths(self) -> Mapping[str, Path]:
        return {
            "visible": self.visible_path,
            "infrared": self.infrared_path,
            "depth": self.depth_path,
        }


@dataclass(frozen=True)
class DatasetSummary:
    root: Path
    labels_dir: Path | None
    aligned_count: int
    modality_counts: dict[str, int]
    missing_by_modality: dict[str, list[str]]
    labeled_aligned_count: int
    extra_label_ids: list[str]


def discover_aligned_samples(
    root: str | Path,
    *,
    labels_dir: str | Path | None = None,
    require_labels: bool = False,
) -> list[MultimodalSample]:
    """Return samples that have image files in all required modality folders."""
    root_path = Path(root)
    labels_path = Path(labels_dir) if labels_dir is not None else None
    modality_maps = {modality: _collect_modality_files(root_path, modality) for modality in MODALITIES}

    aligned_ids = set.intersection(*(set(paths) for paths in modality_maps.values()))
    label_ids = _collect_label_ids(labels_path) if labels_path is not None else {}
    if require_labels:
        aligned_ids &= set(label_ids)

    samples: list[MultimodalSample] = []
    for sample_id in sorted(aligned_ids):
        samples.append(
            MultimodalSample(
                sample_id=sample_id,
                visible_path=modality_maps["visible"][sample_id],
                infrared_path=modality_maps["infrared"][sample_id],
                depth_path=modality_maps["depth"][sample_id],
                label_path=label_ids.get(sample_id),
            )
        )
    return samples


def summarize_multimodal_dataset(
    root: str | Path,
    labels_dir: str | Path | None = None,
) -> DatasetSummary:
    """Summarize alignment gaps without reading image pixels."""
    root_path = Path(root)
    labels_path = Path(labels_dir) if labels_dir is not None else None
    modality_maps = {modality: _collect_modality_files(root_path, modality) for modality in MODALITIES}
    all_image_ids = set().union(*(set(paths) for paths in modality_maps.values()))
    aligned_ids = set.intersection(*(set(paths) for paths in modality_maps.values()))
    label_ids = _collect_label_ids(labels_path) if labels_path is not None else {}

    missing_by_modality: dict[str, list[str]] = {}
    for modality, paths in modality_maps.items():
        missing_by_modality[modality] = sorted(all_image_ids - set(paths))

    return DatasetSummary(
        root=root_path,
        labels_dir=labels_path,
        aligned_count=len(aligned_ids),
        modality_counts={modality: len(paths) for modality, paths in modality_maps.items()},
        missing_by_modality=missing_by_modality,
        labeled_aligned_count=len(aligned_ids & set(label_ids)),
        extra_label_ids=sorted(set(label_ids) - aligned_ids),
    )


def write_manifest_jsonl(samples: list[MultimodalSample], output_path: str | Path) -> Path:
    """Write aligned samples as JSON Lines for downstream scripts."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for sample in samples:
        lines.append(
            json.dumps(
                {
                    "sample_id": sample.sample_id,
                    "modalities": {
                        modality: str(modality_path)
                        for modality, modality_path in sample.image_paths.items()
                    },
                    "label_path": str(sample.label_path) if sample.label_path is not None else None,
                },
                sort_keys=True,
            )
        )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


def _collect_modality_files(root: Path, modality: str) -> dict[str, Path]:
    modality_dir = root / modality
    if not modality_dir.exists():
        return {}

    files: dict[str, Path] = {}
    for path in sorted(modality_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        sample_id = path.stem
        if sample_id in files:
            raise ValueError(f"duplicate sample id {sample_id!r} in {modality_dir}")
        files[sample_id] = path
    return files


def _collect_label_ids(labels_dir: Path | None) -> dict[str, Path]:
    if labels_dir is None or not labels_dir.exists():
        return {}
    return {path.stem: path for path in sorted(labels_dir.glob("*.txt"))}
