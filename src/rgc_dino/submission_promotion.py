"""Safe promotion helpers for leaderboard candidate ZIP files."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import shutil
from pathlib import Path
from typing import Any, Iterable, Sequence
import zipfile

from rgc_dino.labels import load_label_file
from rgc_dino.submission_manifest import file_sha256


BLOCKED_NAME_TOKENS = (
    "val",
    "valid",
    "validation",
    "oof",
    "debug",
    "empty",
    "partial",
    "sample",
    "toy",
    "tmp",
    "fold_eval",
)

DEFAULT_READY_FOR_SUBMIT = True


@dataclass(frozen=True)
class PromotedSubmission:
    zip_path: Path
    metadata_path: Path
    zip_sha256: str


@dataclass(frozen=True)
class ZipValidationSummary:
    root_txt_count: int
    nested_txt_count: int
    expected_txt_count: int | None
    prediction_objects: int


def promote_submission_candidate(
    *,
    candidate_zip: str | Path,
    submissions_dir: str | Path,
    reason: str,
    local_map: float | None = None,
    leaderboard_baseline: float | None = None,
    force: bool = False,
    manifest_path: str | Path | None = None,
    expected_ids: Sequence[str] | None = None,
    dataset_root: str | Path | None = None,
    candidate_kind: str = "manual",
    checkpoint_path: str | Path | None = None,
    epoch: int | None = None,
    train_dir: str | Path | None = None,
    val_map_50_95: float | None = None,
    val_map_50: float | None = None,
    prediction_objects: int | None = None,
    score_threshold: float | None = None,
    class_score_thresholds_path: str | Path | None = None,
    class_score_thresholds: Sequence[float] | None = None,
    candidate_score_threshold: float | None = None,
    nms_iou_threshold: float | None = None,
    pre_limit_per_image: int | None = None,
    max_detections: int | None = None,
    image_max_side: int | None = None,
    config_path: str | Path | None = None,
    quality_cache: str | Path | None = None,
    split_manifest: str | Path | None = None,
    git_commit: str | None = None,
    source_ranking_json: str | Path | None = None,
    source_sweep_ranking_json: str | Path | None = None,
    diagnostics_path: str | Path | None = None,
    hard_val_report_path: str | Path | None = None,
    conversion_summary_path: str | Path | None = None,
    threshold_sweep_json: str | Path | None = None,
    raw_prediction_summary: dict[str, Any] | None = None,
    after_class_threshold_summary: dict[str, Any] | None = None,
    after_nms_summary: dict[str, Any] | None = None,
    after_topk_summary: dict[str, Any] | None = None,
    hard_val_status: str | None = None,
    hard_val_map_50_95: float | None = None,
    ready_for_submit: bool = DEFAULT_READY_FOR_SUBMIT,
    extra_metadata: dict[str, Any] | None = None,
) -> PromotedSubmission:
    """Copy a deliberate test-set ZIP into the monitor-watched submissions dir."""
    source = Path(candidate_zip)
    destination_dir = Path(submissions_dir)
    expected = list(expected_ids) if expected_ids is not None else _sample_ids_from_dataset(dataset_root)
    validation_summary = _validate_candidate_zip(source, force=force, expected_ids=expected)
    if not reason.strip():
        raise ValueError("promotion reason must not be empty")

    resolved_manifest = _resolve_manifest_path(source, manifest_path=manifest_path, force=force)
    source_sha = file_sha256(source)
    manifest_data = _load_manifest(resolved_manifest)
    if manifest_data is not None:
        manifest_sha = manifest_data.get("zip_sha256")
        if manifest_sha and manifest_sha != source_sha and not force:
            raise ValueError(
                f"manifest zip_sha256 does not match candidate: {manifest_sha} != {source_sha}"
            )

    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source.name
    if destination.exists() and not force:
        raise FileExistsError(f"submission target already exists: {destination}")

    shutil.copy2(source, destination)
    zip_sha = file_sha256(destination)
    metadata = {
        "candidate_zip": str(source),
        "promoted_zip": str(destination),
        "zip_sha256": zip_sha,
        "reason": reason,
        "local_map": local_map,
        "leaderboard_baseline": leaderboard_baseline,
        "candidate_kind": candidate_kind,
        "checkpoint_path": _string_or_none(checkpoint_path),
        "checkpoint_sha256": _sha256_or_none(checkpoint_path),
        "epoch": epoch,
        "train_dir": _string_or_none(train_dir),
        "val_map_50_95": val_map_50_95 if val_map_50_95 is not None else local_map,
        "val_map_50": val_map_50,
        "prediction_objects": prediction_objects if prediction_objects is not None else validation_summary.prediction_objects,
        "score_threshold": score_threshold,
        "class_score_thresholds_path": _string_or_none(class_score_thresholds_path),
        "class_score_thresholds": list(class_score_thresholds) if class_score_thresholds is not None else None,
        "candidate_score_threshold": candidate_score_threshold if candidate_score_threshold is not None else score_threshold,
        "nms_iou_threshold": nms_iou_threshold,
        "pre_limit_per_image": pre_limit_per_image,
        "max_detections": max_detections,
        "image_max_side": image_max_side,
        "config_path": _string_or_none(config_path),
        "quality_cache": _string_or_none(quality_cache),
        "split_manifest": _string_or_none(split_manifest),
        "git_commit": git_commit,
        "manifest_path": str(resolved_manifest) if resolved_manifest is not None else None,
        "manifest_sha256": file_sha256(resolved_manifest) if resolved_manifest is not None else None,
        "source_ranking_json": _string_or_none(source_ranking_json),
        "source_sweep_ranking_json": _string_or_none(source_sweep_ranking_json),
        "threshold_sweep_json": _string_or_none(threshold_sweep_json),
        "diagnostics_path": _string_or_none(diagnostics_path),
        "hard_val_report_path": _string_or_none(hard_val_report_path),
        "conversion_summary_path": _string_or_none(conversion_summary_path),
        "raw_prediction_summary": raw_prediction_summary,
        "after_class_threshold_summary": after_class_threshold_summary,
        "after_nms_summary": after_nms_summary,
        "after_topk_summary": after_topk_summary,
        "hard_val_status": hard_val_status,
        "hard_val_map_50_95": hard_val_map_50_95,
        "created_at": datetime.now().isoformat(),
        "ready_for_submit": ready_for_submit,
        "submission_guard": {
            "root_txt_count": validation_summary.root_txt_count,
            "nested_txt_count": validation_summary.nested_txt_count,
            "expected_txt_count": validation_summary.expected_txt_count,
            "manifest_required": not force,
            "manifest_present": resolved_manifest is not None,
        },
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    metadata_path = destination.with_suffix(".promotion.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return PromotedSubmission(
        zip_path=destination,
        metadata_path=metadata_path,
        zip_sha256=zip_sha,
    )


def _validate_candidate_zip(
    path: Path,
    *,
    force: bool,
    expected_ids: Sequence[str] | None = None,
) -> ZipValidationSummary:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() != ".zip":
        raise ValueError(f"candidate must be a .zip file: {path}")
    if path.stat().st_size <= 0:
        raise ValueError(f"candidate zip is empty: {path}")
    lowered_parts = _name_tokens(path.with_suffix("").name)
    blocked = sorted(lowered_parts.intersection(BLOCKED_NAME_TOKENS))
    if blocked and not force:
        raise ValueError(f"candidate name looks non-submittable ({blocked}): {path}")

    prediction_objects = 0
    with zipfile.ZipFile(path) as archive:
        txt_files = [
            info.filename
            for info in archive.infolist()
            if not info.is_dir() and Path(info.filename).suffix.lower() == ".txt"
        ]
        root_txt_files = [name for name in txt_files if "/" not in name.strip("/")]
        nested_txt_files = [name for name in txt_files if "/" in name.strip("/")]
        if not root_txt_files:
            raise ValueError("candidate zip must contain prediction .txt files at archive root")
        if nested_txt_files and not force:
            raise ValueError(f"candidate zip contains nested prediction txt files: {nested_txt_files[:5]}")
        if expected_ids is not None:
            expected_names = {f"{sample_id}.txt" for sample_id in expected_ids}
            actual_names = {Path(name).name for name in root_txt_files}
            missing = sorted(expected_names - actual_names)
            extra = sorted(actual_names - expected_names)
            if missing and not force:
                raise ValueError(f"candidate zip is missing required txt files: {missing[:5]}")
            if extra and not force:
                raise ValueError(f"candidate zip contains unexpected txt files: {extra[:5]}")
        for name in root_txt_files:
            with archive.open(name) as handle:
                text = handle.read().decode("utf-8")
            if text.strip():
                with _temporary_label_file(path, name, text) as label_path:
                    prediction_objects += len(load_label_file(label_path, require_confidence=True))
    return ZipValidationSummary(
        root_txt_count=len(root_txt_files),
        nested_txt_count=len(nested_txt_files),
        expected_txt_count=len(expected_ids) if expected_ids is not None else None,
        prediction_objects=prediction_objects,
    )


class _temporary_label_file:
    def __init__(self, zip_path: Path, name: str, text: str) -> None:
        self.path = zip_path.parent / f".{zip_path.stem}.{Path(name).stem}.validate.tmp"
        self.text = text

    def __enter__(self) -> Path:
        self.path.write_text(self.text, encoding="utf-8")
        return self.path

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def _resolve_manifest_path(
    candidate_zip: Path,
    *,
    manifest_path: str | Path | None,
    force: bool,
) -> Path | None:
    candidates: list[Path] = []
    if manifest_path is not None:
        candidates.append(Path(manifest_path))
    candidates.extend(
        [
            candidate_zip.with_suffix(".manifest.json"),
            candidate_zip.with_name(candidate_zip.name + ".manifest.json"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if force:
        return None
    raise FileNotFoundError(
        f"submission manifest is required; expected one of: {', '.join(str(p) for p in candidates)}"
    )


def _load_manifest(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _sample_ids_from_dataset(dataset_root: str | Path | None) -> list[str] | None:
    if dataset_root is None:
        return None
    root = Path(dataset_root)
    visible = root / "visible"
    search_root = visible if visible.exists() else root
    ids = sorted(path.stem for path in search_root.iterdir() if path.is_file() and not path.name.startswith("."))
    return ids or None


def _name_tokens(name: str) -> set[str]:
    normalized = name.lower().replace("-", "_").replace(".", "_")
    parts = set(normalized.split("_"))
    if "fold" in parts and "eval" in parts:
        parts.add("fold_eval")
    return parts


def _string_or_none(value: str | Path | None) -> str | None:
    return str(value) if value is not None else None


def _sha256_or_none(value: str | Path | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    return file_sha256(path) if path.exists() else None


def metadata_as_json(result: PromotedSubmission) -> dict[str, Any]:
    return json.loads(result.metadata_path.read_text(encoding="utf-8"))
