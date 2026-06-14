"""Safe promotion helpers for leaderboard candidate ZIP files."""

from __future__ import annotations

from dataclasses import dataclass
import json
import shutil
from pathlib import Path
from typing import Any
import zipfile

from rgc_dino.submission_manifest import file_sha256


BLOCKED_NAME_TOKENS = ("val", "valid", "validation", "oof", "debug", "empty")


@dataclass(frozen=True)
class PromotedSubmission:
    zip_path: Path
    metadata_path: Path
    zip_sha256: str


def promote_submission_candidate(
    *,
    candidate_zip: str | Path,
    submissions_dir: str | Path,
    reason: str,
    local_map: float | None = None,
    leaderboard_baseline: float | None = None,
    force: bool = False,
) -> PromotedSubmission:
    """Copy a deliberate test-set ZIP into the monitor-watched submissions dir."""
    source = Path(candidate_zip)
    destination_dir = Path(submissions_dir)
    _validate_candidate_zip(source, force=force)
    if not reason.strip():
        raise ValueError("promotion reason must not be empty")

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
    }
    metadata_path = destination.with_suffix(".promotion.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return PromotedSubmission(
        zip_path=destination,
        metadata_path=metadata_path,
        zip_sha256=zip_sha,
    )


def _validate_candidate_zip(path: Path, *, force: bool) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() != ".zip":
        raise ValueError(f"candidate must be a .zip file: {path}")
    if path.stat().st_size <= 0:
        raise ValueError(f"candidate zip is empty: {path}")
    lowered_parts = {part.lower() for part in path.with_suffix("").name.replace("-", "_").split("_")}
    blocked = sorted(lowered_parts.intersection(BLOCKED_NAME_TOKENS))
    if blocked and not force:
        raise ValueError(f"candidate name looks non-submittable ({blocked}): {path}")
    with zipfile.ZipFile(path) as archive:
        txt_files = [
            info.filename
            for info in archive.infolist()
            if not info.is_dir() and Path(info.filename).suffix.lower() == ".txt"
        ]
    root_txt_files = [name for name in txt_files if "/" not in name.strip("/")]
    if not root_txt_files:
        raise ValueError("candidate zip must contain prediction .txt files at archive root")


def metadata_as_json(result: PromotedSubmission) -> dict[str, Any]:
    return json.loads(result.metadata_path.read_text(encoding="utf-8"))
