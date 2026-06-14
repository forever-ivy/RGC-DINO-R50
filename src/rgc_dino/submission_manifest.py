"""Submission manifest helpers for reproducible leaderboard artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class SubmissionManifest:
    zip_path: str
    zip_sha256: str
    checkpoint_path: str
    checkpoint_sha256: str
    git_commit: str
    split_manifest_path: str
    split_manifest_sha256: str
    calibrator_version: str
    config_path: str


def build_submission_manifest(
    *,
    zip_path: str | Path,
    checkpoint_path: str | Path,
    git_commit: str,
    split_manifest_path: str | Path,
    calibrator_version: str,
    config_path: str | Path,
) -> SubmissionManifest:
    zip_file = Path(zip_path)
    checkpoint_file = Path(checkpoint_path)
    split_file = Path(split_manifest_path)
    return SubmissionManifest(
        zip_path=str(zip_file),
        zip_sha256=file_sha256(zip_file),
        checkpoint_path=str(checkpoint_file),
        checkpoint_sha256=file_sha256(checkpoint_file),
        git_commit=git_commit,
        split_manifest_path=str(split_file),
        split_manifest_sha256=file_sha256(split_file),
        calibrator_version=calibrator_version,
        config_path=str(config_path),
    )


def write_submission_manifest(path: str | Path, manifest: SubmissionManifest) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
