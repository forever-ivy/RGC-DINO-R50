"""Helpers for checking a local Co-DETR integration tree.

The project keeps third-party detector code under ``external/`` and never
starts heavy training from integration checks.  These helpers only verify that
an offline Co-DETR checkout has the minimum structure needed for later smoke
scripts and that optional pretrained-weight paths exist when explicitly
required.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_CODETR_ROOT = Path("external") / "Co-DETR"
DEFAULT_REQUIRED_PATHS: tuple[Path, ...] = (
    Path("tools") / "train.py",
    Path("tools") / "test.py",
    Path("tools") / "dist_train.sh",
    Path("mmdet"),
)
DEFAULT_CONFIG_CANDIDATES: tuple[Path, ...] = (
    Path("projects") / "configs",
    Path("configs"),
)
DEFAULT_CLONE_HINT = "git clone https://github.com/Sense-X/Co-DETR external/Co-DETR"


@dataclass(frozen=True)
class CodetrIntegrationStatus:
    root: Path
    ok: bool
    missing_paths: tuple[Path, ...]
    missing_weight_paths: tuple[Path, ...]
    messages: tuple[str, ...]
    clone_hint: str = DEFAULT_CLONE_HINT


def check_codetr_tree(
    root: str | Path = DEFAULT_CODETR_ROOT,
    *,
    required_paths: Iterable[str | Path] = DEFAULT_REQUIRED_PATHS,
    config_candidates: Sequence[str | Path] = DEFAULT_CONFIG_CANDIDATES,
    weight_paths: Iterable[str | Path] = (),
    require_weights: bool = False,
) -> CodetrIntegrationStatus:
    """Return an actionable status for a local Co-DETR checkout.

    ``weight_paths`` are treated as informational unless ``require_weights`` is
    true.  This lets a first smoke check validate the repository layout before
    the user has downloaded large public pretrained weights.
    """

    root_path = Path(root)
    if not root_path.exists():
        return CodetrIntegrationStatus(
            root=root_path,
            ok=False,
            missing_paths=(root_path,),
            missing_weight_paths=(),
            messages=(f"Co-DETR root not found: {root_path}",),
        )

    required = tuple(Path(path) for path in required_paths)
    missing = [relative for relative in required if not (root_path / relative).exists()]

    config_options = tuple(Path(path) for path in config_candidates)
    if config_options and not any((root_path / candidate).exists() for candidate in config_options):
        missing.append(Path("<one of: " + ", ".join(str(path) for path in config_options) + ">"))

    missing_weights = tuple(Path(path) for path in weight_paths if path and not Path(path).exists())

    messages: list[str] = []
    messages.extend(f"missing required Co-DETR path: {root_path / relative}" for relative in missing)
    if missing_weights:
        prefix = "missing required" if require_weights else "missing optional"
        messages.extend(f"{prefix} pretrained weight: {path}" for path in missing_weights)
    if not messages:
        messages.append(f"Co-DETR integration tree looks complete: {root_path}")

    ok = not missing and (not require_weights or not missing_weights)
    return CodetrIntegrationStatus(
        root=root_path,
        ok=ok,
        missing_paths=tuple(missing),
        missing_weight_paths=missing_weights,
        messages=tuple(messages),
    )


def format_status(status: CodetrIntegrationStatus) -> str:
    lines = [f"codetr_root: {status.root}", f"ok: {str(status.ok).lower()}"]
    lines.extend(status.messages)
    if not status.ok and status.root in status.missing_paths:
        lines.append(f"clone_hint: {status.clone_hint}")
    return "\n".join(lines)
