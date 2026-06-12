"""Helpers for checking the external IDEA-Research DINO tree."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

DEFAULT_DINO_ROOT = Path("external") / "IDEA-Research-DINO"
DEFAULT_OFFICIAL_CONFIG = Path("config") / "DINO" / "DINO_4scale.py"
DEFAULT_REQUIRED_PATHS: tuple[Path, ...] = (
    Path("main.py"),
    Path("models") / "dino",
    Path("models") / "dino" / "ops",
)
DEFAULT_CLONE_HINT = "git clone https://github.com/IDEA-Research/DINO external/IDEA-Research-DINO"


@dataclass(frozen=True)
class DinoIntegrationStatus:
    root: Path
    ok: bool
    missing_paths: tuple[Path, ...]
    messages: tuple[str, ...]
    clone_hint: str = DEFAULT_CLONE_HINT


def check_dino_tree(
    root: str | Path = DEFAULT_DINO_ROOT,
    *,
    required_paths: Iterable[str | Path] = DEFAULT_REQUIRED_PATHS,
    official_config: str | Path = DEFAULT_OFFICIAL_CONFIG,
) -> DinoIntegrationStatus:
    root_path = Path(root)
    if not root_path.exists():
        return DinoIntegrationStatus(
            root=root_path,
            ok=False,
            missing_paths=(root_path,),
            messages=(f"DINO root not found: {root_path}",),
        )

    required = tuple(Path(path) for path in required_paths) + (Path(official_config),)
    missing = tuple(relative for relative in required if not (root_path / relative).exists())
    messages = tuple(f"missing required DINO path: {root_path / relative}" for relative in missing)
    if not messages:
        messages = (f"DINO integration tree looks complete: {root_path}",)
    return DinoIntegrationStatus(
        root=root_path,
        ok=not missing,
        missing_paths=missing,
        messages=messages,
    )


def format_status(status: DinoIntegrationStatus) -> str:
    lines = [f"dino_root: {status.root}", f"ok: {str(status.ok).lower()}"]
    lines.extend(status.messages)
    if not status.ok:
        lines.append(f"clone_hint: {status.clone_hint}")
    return "\n".join(lines)
