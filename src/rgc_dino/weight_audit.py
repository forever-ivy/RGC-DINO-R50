"""Pretrained-weight load-report checks for high-ceiling experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class WeightLoadAudit:
    ok: bool
    fatal_missing_keys: tuple[str, ...]
    fatal_skipped_keys: tuple[str, ...]
    messages: tuple[str, ...]


def audit_load_report(
    *,
    missing_keys: Sequence[str],
    skipped_keys: Sequence[str] = (),
    fatal_missing_patterns: Iterable[str] = (),
    fatal_skipped_patterns: Iterable[str] = (),
    allowed_missing_patterns: Iterable[str] = (),
) -> WeightLoadAudit:
    """Classify a checkpoint-load report into pass/fail diagnostics.

    Large-backbone experiments must fail fast when core backbone stages are not
    initialized.  Class-dependent detector heads may be deliberately missing or
    skipped, so callers can pass those prefixes via ``allowed_missing_patterns``.
    Matching is substring-based to stay robust across DINO/MMDetection key
    prefixes such as ``backbone.patch_embed`` vs ``module.backbone.patch_embed``.
    """

    allowed = tuple(allowed_missing_patterns)
    fatal_missing = tuple(
        key
        for key in missing_keys
        if _matches_any(key, fatal_missing_patterns) and not _matches_any(key, allowed)
    )
    fatal_skipped = tuple(
        key
        for key in skipped_keys
        if _matches_any(key, fatal_skipped_patterns) and not _matches_any(key, allowed)
    )

    messages: list[str] = []
    if fatal_missing:
        messages.append(
            "fatal missing pretrained keys: "
            + ", ".join(fatal_missing[:8])
            + (" ..." if len(fatal_missing) > 8 else "")
        )
    if fatal_skipped:
        messages.append(
            "fatal skipped pretrained keys: "
            + ", ".join(fatal_skipped[:8])
            + (" ..." if len(fatal_skipped) > 8 else "")
        )
    if not messages:
        messages.append("pretrained load report passed fatal-pattern audit")

    return WeightLoadAudit(
        ok=not fatal_missing and not fatal_skipped,
        fatal_missing_keys=fatal_missing,
        fatal_skipped_keys=fatal_skipped,
        messages=tuple(messages),
    )


def _matches_any(key: str, patterns: Iterable[str]) -> bool:
    return any(pattern and pattern in key for pattern in patterns)
