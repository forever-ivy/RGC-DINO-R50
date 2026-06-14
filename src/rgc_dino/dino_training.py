"""Training helpers shared by RGC-DINO command-line entrypoints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class CheckpointLoadReport:
    checkpoint_path: Path
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    skipped_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrainingStateLoadReport:
    checkpoint_path: Path
    start_epoch: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    optimizer_loaded: bool
    lr_scheduler_loaded: bool


def load_checkpoint_into_model(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    state_key: str = "model",
    map_location: str | torch.device = "cpu",
    strip_prefix: str | None = None,
    skip_mismatched_shapes: bool = False,
    weights_only: bool = False,
) -> CheckpointLoadReport:
    """Load a checkpoint payload into ``model`` with a concise report."""
    path = Path(checkpoint_path)
    payload = torch.load(path, map_location=map_location, weights_only=weights_only)
    state_dict = _extract_state_dict(payload, state_key=state_key)
    if strip_prefix:
        state_dict = _strip_prefix(state_dict, strip_prefix)
    skipped_keys: tuple[str, ...] = ()
    if skip_mismatched_shapes:
        state_dict, skipped_keys = _drop_mismatched_shapes(model, state_dict)
    incompatible = model.load_state_dict(state_dict, strict=False)
    return CheckpointLoadReport(
        checkpoint_path=path,
        missing_keys=tuple(incompatible.missing_keys),
        unexpected_keys=tuple(incompatible.unexpected_keys),
        skipped_keys=skipped_keys,
    )


def load_training_state(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    state_key: str = "model",
    map_location: str | torch.device = "cpu",
    weights_only: bool = False,
) -> TrainingStateLoadReport:
    """Restore model and optional optimizer/scheduler state for resumed training."""
    path = Path(checkpoint_path)
    payload = torch.load(path, map_location=map_location, weights_only=weights_only)
    state_dict = _extract_state_dict(payload, state_key=state_key)
    incompatible = model.load_state_dict(state_dict, strict=False)

    optimizer_loaded = False
    lr_scheduler_loaded = False
    if isinstance(payload, Mapping):
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
            optimizer_loaded = True
        if lr_scheduler is not None and "lr_scheduler" in payload:
            lr_scheduler.load_state_dict(payload["lr_scheduler"])
            lr_scheduler_loaded = True
        start_epoch = int(payload.get("epoch", -1)) + 1
    else:
        start_epoch = 0

    return TrainingStateLoadReport(
        checkpoint_path=path,
        start_epoch=start_epoch,
        missing_keys=tuple(incompatible.missing_keys),
        unexpected_keys=tuple(incompatible.unexpected_keys),
        optimizer_loaded=optimizer_loaded,
        lr_scheduler_loaded=lr_scheduler_loaded,
    )


def move_targets_to_device(
    targets: Sequence[Mapping[str, Any]],
    device: torch.device | str,
) -> list[dict[str, Any]]:
    """Move tensor values in DINO target dictionaries to a device."""
    moved: list[dict[str, Any]] = []
    for target in targets:
        moved.append(
            {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in target.items()
            }
        )
    return moved


def _extract_state_dict(payload: Any, *, state_key: str) -> Mapping[str, Tensor]:
    if isinstance(payload, Mapping) and state_key in payload:
        payload = payload[state_key]
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint payload is not a state dict or a mapping containing one")
    return payload


def _strip_prefix(state_dict: Mapping[str, Tensor], prefix: str) -> dict[str, Tensor]:
    return {
        key[len(prefix) :] if key.startswith(prefix) else key: value
        for key, value in state_dict.items()
    }


def _drop_mismatched_shapes(
    model: nn.Module,
    state_dict: Mapping[str, Tensor],
) -> tuple[dict[str, Tensor], tuple[str, ...]]:
    target_state = model.state_dict()
    filtered: dict[str, Tensor] = {}
    skipped: list[str] = []
    for key, value in state_dict.items():
        target_value = target_state.get(key)
        if target_value is not None and tuple(value.shape) != tuple(target_value.shape):
            skipped.append(key)
            continue
        filtered[key] = value
    return filtered, tuple(sorted(skipped))
