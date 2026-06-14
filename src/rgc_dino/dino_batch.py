"""Batch collation helpers for RGC-DINO training and evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .models.rgc_dino_adapter import RgcDinoSamples


@dataclass(frozen=True)
class NestedTensorBatch:
    """Minimal DINO-compatible padded tensor batch."""

    tensors: Tensor
    mask: Tensor

    def __post_init__(self) -> None:
        if self.tensors.ndim != 4:
            raise ValueError(f"tensors must be BCHW, got {tuple(self.tensors.shape)}")
        if self.mask.ndim != 3:
            raise ValueError(f"mask must be BHW, got {tuple(self.mask.shape)}")
        if self.mask.shape != self.tensors.shape[0:1] + self.tensors.shape[-2:]:
            raise ValueError("mask shape must match tensor batch and spatial dimensions")

    def to(self, device: torch.device | str) -> "NestedTensorBatch":
        return NestedTensorBatch(self.tensors.to(device), self.mask.to(device))

    def decompose(self) -> tuple[Tensor, Tensor]:
        return self.tensors, self.mask

    @property
    def shape(self) -> dict[str, torch.Size]:
        return {"tensors.shape": self.tensors.shape, "mask.shape": self.mask.shape}


def collate_rgc_dino_batch(
    batch: Iterable[tuple[Mapping[str, Tensor], Mapping[str, Any]]],
) -> tuple[RgcDinoSamples, list[Mapping[str, Any]]]:
    """Collate dataset items into the two-tuple expected by DINO's engine."""
    items = list(batch)
    if not items:
        raise ValueError("cannot collate an empty batch")

    sample_rows, target_rows = zip(*items)
    rgb = nested_tensor_from_tensor_list([_sample_tensor(sample, "rgb") for sample in sample_rows], name="rgb")
    infrared = nested_tensor_from_tensor_list(
        [_sample_tensor(sample, "infrared") for sample in sample_rows],
        name="infrared",
    )
    depth = nested_tensor_from_tensor_list([_sample_tensor(sample, "depth") for sample in sample_rows], name="depth")
    quality = torch.stack([_sample_tensor(sample, "quality") for sample in sample_rows])

    return (
        RgcDinoSamples(
            rgb=rgb,
            infrared=infrared.tensors,
            depth=depth.tensors,
            quality=quality,
        ),
        list(target_rows),
    )


def nested_tensor_from_tensor_list(tensors: Sequence[Tensor], *, name: str = "image") -> NestedTensorBatch:
    """Pad a list of CHW tensors into a DINO-style ``NestedTensor`` batch."""
    if not tensors:
        raise ValueError(f"{name} tensor list is empty")
    for index, tensor in enumerate(tensors):
        if not torch.is_tensor(tensor):
            raise TypeError(f"{name} item {index} must be a torch.Tensor")
        if tensor.ndim != 3:
            raise ValueError(f"{name} item {index} must be CHW, got {tuple(tensor.shape)}")
        if tensor.shape[0] != tensors[0].shape[0]:
            raise ValueError(f"{name} item {index} channel count does not match the first item")

    channels = tensors[0].shape[0]
    max_height = max(int(tensor.shape[1]) for tensor in tensors)
    max_width = max(int(tensor.shape[2]) for tensor in tensors)
    batch_tensor = torch.zeros(
        (len(tensors), channels, max_height, max_width),
        dtype=tensors[0].dtype,
        device=tensors[0].device,
    )
    mask = torch.ones((len(tensors), max_height, max_width), dtype=torch.bool, device=tensors[0].device)

    for index, tensor in enumerate(tensors):
        tensor = tensor.to(device=batch_tensor.device, dtype=batch_tensor.dtype)
        _, height, width = tensor.shape
        batch_tensor[index, :, :height, :width].copy_(tensor)
        mask[index, :height, :width] = False

    return NestedTensorBatch(batch_tensor, mask)


def _sample_tensor(sample: Mapping[str, Tensor], key: str) -> Tensor:
    try:
        tensor = sample[key]
    except KeyError as exc:
        raise KeyError(f"sample is missing required key: {key}") from exc
    if not torch.is_tensor(tensor):
        raise TypeError(f"sample[{key!r}] must be a torch.Tensor")
    return tensor
