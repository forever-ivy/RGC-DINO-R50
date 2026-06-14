"""Lightweight side encoders for auxiliary modalities."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LightweightSideEncoder(nn.Module):
    """Encode one auxiliary modality into DINO-aligned multi-scale features."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        channels: int = 256,
        num_levels: int = 4,
        base_channels: int = 32,
        collapse_input_channels: bool = True,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if channels <= 0:
            raise ValueError("channels must be positive")
        if num_levels <= 0:
            raise ValueError("num_levels must be positive")
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")

        self.in_channels = in_channels
        self.channels = channels
        self.num_levels = num_levels
        self.collapse_input_channels = collapse_input_channels

        stages: list[nn.Module] = []
        projectors: list[nn.Module] = []
        current_channels = in_channels
        for level in range(num_levels):
            stage_channels = min(base_channels * (2**level), channels)
            stages.append(
                nn.Sequential(
                    nn.Conv2d(current_channels, stage_channels, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(num_groups=_group_count(stage_channels), num_channels=stage_channels),
                    nn.GELU(),
                    nn.Conv2d(stage_channels, stage_channels, kernel_size=3, padding=1),
                    nn.GroupNorm(num_groups=_group_count(stage_channels), num_channels=stage_channels),
                    nn.GELU(),
                )
            )
            projectors.append(nn.Conv2d(stage_channels, channels, kernel_size=1))
            current_channels = stage_channels

        self.stages = nn.ModuleList(stages)
        self.projectors = nn.ModuleList(projectors)

    def forward(
        self,
        image: Tensor,
        *,
        reference_features: list[Tensor] | tuple[Tensor, ...] | None = None,
    ) -> list[Tensor]:
        if image.ndim != 4:
            raise ValueError(f"expected image shape BCHW, got {tuple(image.shape)}")
        if self.collapse_input_channels and image.shape[1] != self.in_channels:
            image = image.mean(dim=1, keepdim=True)
        if image.shape[1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} input channels, got {image.shape[1]}")
        if reference_features is not None and len(reference_features) != self.num_levels:
            raise ValueError(f"expected {self.num_levels} reference feature levels, got {len(reference_features)}")

        x = image
        outputs: list[Tensor] = []
        for level, (stage, projector) in enumerate(zip(self.stages, self.projectors)):
            x = stage(x)
            projected = projector(x)
            if reference_features is not None:
                reference = reference_features[level]
                if reference.ndim != 4:
                    raise ValueError(f"reference level {level} must be BCHW, got {tuple(reference.shape)}")
                projected = F.interpolate(
                    projected,
                    size=reference.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            outputs.append(projected)
        return outputs


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1
