"""Reliability-gated residual fusion blocks."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn


class ReliabilityGatedResidualFusion(nn.Module):
    """Fuse IR/depth residual cues into RGB multi-scale DINO features."""

    def __init__(
        self,
        *,
        channels: int,
        quality_dim: int,
        num_levels: int,
        hidden_dim: int = 128,
        alpha_prior: float = 0.35,
        gate_min: float = 0.0,
        gate_max: float = 0.50,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if quality_dim <= 0:
            raise ValueError("quality_dim must be positive")
        if num_levels <= 0:
            raise ValueError("num_levels must be positive")
        if not 0.0 <= alpha_prior <= 1.0:
            raise ValueError("alpha_prior must be in [0, 1]")
        if not 0.0 <= gate_min <= gate_max <= 1.0:
            raise ValueError("gate bounds must satisfy 0 <= gate_min <= gate_max <= 1")

        self.channels = channels
        self.quality_dim = quality_dim
        self.num_levels = num_levels
        self.alpha_prior = alpha_prior
        self.gate_min = gate_min
        self.gate_max = gate_max
        self.register_buffer("quality_median", torch.zeros(quality_dim), persistent=True)
        self.register_buffer("quality_mad", torch.ones(quality_dim), persistent=True)

        gate_input_dim = quality_dim + channels * 3
        self.gate_predictors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(gate_input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, 3),
                )
                for _ in range(num_levels)
            ]
        )
        self.prior_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(quality_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, 3),
                )
                for _ in range(num_levels)
            ]
        )
        self.ir_projections = nn.ModuleList([nn.Conv2d(channels, channels, kernel_size=1) for _ in range(num_levels)])
        self.depth_projections = nn.ModuleList([nn.Conv2d(channels, channels, kernel_size=1) for _ in range(num_levels)])
        self.residual_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
                    nn.GroupNorm(num_groups=_group_count(channels), num_channels=channels),
                    nn.GELU(),
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
                    nn.GroupNorm(num_groups=_group_count(channels), num_channels=channels),
                    nn.GELU(),
                    nn.Conv2d(channels, channels, kernel_size=1),
                )
                for _ in range(num_levels)
            ]
        )
        self._zero_initialize_residual_outputs()

    def forward(
        self,
        rgb_features: list[Tensor] | tuple[Tensor, ...],
        infrared_features: list[Tensor] | tuple[Tensor, ...],
        depth_features: list[Tensor] | tuple[Tensor, ...],
        quality: Tensor,
        *,
        return_gates: bool = False,
    ) -> list[Tensor] | tuple[list[Tensor], Mapping[str, list[Tensor]]]:
        self._validate_inputs(rgb_features, infrared_features, depth_features, quality)

        fused_features: list[Tensor] = []
        ir_gates: list[Tensor] = []
        depth_gates: list[Tensor] = []

        normalized_quality = self.normalize_quality(quality)
        for level, (rgb, infrared, depth) in enumerate(zip(rgb_features, infrared_features, depth_features)):
            pooled = torch.cat(
                [
                    _global_average(rgb),
                    _global_average(infrared),
                    _global_average(depth),
                    normalized_quality.to(dtype=rgb.dtype, device=rgb.device),
                ],
                dim=1,
            )
            feature_gate = self.gate_predictors[level](pooled).softmax(dim=1)
            prior_gate = self.prior_heads[level](
                normalized_quality.to(dtype=rgb.dtype, device=rgb.device)
            ).softmax(dim=1)
            mixed_gate = (1.0 - self.alpha_prior) * feature_gate + self.alpha_prior * prior_gate
            bounded_gate = _clip_and_normalize_gates(mixed_gate, floor=self.gate_min, ceil=self.gate_max)
            ir_gate = bounded_gate[:, 1].view(-1, 1, 1, 1)
            depth_gate = bounded_gate[:, 2].view(-1, 1, 1, 1)

            residual = (
                ir_gate * self.ir_projections[level](infrared)
                + depth_gate * self.depth_projections[level](depth)
            )
            residual_input = torch.cat([rgb, residual, rgb * residual], dim=1)
            fused_features.append(rgb + self.residual_blocks[level](residual_input))
            ir_gates.append(ir_gate)
            depth_gates.append(depth_gate)

        if return_gates:
            return fused_features, {"ir": ir_gates, "depth": depth_gates}
        return fused_features

    def _validate_inputs(
        self,
        rgb_features: list[Tensor] | tuple[Tensor, ...],
        infrared_features: list[Tensor] | tuple[Tensor, ...],
        depth_features: list[Tensor] | tuple[Tensor, ...],
        quality: Tensor,
    ) -> None:
        if len(rgb_features) != self.num_levels:
            raise ValueError(f"expected {self.num_levels} rgb feature levels, got {len(rgb_features)}")
        if len(infrared_features) != self.num_levels:
            raise ValueError(f"expected {self.num_levels} infrared feature levels, got {len(infrared_features)}")
        if len(depth_features) != self.num_levels:
            raise ValueError(f"expected {self.num_levels} depth feature levels, got {len(depth_features)}")
        if quality.ndim != 2 or quality.shape[1] != self.quality_dim:
            raise ValueError(f"expected quality shape (batch, {self.quality_dim}), got {tuple(quality.shape)}")

        batch = quality.shape[0]
        for level, (rgb, infrared, depth) in enumerate(zip(rgb_features, infrared_features, depth_features)):
            expected_shape = rgb.shape
            if rgb.ndim != 4:
                raise ValueError(f"rgb level {level} must be BCHW, got {tuple(rgb.shape)}")
            if rgb.shape[0] != batch:
                raise ValueError(f"rgb level {level} batch does not match quality batch")
            if rgb.shape[1] != self.channels:
                raise ValueError(f"rgb level {level} has {rgb.shape[1]} channels, expected {self.channels}")
            if infrared.shape != expected_shape:
                raise ValueError(f"infrared level {level} shape {tuple(infrared.shape)} != rgb {tuple(expected_shape)}")
            if depth.shape != expected_shape:
                raise ValueError(f"depth level {level} shape {tuple(depth.shape)} != rgb {tuple(expected_shape)}")

    def set_quality_stats(self, *, median: Tensor, mad: Tensor) -> None:
        if median.shape != (self.quality_dim,):
            raise ValueError(f"median must have shape ({self.quality_dim},), got {tuple(median.shape)}")
        if mad.shape != (self.quality_dim,):
            raise ValueError(f"mad must have shape ({self.quality_dim},), got {tuple(mad.shape)}")
        self.quality_median.copy_(median.detach().to(device=self.quality_median.device, dtype=self.quality_median.dtype))
        safe_mad = mad.detach().to(device=self.quality_mad.device, dtype=self.quality_mad.dtype).clamp_min(1e-6)
        self.quality_mad.copy_(safe_mad)

    def normalize_quality(self, quality: Tensor) -> Tensor:
        if quality.ndim != 2 or quality.shape[1] != self.quality_dim:
            raise ValueError(f"expected quality shape (batch, {self.quality_dim}), got {tuple(quality.shape)}")
        median = self.quality_median.to(device=quality.device, dtype=quality.dtype)
        mad = self.quality_mad.to(device=quality.device, dtype=quality.dtype).clamp_min(1e-6)
        normalized = (quality - median.view(1, -1)) / (1.4826 * mad.view(1, -1) + 1e-6)
        return normalized.clamp(-3.0, 3.0)

    def _zero_initialize_residual_outputs(self) -> None:
        for block in self.residual_blocks:
            final_conv = block[-1]
            if not isinstance(final_conv, nn.Conv2d):
                raise TypeError("residual block must end with Conv2d")
            nn.init.zeros_(final_conv.weight)
            if final_conv.bias is not None:
                nn.init.zeros_(final_conv.bias)


def _global_average(feature: Tensor) -> Tensor:
    return feature.mean(dim=(2, 3))


def _clip_and_normalize_gates(gates: Tensor, *, floor: float, ceil: float) -> Tensor:
    clipped = gates.clamp(min=floor, max=ceil)
    return clipped / clipped.sum(dim=1, keepdim=True).clamp_min(1e-6)


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1
