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
        if not 0.0 <= gate_min <= gate_max <= 1.0:
            raise ValueError("gate bounds must satisfy 0 <= gate_min <= gate_max <= 1")

        self.channels = channels
        self.quality_dim = quality_dim
        self.num_levels = num_levels
        self.gate_min = gate_min
        self.gate_max = gate_max

        gate_input_dim = quality_dim + channels * 3
        self.gate_predictors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(gate_input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, 2),
                )
                for _ in range(num_levels)
            ]
        )
        self.ir_projections = nn.ModuleList([nn.Conv2d(channels, channels, kernel_size=1) for _ in range(num_levels)])
        self.depth_projections = nn.ModuleList([nn.Conv2d(channels, channels, kernel_size=1) for _ in range(num_levels)])
        self.residual_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=1),
                    nn.GELU(),
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1),
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

        for level, (rgb, infrared, depth) in enumerate(zip(rgb_features, infrared_features, depth_features)):
            pooled = torch.cat(
                [
                    _global_average(rgb),
                    _global_average(infrared),
                    _global_average(depth),
                    quality.to(dtype=rgb.dtype, device=rgb.device),
                ],
                dim=1,
            )
            raw_gate = self.gate_predictors[level](pooled)
            bounded_gate = self.gate_min + (self.gate_max - self.gate_min) * raw_gate.sigmoid()
            ir_gate = bounded_gate[:, 0].view(-1, 1, 1, 1)
            depth_gate = bounded_gate[:, 1].view(-1, 1, 1, 1)

            residual = (
                ir_gate * self.ir_projections[level](infrared)
                + depth_gate * self.depth_projections[level](depth)
            )
            fused_features.append(rgb + self.residual_blocks[level](residual))
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
