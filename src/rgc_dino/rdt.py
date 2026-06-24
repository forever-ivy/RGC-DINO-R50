"""RGB-guided RDT saliency utilities for lightweight multimodal diagnostics.

These helpers mirror the useful part of the opponent RGB-guided-RDT idea while
staying dependency-light: RGB remains the primary image, IR and depth only build
an auxiliary spatial attention/gain map.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image, ImageOps

from .constants import DEPTH_VALID_MAX_MM, DEPTH_VALID_MIN_MM


@dataclass(frozen=True)
class RdtResult:
    rgb: np.ndarray
    infrared: np.ndarray
    infrared_enhanced: np.ndarray
    depth_normalized: np.ndarray
    depth_valid_mask: np.ndarray
    ir_attention: np.ndarray
    depth_attention: np.ndarray
    attention: np.ndarray
    guided_rgb: np.ndarray
    stats: dict[str, float]


def load_rdt_result(
    visible_path: str | Path,
    infrared_path: str | Path,
    depth_path: str | Path,
    *,
    max_side: int | None = None,
    ir_weight: float = 0.55,
    depth_weight: float = 0.45,
    base_gate: float = 0.85,
    gain: float = 0.30,
) -> RdtResult:
    """Load three modality files and compute RGB-guided RDT diagnostics."""
    rgb_image = _open_image(visible_path, mode="RGB", max_side=max_side, resample=Image.Resampling.BILINEAR)
    ir_image = _open_image(infrared_path, mode="L", max_side=max_side, resample=Image.Resampling.BILINEAR)
    depth_image = _open_image(depth_path, mode=None, max_side=max_side, resample=Image.Resampling.NEAREST)
    return compute_rdt_result(
        np.asarray(rgb_image),
        np.asarray(ir_image),
        np.asarray(depth_image),
        ir_weight=ir_weight,
        depth_weight=depth_weight,
        base_gate=base_gate,
        gain=gain,
    )


def compute_rdt_result(
    rgb: np.ndarray,
    infrared: np.ndarray,
    depth: np.ndarray,
    *,
    ir_weight: float = 0.55,
    depth_weight: float = 0.45,
    base_gate: float = 0.85,
    gain: float = 0.30,
) -> RdtResult:
    """Compute RGB-guided-RDT image, attention maps, and scalar stats."""
    rgb_u8 = _to_rgb_uint8(rgb)
    ir_u8 = _to_gray_uint8(infrared)
    depth_norm, valid_mask = normalize_depth_to_uint8(depth)
    if rgb_u8.shape[:2] != ir_u8.shape or rgb_u8.shape[:2] != depth_norm.shape:
        raise ValueError(
            "modality shapes must match: "
            f"rgb={rgb_u8.shape[:2]} ir={ir_u8.shape} depth={depth_norm.shape}"
        )
    if ir_weight < 0.0 or depth_weight < 0.0 or ir_weight + depth_weight <= 0.0:
        raise ValueError("ir_weight and depth_weight must be non-negative with positive sum")

    ir_enhanced = enhance_infrared(ir_u8)
    ir_attention = _normalize_float01(ir_enhanced)
    near_depth = (1.0 - depth_norm.astype(np.float32) / 255.0) * valid_mask.astype(np.float32)
    depth_attention = _normalize_float01(near_depth)
    attention = (ir_weight * ir_attention + depth_weight * depth_attention) / (ir_weight + depth_weight)
    attention = _smooth_attention(attention)
    gate = np.clip(base_gate + gain * attention, 0.0, 2.0)
    guided_rgb = np.clip(rgb_u8.astype(np.float32) * gate[..., None], 0.0, 255.0).astype(np.uint8)
    stats = rdt_stats(
        ir_attention=ir_attention,
        depth_attention=depth_attention,
        attention=attention,
        depth_valid_mask=valid_mask,
        base_gate=base_gate,
        gain=gain,
    )
    return RdtResult(
        rgb=rgb_u8,
        infrared=ir_u8,
        infrared_enhanced=ir_enhanced,
        depth_normalized=depth_norm,
        depth_valid_mask=valid_mask,
        ir_attention=ir_attention,
        depth_attention=depth_attention,
        attention=attention,
        guided_rgb=guided_rgb,
        stats=stats,
    )


def enhance_infrared(gray: np.ndarray) -> np.ndarray:
    """Enhance infrared contrast with dependency-light histogram equalization."""
    gray_u8 = _to_gray_uint8(gray)
    return np.asarray(ImageOps.equalize(Image.fromarray(gray_u8, mode="L")), dtype=np.uint8)


def normalize_depth_to_uint8(
    depth: np.ndarray,
    *,
    min_depth: int = DEPTH_VALID_MIN_MM,
    max_depth: int = DEPTH_VALID_MAX_MM,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize depth maps and return a valid-depth mask.

    16-bit depth is treated as millimeters. If an input is already 8-bit or its
    observed max is <=255, it is treated as an image-scale depth map and nonzero
    pixels are valid.
    """
    if min_depth >= max_depth:
        raise ValueError("min_depth must be smaller than max_depth")
    arr = np.asarray(depth)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.size == 0:
        return np.zeros((1, 1), dtype=np.uint8), np.zeros((1, 1), dtype=bool)
    finite = arr[np.isfinite(arr)] if np.issubdtype(arr.dtype, np.floating) else arr
    observed_max = float(np.nanmax(finite)) if finite.size else 0.0
    if arr.dtype == np.uint8 or observed_max <= 255.0:
        normalized = _to_gray_uint8(arr)
        return normalized, normalized > 0

    depth_float = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    valid = (depth_float >= float(min_depth)) & (depth_float <= float(max_depth))
    clipped = np.clip(depth_float, float(min_depth), float(max_depth))
    normalized = (clipped - float(min_depth)) / float(max_depth - min_depth)
    normalized = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
    normalized[~valid] = 0
    return normalized, valid


def rdt_stats(
    *,
    ir_attention: np.ndarray,
    depth_attention: np.ndarray,
    attention: np.ndarray,
    depth_valid_mask: np.ndarray,
    base_gate: float,
    gain: float,
) -> dict[str, float]:
    """Summarize RDT attention maps for gate-quality diagnostics."""
    gate = np.clip(base_gate + gain * attention.astype(np.float32), 0.0, 2.0)
    return {
        "rdt_ir_attention_mean": _mean(ir_attention),
        "rdt_ir_attention_std": _std(ir_attention),
        "rdt_depth_attention_mean": _mean(depth_attention),
        "rdt_depth_attention_std": _std(depth_attention),
        "rdt_attention_mean": _mean(attention),
        "rdt_attention_std": _std(attention),
        "rdt_attention_top10_mean": _top_fraction_mean(attention, 0.10),
        "rdt_attention_hot_ratio": _ratio(attention >= 0.75),
        "rdt_depth_valid_ratio": _ratio(depth_valid_mask),
        "rdt_gate_mean": _mean(gate),
        "rdt_gate_std": _std(gate),
    }


def write_rdt_preview(result: RdtResult, output_path: str | Path) -> Path:
    """Write a horizontal preview strip: RGB, IR, depth, attention, guided RGB."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    panels = [
        result.rgb,
        _gray_to_rgb(result.infrared_enhanced),
        _gray_to_rgb(result.depth_normalized),
        _gray_to_rgb((np.clip(result.attention, 0.0, 1.0) * 255).astype(np.uint8)),
        result.guided_rgb,
    ]
    height = max(panel.shape[0] for panel in panels)
    normalized = [_pad_to_height(panel, height) for panel in panels]
    strip = np.concatenate(normalized, axis=1)
    Image.fromarray(strip).save(output)
    return output


def write_rdt_stats(path: str | Path, stats_by_sample_id: Mapping[str, Mapping[str, float]]) -> None:
    import json

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(stats_by_sample_id, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _open_image(path: str | Path, *, mode: str | None, max_side: int | None, resample: int) -> Image.Image:
    with Image.open(path) as image:
        if mode is not None:
            image = image.convert(mode)
        else:
            image = image.copy()
    if max_side is None:
        return image
    if max_side <= 0:
        raise ValueError("max_side must be positive")
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image
    scale = float(max_side) / float(longest)
    return image.resize((max(1, round(width * scale)), max(1, round(height * scale))), resample=resample)


def _to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"expected RGB-like image, got shape {arr.shape}")
    arr = arr[..., :3]
    if arr.dtype == np.uint8:
        return arr.copy()
    arr = arr.astype(np.float32)
    max_value = float(np.nanmax(arr)) if arr.size else 0.0
    if max_value <= 1.0:
        arr = arr * 255.0
    return np.clip(np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0), 0.0, 255.0).astype(np.uint8)


def _to_gray_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3:
        arr = arr[..., :3].astype(np.float32).mean(axis=2)
    else:
        arr = arr.astype(np.float32)
    if arr.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    if np.issubdtype(np.asarray(image).dtype, np.integer):
        max_value = float(np.iinfo(np.asarray(image).dtype).max)
        if max_value > 0:
            arr = arr / max_value * 255.0
    else:
        max_value = float(np.nanmax(arr)) if np.isfinite(arr).any() else 0.0
        if max_value <= 1.0:
            arr = arr * 255.0
    return np.clip(np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0), 0.0, 255.0).astype(np.uint8)


def _normalize_float01(values: np.ndarray) -> np.ndarray:
    arr = values.astype(np.float32)
    if arr.size == 0:
        return np.zeros((1, 1), dtype=np.float32)
    min_value = float(np.nanmin(arr))
    max_value = float(np.nanmax(arr))
    if max_value <= min_value:
        return np.zeros(arr.shape, dtype=np.float32)
    return np.clip((arr - min_value) / (max_value - min_value), 0.0, 1.0).astype(np.float32)


def _smooth_attention(attention: np.ndarray) -> np.ndarray:
    arr = attention.astype(np.float32)
    if min(arr.shape[:2]) < 3:
        return arr
    padded = np.pad(arr, 1, mode="edge")
    windows = np.stack(
        [padded[dy : dy + arr.shape[0], dx : dx + arr.shape[1]] for dy in range(3) for dx in range(3)],
        axis=0,
    )
    return np.mean(windows, axis=0).astype(np.float32)


def _gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    arr = _to_gray_uint8(gray)
    return np.repeat(arr[:, :, None], 3, axis=2)


def _pad_to_height(image: np.ndarray, height: int) -> np.ndarray:
    if image.shape[0] == height:
        return image
    pad = height - image.shape[0]
    return np.pad(image, ((0, pad), (0, 0), (0, 0)), mode="constant", constant_values=0)


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else 0.0


def _std(values: np.ndarray) -> float:
    return float(np.std(values)) if values.size else 0.0


def _ratio(mask: np.ndarray) -> float:
    arr = np.asarray(mask)
    return float(np.mean(arr.astype(np.float32))) if arr.size else 0.0


def _top_fraction_mean(values: np.ndarray, fraction: float) -> float:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return 0.0
    count = max(1, int(round(arr.size * fraction)))
    return float(np.mean(np.partition(arr, arr.size - count)[-count:]))
