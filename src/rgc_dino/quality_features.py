"""Static image-quality features for RGC gating priors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image

from .constants import DEPTH_VALID_MAX_MM, DEPTH_VALID_MIN_MM

QUALITY_FEATURE_NAMES: tuple[str, ...] = (
    "rgb_entropy",
    "rgb_brightness_mean",
    "rgb_brightness_std",
    "rgb_laplace_var",
    "rgb_edge_density",
    "rgb_overexposed_ratio",
    "rgb_underexposed_ratio",
    "rgb_local_contrast",
    "ir_entropy",
    "ir_mean",
    "ir_std",
    "ir_laplace_mean",
    "ir_tophat_mean",
    "ir_blackhat_mean",
    "ir_hot_ratio",
    "ir_cold_ratio",
    "depth_valid_ratio",
    "depth_hole_ratio",
    "depth_log_mean",
    "depth_log_std",
    "depth_inv_mean",
    "depth_edge_density",
    "depth_near_ratio",
    "depth_far_ratio",
)

RDT_QUALITY_FEATURE_NAMES: tuple[str, ...] = (
    "rdt_ir_attention_mean",
    "rdt_ir_attention_std",
    "rdt_depth_attention_mean",
    "rdt_depth_attention_std",
    "rdt_attention_mean",
    "rdt_attention_std",
    "rdt_attention_top10_mean",
    "rdt_attention_hot_ratio",
    "rdt_depth_valid_ratio",
    "rdt_gate_mean",
    "rdt_gate_std",
)

QUALITY_FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "base": QUALITY_FEATURE_NAMES,
    "base_rdt": QUALITY_FEATURE_NAMES + RDT_QUALITY_FEATURE_NAMES,
}


def load_quality_features(
    visible_path: str | Path,
    infrared_path: str | Path,
    depth_path: str | Path,
    *,
    max_side: int | None = None,
) -> dict[str, float]:
    """Load three modality images and compute the 24-D static quality vector."""
    visible_image = _open_quality_image(visible_path, max_side=max_side, resample=Image.Resampling.BILINEAR)
    infrared_image = _open_quality_image(infrared_path, max_side=max_side, resample=Image.Resampling.BILINEAR)
    depth_image = _open_quality_image(depth_path, max_side=max_side, resample=Image.Resampling.NEAREST)
    visible = np.asarray(visible_image)
    infrared = np.asarray(infrared_image)
    depth = np.asarray(depth_image)
    return compute_quality_features(visible, infrared, depth)


def feature_names_for_set(feature_set: str = "base") -> tuple[str, ...]:
    """Return the ordered quality feature names for a named ablation set."""
    try:
        return QUALITY_FEATURE_SETS[feature_set]
    except KeyError as exc:
        choices = ", ".join(sorted(QUALITY_FEATURE_SETS))
        raise ValueError(f"unknown quality feature set {feature_set!r}; choices: {choices}") from exc


def load_quality_feature_cache(
    path: str | Path,
    *,
    feature_set: str = "base",
) -> dict[str, dict[str, float]]:
    """Load a sample_id -> quality feature cache from JSON."""
    cache_path = Path(path)
    feature_names = feature_names_for_set(feature_set)
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("quality feature cache must be a JSON object")
    return {
        str(sample_id): _normalize_cached_features(features, source=f"{cache_path}:{sample_id}", feature_names=feature_names)
        for sample_id, features in payload.items()
    }


def write_quality_feature_cache(
    path: str | Path,
    features_by_sample_id: Mapping[str, Mapping[str, float]],
    *,
    feature_set: str = "base",
) -> None:
    """Write a stable JSON quality feature cache."""
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    feature_names = feature_names_for_set(feature_set)
    normalized = {
        str(sample_id): _normalize_cached_features(features, source=str(sample_id), feature_names=feature_names)
        for sample_id, features in sorted(features_by_sample_id.items())
    }
    cache_path.write_text(
        json.dumps(normalized, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def compute_quality_features(
    visible: np.ndarray,
    infrared: np.ndarray,
    depth: np.ndarray,
) -> dict[str, float]:
    """Compute RGB, infrared, and depth quality features in a stable order."""
    rgb_gray = _to_unit_gray(visible)
    ir_gray = _to_unit_gray(infrared)
    depth_mm = _to_depth_mm(depth)

    features: dict[str, float] = {}
    features.update(_rgb_features(rgb_gray))
    features.update(_ir_features(ir_gray))
    features.update(_depth_features(depth_mm))
    return {name: float(features[name]) for name in QUALITY_FEATURE_NAMES}


def _normalize_cached_features(
    features: Mapping[str, float],
    *,
    source: str,
    feature_names: tuple[str, ...] = QUALITY_FEATURE_NAMES,
) -> dict[str, float]:
    if not isinstance(features, Mapping):
        raise ValueError(f"{source}: cached quality features must be an object")
    missing = [name for name in feature_names if name not in features]
    allowed = set(feature_names)
    extra = sorted(str(name) for name in features if name not in allowed)
    if missing or extra:
        raise ValueError(f"{source}: invalid quality feature keys missing={missing[:5]} extra={extra[:5]}")
    return {name: _safe_float(features[name]) for name in feature_names}


def _rgb_features(gray: np.ndarray) -> Mapping[str, float]:
    laplace = _laplace(gray)
    return {
        "rgb_entropy": _entropy01(gray),
        "rgb_brightness_mean": _mean(gray),
        "rgb_brightness_std": _std(gray),
        "rgb_laplace_var": _safe_float(np.var(laplace)),
        "rgb_edge_density": _ratio(np.abs(laplace) > 0.08),
        "rgb_overexposed_ratio": _ratio(gray >= 0.98),
        "rgb_underexposed_ratio": _ratio(gray <= 0.02),
        "rgb_local_contrast": _mean(np.abs(gray - _local_mean(gray))),
    }


def _ir_features(gray: np.ndarray) -> Mapping[str, float]:
    laplace_abs = np.abs(_laplace(gray))
    local_min = _local_min(gray)
    local_max = _local_max(gray)
    mean = _mean(gray)
    std = _std(gray)
    return {
        "ir_entropy": _entropy01(gray),
        "ir_mean": mean,
        "ir_std": std,
        "ir_laplace_mean": _mean(laplace_abs),
        "ir_tophat_mean": _mean(np.maximum(gray - local_min, 0.0)),
        "ir_blackhat_mean": _mean(np.maximum(local_max - gray, 0.0)),
        "ir_hot_ratio": _ratio(gray > mean + std),
        "ir_cold_ratio": _ratio(gray < mean - std),
    }


def _depth_features(depth_mm: np.ndarray) -> Mapping[str, float]:
    finite = np.isfinite(depth_mm)
    valid = finite & (depth_mm >= DEPTH_VALID_MIN_MM) & (depth_mm <= DEPTH_VALID_MAX_MM)
    valid_depth = depth_mm[valid]
    valid_ratio = _ratio(valid)
    hole_ratio = 1.0 - valid_ratio

    if valid_depth.size == 0:
        return {
            "depth_valid_ratio": 0.0,
            "depth_hole_ratio": 1.0,
            "depth_log_mean": 0.0,
            "depth_log_std": 0.0,
            "depth_inv_mean": 0.0,
            "depth_edge_density": 0.0,
            "depth_near_ratio": 0.0,
            "depth_far_ratio": 0.0,
        }

    log_depth = np.log(valid_depth)
    inv_depth = 1.0 / valid_depth
    filled_log = np.full(depth_mm.shape, float(np.median(log_depth)), dtype=np.float32)
    filled_log[valid] = np.log(depth_mm[valid])
    edge_map = np.abs(_laplace(_normalize_valid(filled_log, valid)))

    return {
        "depth_valid_ratio": valid_ratio,
        "depth_hole_ratio": hole_ratio,
        "depth_log_mean": _mean(log_depth),
        "depth_log_std": _std(log_depth),
        "depth_inv_mean": _mean(inv_depth),
        "depth_edge_density": _ratio(edge_map[valid] > 0.05),
        "depth_near_ratio": _ratio(valid_depth <= 2000.0),
        "depth_far_ratio": _ratio(valid_depth >= 10000.0),
    }


def _to_unit_gray(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3:
        arr = arr[..., :3].astype(np.float32).mean(axis=2)
    else:
        arr = arr.astype(np.float32)

    if arr.size == 0:
        return np.zeros((1, 1), dtype=np.float32)

    if np.issubdtype(np.asarray(image).dtype, np.integer):
        max_value = float(np.iinfo(np.asarray(image).dtype).max)
        if max_value > 0:
            arr = arr / max_value
    else:
        max_value = float(np.nanmax(arr)) if np.isfinite(arr).any() else 1.0
        if max_value > 1.0:
            arr = arr / max_value

    return np.clip(np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32)


def _to_depth_mm(depth: np.ndarray) -> np.ndarray:
    arr = np.asarray(depth)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.size == 0:
        return np.zeros((1, 1), dtype=np.float32)
    return np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _entropy01(values: np.ndarray, bins: int = 32) -> float:
    hist, _ = np.histogram(values, bins=bins, range=(0.0, 1.0))
    total = int(hist.sum())
    if total == 0:
        return 0.0
    probs = hist.astype(np.float64) / total
    probs = probs[probs > 0]
    entropy = -float(np.sum(probs * np.log2(probs)))
    return entropy / float(np.log2(bins))


def _laplace(values: np.ndarray) -> np.ndarray:
    padded = np.pad(values.astype(np.float32), 1, mode="edge")
    center = padded[1:-1, 1:-1]
    return (
        4.0 * center
        - padded[:-2, 1:-1]
        - padded[2:, 1:-1]
        - padded[1:-1, :-2]
        - padded[1:-1, 2:]
    )


def _local_mean(values: np.ndarray) -> np.ndarray:
    windows = _neighbor_stack(values)
    return np.mean(windows, axis=0)


def _local_min(values: np.ndarray) -> np.ndarray:
    return np.min(_neighbor_stack(values), axis=0)


def _local_max(values: np.ndarray) -> np.ndarray:
    return np.max(_neighbor_stack(values), axis=0)


def _neighbor_stack(values: np.ndarray) -> np.ndarray:
    padded = np.pad(values.astype(np.float32), 1, mode="edge")
    return np.stack(
        [
            padded[dy : dy + values.shape[0], dx : dx + values.shape[1]]
            for dy in range(3)
            for dx in range(3)
        ],
        axis=0,
    )


def _normalize_valid(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    valid_values = values[valid]
    if valid_values.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    min_value = float(valid_values.min())
    max_value = float(valid_values.max())
    if max_value <= min_value:
        return np.zeros_like(values, dtype=np.float32)
    normalized = (values - min_value) / (max_value - min_value)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def _resize_max_side(image: Image.Image, *, max_side: int, resample: int) -> Image.Image:
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image
    scale = float(max_side) / float(longest)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    return image.resize((resized_width, resized_height), resample=resample)


def _open_quality_image(path: str | Path, *, max_side: int | None, resample: int) -> Image.Image:
    if max_side is not None and max_side <= 0:
        raise ValueError("max_side must be positive")
    with Image.open(path) as image:
        image = image.copy()
    if max_side is None:
        return image
    return _resize_max_side(image, max_side=max_side, resample=resample)


def _mean(values: np.ndarray) -> float:
    return _safe_float(np.mean(values)) if values.size else 0.0


def _std(values: np.ndarray) -> float:
    return _safe_float(np.std(values)) if values.size else 0.0


def _ratio(mask: np.ndarray) -> float:
    arr = np.asarray(mask)
    return _safe_float(np.mean(arr.astype(np.float32))) if arr.size else 0.0


def _safe_float(value: float | np.floating) -> float:
    result = float(value)
    if not np.isfinite(result):
        return 0.0
    return result
