"""PyTorch datasets for project-owned RGC-DINO training loops."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from .constants import DEPTH_VALID_MAX_MM, DEPTH_VALID_MIN_MM, RGB_MEAN, RGB_STD
from .dataset import MultimodalSample, discover_aligned_samples
from .labels import DetectionLabel, load_label_file, load_label_file_clipped
from .quality_features import QUALITY_FEATURE_NAMES, load_quality_feature_cache, load_quality_features


class MultimodalDinoDataset(Dataset):
    """Load aligned RGB/IR/depth samples with DINO-compatible targets."""

    def __init__(
        self,
        samples: Sequence[MultimodalSample],
        *,
        image_max_side: int | None = 640,
        image_max_sides: Sequence[int] | None = None,
        clip_labels: bool = True,
        random_horizontal_flip_prob: float = 0.0,
        quality_cache: dict[str, dict[str, float]] | None = None,
    ) -> None:
        if not samples:
            raise ValueError("samples must not be empty")
        if image_max_side is not None and image_max_side <= 0:
            raise ValueError("image_max_side must be positive")
        if image_max_sides is not None:
            if not image_max_sides:
                raise ValueError("image_max_sides must not be empty")
            if any(side <= 0 for side in image_max_sides):
                raise ValueError("image_max_sides must contain only positive integers")
        if not 0.0 <= random_horizontal_flip_prob <= 1.0:
            raise ValueError("random_horizontal_flip_prob must be in [0, 1]")
        self.samples = tuple(samples)
        self.image_max_side = image_max_side
        self.image_max_sides = tuple(image_max_sides) if image_max_sides is not None else None
        self.clip_labels = clip_labels
        self.random_horizontal_flip_prob = random_horizontal_flip_prob
        self.quality_cache = quality_cache

    @classmethod
    def from_paths(
        cls,
        *,
        dataset_root: str | Path,
        labels_dir: str | Path,
        sample_ids: Sequence[str] | None = None,
        image_max_side: int | None = 640,
        image_max_sides: Sequence[int] | None = None,
        clip_labels: bool = True,
        random_horizontal_flip_prob: float = 0.0,
        quality_cache: dict[str, dict[str, float]] | None = None,
        quality_cache_path: str | Path | None = None,
    ) -> "MultimodalDinoDataset":
        samples = discover_aligned_samples(dataset_root, labels_dir=labels_dir, require_labels=True)
        if sample_ids is not None:
            by_id = {sample.sample_id: sample for sample in samples}
            missing = [sample_id for sample_id in sample_ids if sample_id not in by_id]
            if missing:
                raise ValueError(f"sample ids not found in aligned labeled dataset: {missing[:5]}")
            samples = [by_id[sample_id] for sample_id in sample_ids]
        loaded_quality_cache = quality_cache
        if quality_cache_path is not None:
            loaded_quality_cache = load_quality_feature_cache(quality_cache_path)
        return cls(
            samples,
            image_max_side=image_max_side,
            image_max_sides=image_max_sides,
            clip_labels=clip_labels,
            random_horizontal_flip_prob=random_horizontal_flip_prob,
            quality_cache=loaded_quality_cache,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        sample = self.samples[index]
        horizontal_flip = self._should_flip_horizontally()
        loaded = _load_modalities(
            sample,
            image_max_side=self._select_image_max_side(),
            horizontal_flip=horizontal_flip,
            quality_cache=self.quality_cache,
        )
        records = _load_records(sample, clip_labels=self.clip_labels)
        if horizontal_flip:
            records = _flip_records_horizontally(records)

        return (
            loaded.tensors,
            _dino_target(
                records,
                image_id=index,
                original_height=loaded.original_height,
                original_width=loaded.original_width,
                resized_height=loaded.resized_height,
                resized_width=loaded.resized_width,
            ),
        )

    def _should_flip_horizontally(self) -> bool:
        return self.random_horizontal_flip_prob > 0.0 and random.random() < self.random_horizontal_flip_prob

    def _select_image_max_side(self) -> int | None:
        if self.image_max_sides is None:
            return self.image_max_side
        return random.choice(self.image_max_sides)


class MultimodalDinoInferenceDataset(Dataset):
    """Load aligned RGB/IR/depth samples without labels for prediction."""

    def __init__(
        self,
        samples: Sequence[MultimodalSample],
        *,
        image_max_side: int | None = 640,
        quality_cache: dict[str, dict[str, float]] | None = None,
    ) -> None:
        if not samples:
            raise ValueError("samples must not be empty")
        if image_max_side is not None and image_max_side <= 0:
            raise ValueError("image_max_side must be positive")
        self.samples = tuple(samples)
        self.image_max_side = image_max_side
        self.quality_cache = quality_cache

    @classmethod
    def from_paths(
        cls,
        *,
        dataset_root: str | Path,
        sample_ids: Sequence[str] | None = None,
        image_max_side: int | None = 640,
        quality_cache: dict[str, dict[str, float]] | None = None,
        quality_cache_path: str | Path | None = None,
    ) -> "MultimodalDinoInferenceDataset":
        samples = discover_aligned_samples(dataset_root)
        if sample_ids is not None:
            by_id = {sample.sample_id: sample for sample in samples}
            missing = [sample_id for sample_id in sample_ids if sample_id not in by_id]
            if missing:
                raise ValueError(f"sample ids not found in aligned dataset: {missing[:5]}")
            samples = [by_id[sample_id] for sample_id in sample_ids]
        loaded_quality_cache = quality_cache
        if quality_cache_path is not None:
            loaded_quality_cache = load_quality_feature_cache(quality_cache_path)
        return cls(samples, image_max_side=image_max_side, quality_cache=loaded_quality_cache)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[dict[str, Tensor], dict[str, Tensor | str]]:
        sample = self.samples[index]
        loaded = _load_modalities(sample, image_max_side=self.image_max_side, quality_cache=self.quality_cache)
        return (
            loaded.tensors,
            {
                "image_id": torch.tensor(index, dtype=torch.int64),
                "orig_size": torch.tensor([loaded.original_height, loaded.original_width], dtype=torch.int64),
                "size": torch.tensor([loaded.resized_height, loaded.resized_width], dtype=torch.int64),
                "sample_id": sample.sample_id,
            },
        )


@dataclass(frozen=True)
class _LoadedModalities:
    tensors: dict[str, Tensor]
    original_height: int
    original_width: int
    resized_height: int
    resized_width: int


def _load_modalities(
    sample: MultimodalSample,
    *,
    image_max_side: int | None,
    horizontal_flip: bool = False,
    quality_cache: dict[str, dict[str, float]] | None = None,
) -> _LoadedModalities:
    rgb_image = _load_pil(sample.visible_path, mode="RGB")
    infrared_image = _load_pil(sample.infrared_path, mode="L")
    depth_image = _load_pil(sample.depth_path)

    original_width, original_height = rgb_image.size
    resized_width, resized_height = _resized_size(
        width=original_width,
        height=original_height,
        max_side=image_max_side,
    )
    if infrared_image.size != (original_width, original_height):
        raise ValueError(f"infrared size for {sample.sample_id} does not match visible image")
    if depth_image.size != (original_width, original_height):
        raise ValueError(f"depth size for {sample.sample_id} does not match visible image")
    if horizontal_flip:
        rgb_image = _flip_image_horizontally(rgb_image)
        infrared_image = _flip_image_horizontally(infrared_image)
        depth_image = _flip_image_horizontally(depth_image)

    return _LoadedModalities(
        tensors={
            "rgb": _rgb_tensor(_resize(rgb_image, resized_width, resized_height, Image.Resampling.BILINEAR)),
            "infrared": _unit_gray_tensor(
                _resize(infrared_image, resized_width, resized_height, Image.Resampling.BILINEAR)
            ),
            "depth": _depth_tensor(_resize(depth_image, resized_width, resized_height, Image.Resampling.NEAREST)),
            "quality": _quality_tensor(sample, quality_cache=quality_cache),
        },
        original_height=original_height,
        original_width=original_width,
        resized_height=resized_height,
        resized_width=resized_width,
    )


def _load_pil(path: Path, mode: str | None = None) -> Image.Image:
    with Image.open(path) as image:
        if mode is not None:
            image = image.convert(mode)
        else:
            image = image.copy()
    return image


def _resize(image: Image.Image, width: int, height: int, resample: int) -> Image.Image:
    if image.size == (width, height):
        return image
    return image.resize((width, height), resample=resample)


def _flip_image_horizontally(image: Image.Image) -> Image.Image:
    return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)


def _resized_size(*, width: int, height: int, max_side: int | None) -> tuple[int, int]:
    if max_side is None:
        return width, height
    scale = float(max_side) / float(max(width, height))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    return resized_width, resized_height


def _rgb_tensor(image: Image.Image) -> Tensor:
    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    mean = torch.tensor(RGB_MEAN, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(RGB_STD, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


def _unit_gray_tensor(image: Image.Image) -> Tensor:
    arr = np.asarray(image, dtype=np.float32)
    if arr.size == 0:
        arr = np.zeros((1, 1), dtype=np.float32)
    max_value = 255.0
    if np.issubdtype(np.asarray(image).dtype, np.integer):
        max_value = float(np.iinfo(np.asarray(image).dtype).max)
    if max_value > 0:
        arr = arr / max_value
    arr = np.clip(np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    return torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).contiguous()


def _depth_tensor(image: Image.Image) -> Tensor:
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    valid = (arr >= DEPTH_VALID_MIN_MM) & (arr <= DEPTH_VALID_MAX_MM)
    clipped = np.clip(arr, DEPTH_VALID_MIN_MM, DEPTH_VALID_MAX_MM)
    log_depth = np.log(clipped)
    log_min = float(np.log(DEPTH_VALID_MIN_MM))
    log_max = float(np.log(DEPTH_VALID_MAX_MM))
    log_depth = (log_depth - log_min) / (log_max - log_min)

    inv_depth = 1.0 / clipped
    inv_min = 1.0 / float(DEPTH_VALID_MAX_MM)
    inv_max = 1.0 / float(DEPTH_VALID_MIN_MM)
    inv_depth = (inv_depth - inv_min) / (inv_max - inv_min)

    log_depth[~valid] = 0.0
    inv_depth[~valid] = 0.0
    stacked = np.stack([log_depth, inv_depth, valid.astype(np.float32)], axis=0)
    return torch.from_numpy(stacked.astype(np.float32)).contiguous()


def _quality_tensor(sample: MultimodalSample, *, quality_cache: dict[str, dict[str, float]] | None = None) -> Tensor:
    if quality_cache is not None:
        if sample.sample_id not in quality_cache:
            raise KeyError(f"quality cache does not contain sample id {sample.sample_id}")
        features = quality_cache[sample.sample_id]
    else:
        features = load_quality_features(sample.visible_path, sample.infrared_path, sample.depth_path)
    return torch.tensor([features[name] for name in QUALITY_FEATURE_NAMES], dtype=torch.float32)


def _load_records(sample: MultimodalSample, *, clip_labels: bool) -> list[DetectionLabel]:
    if sample.label_path is None:
        return []
    if clip_labels:
        return load_label_file_clipped(sample.label_path)[0]
    return load_label_file(sample.label_path)


def _flip_records_horizontally(records: Sequence[DetectionLabel]) -> list[DetectionLabel]:
    return [
        DetectionLabel(
            class_id=record.class_id,
            norm_center_x=1.0 - record.norm_center_x,
            norm_center_y=record.norm_center_y,
            norm_w=record.norm_w,
            norm_h=record.norm_h,
            confidence=record.confidence,
        )
        for record in records
    ]


def _dino_target(
    records: Sequence[DetectionLabel],
    *,
    image_id: int,
    original_height: int,
    original_width: int,
    resized_height: int,
    resized_width: int,
) -> dict[str, Tensor]:
    labels = torch.tensor([record.class_id for record in records], dtype=torch.int64)
    boxes = torch.tensor(
        [
            [record.norm_center_x, record.norm_center_y, record.norm_w, record.norm_h]
            for record in records
        ],
        dtype=torch.float32,
    ).reshape(-1, 4)
    areas = torch.tensor(
        [
            record.norm_w * float(resized_width) * record.norm_h * float(resized_height)
            for record in records
        ],
        dtype=torch.float32,
    )
    return {
        "boxes": boxes,
        "labels": labels,
        "image_id": torch.tensor(image_id, dtype=torch.int64),
        "area": areas,
        "iscrowd": torch.zeros((len(records),), dtype=torch.int64),
        "orig_size": torch.tensor([original_height, original_width], dtype=torch.int64),
        "size": torch.tensor([resized_height, resized_width], dtype=torch.int64),
    }
