"""Shared constants for the RGC-DINO-R50 competition project."""

from __future__ import annotations

CLASS_NAMES: tuple[str, ...] = (
    "person",
    "boat",
    "animal",
    "seat",
    "sign",
    "bicycle",
    "car",
    "ball",
    "light",
    "garbage can",
    "uav",
    "tricycle",
)

NUM_CLASSES = len(CLASS_NAMES)
MAX_PREDICTIONS_PER_IMAGE = 100

TRAIN_LABEL_FIELDS: tuple[str, ...] = (
    "class_id",
    "norm_center_x",
    "norm_center_y",
    "norm_w",
    "norm_h",
)

SUBMISSION_FIELDS: tuple[str, ...] = TRAIN_LABEL_FIELDS + ("confidence",)

RGB_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
RGB_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

DEPTH_VALID_MIN_MM = 300
DEPTH_VALID_MAX_MM = 20000

