"""RGC-DINO-R50 competition utilities."""

from .constants import CLASS_NAMES, NUM_CLASSES
from .dataset import MultimodalSample, discover_aligned_samples
from .labels import DetectionLabel

__all__ = [
    "CLASS_NAMES",
    "NUM_CLASSES",
    "DetectionLabel",
    "MultimodalSample",
    "discover_aligned_samples",
]
__version__ = "0.1.0"
