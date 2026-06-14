import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from rgc_dino.quality_features import (
    QUALITY_FEATURE_NAMES,
    compute_quality_features,
    load_quality_feature_cache,
    load_quality_features,
    write_quality_feature_cache,
)


class QualityFeaturesTest(unittest.TestCase):
    def test_computes_24_finite_features_from_arrays(self) -> None:
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        rgb[:2, :2] = 255
        infrared = np.full((4, 4, 3), 64, dtype=np.uint8)
        depth = np.array(
            [
                [0, 500, 20000, 25000],
                [300, 299, 1000, 15000],
                [0, 0, 0, 0],
                [4000, 5000, 6000, 7000],
            ],
            dtype=np.uint16,
        )

        features = compute_quality_features(rgb, infrared, depth)

        self.assertEqual(tuple(features), QUALITY_FEATURE_NAMES)
        self.assertEqual(len(features), 24)
        self.assertTrue(all(math.isfinite(value) for value in features.values()))
        self.assertAlmostEqual(features["rgb_overexposed_ratio"], 0.25)
        self.assertAlmostEqual(features["rgb_underexposed_ratio"], 0.75)
        self.assertAlmostEqual(features["depth_valid_ratio"], 9 / 16)
        self.assertAlmostEqual(features["depth_hole_ratio"], 7 / 16)
        self.assertAlmostEqual(features["depth_near_ratio"], 3 / 9)
        self.assertAlmostEqual(features["depth_far_ratio"], 2 / 9)

    def test_collapses_three_channel_ir_like_single_channel(self) -> None:
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        ir_single = np.array([[0, 128], [255, 128]], dtype=np.uint8)
        ir_three = np.repeat(ir_single[:, :, None], 3, axis=2)
        depth = np.full((2, 2), 1000, dtype=np.uint16)

        single_features = compute_quality_features(rgb, ir_single, depth)
        three_features = compute_quality_features(rgb, ir_three, depth)

        self.assertAlmostEqual(single_features["ir_mean"], three_features["ir_mean"])
        self.assertAlmostEqual(single_features["ir_entropy"], three_features["ir_entropy"])

    def test_loads_features_from_image_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rgb_path = root / "rgb.png"
            ir_path = root / "ir.png"
            depth_path = root / "depth.png"

            Image.new("RGB", (3, 3), color=(10, 20, 30)).save(rgb_path)
            Image.new("RGB", (3, 3), color=(80, 80, 80)).save(ir_path)
            Image.fromarray(np.full((3, 3), 1000, dtype=np.uint16)).save(depth_path)

            features = load_quality_features(rgb_path, ir_path, depth_path)

            self.assertEqual(tuple(features), QUALITY_FEATURE_NAMES)
            self.assertAlmostEqual(features["depth_valid_ratio"], 1.0)

    def test_quality_feature_cache_round_trips_with_stable_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "quality.json"
            features = {
                "sample_a": {
                    name: float(index) / 100.0
                    for index, name in enumerate(QUALITY_FEATURE_NAMES)
                }
            }

            write_quality_feature_cache(cache_path, features)
            loaded = load_quality_feature_cache(cache_path)

            self.assertEqual(tuple(loaded["sample_a"]), QUALITY_FEATURE_NAMES)
            self.assertEqual(loaded, features)


if __name__ == "__main__":
    unittest.main()
