import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from rgc_dino.quality_features import (
    QUALITY_FEATURE_NAMES,
    RDT_QUALITY_FEATURE_NAMES,
    compute_quality_features,
    feature_names_for_set,
    load_quality_feature_cache,
    load_quality_features,
    write_quality_feature_cache,
)
from scripts.cache_quality_features import build_quality_cache
from rgc_dino.dataset import MultimodalSample


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

    def test_loads_features_from_downscaled_images_when_max_side_is_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rgb_path = root / "rgb.png"
            ir_path = root / "ir.png"
            depth_path = root / "depth.png"
            rgb = np.zeros((8, 12, 3), dtype=np.uint8)
            rgb[:, 6:] = 255
            Image.fromarray(rgb).save(rgb_path)
            Image.fromarray(rgb).save(ir_path)
            Image.fromarray(np.full((8, 12), 1000, dtype=np.uint16)).save(depth_path)

            full_features = load_quality_features(rgb_path, ir_path, depth_path)
            downscaled_features = load_quality_features(rgb_path, ir_path, depth_path, max_side=4)

            self.assertEqual(tuple(downscaled_features), QUALITY_FEATURE_NAMES)
            self.assertEqual(downscaled_features["depth_valid_ratio"], 1.0)
            self.assertNotEqual(full_features["rgb_laplace_var"], downscaled_features["rgb_laplace_var"])

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

    def test_base_rdt_quality_feature_cache_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "quality_base_rdt.json"
            feature_names = feature_names_for_set("base_rdt")
            features = {
                "sample_a": {
                    name: float(index) / 100.0
                    for index, name in enumerate(feature_names)
                }
            }

            write_quality_feature_cache(cache_path, features, feature_set="base_rdt")
            loaded = load_quality_feature_cache(cache_path, feature_set="base_rdt")

            self.assertEqual(feature_names, QUALITY_FEATURE_NAMES + RDT_QUALITY_FEATURE_NAMES)
            self.assertEqual(len(feature_names), 35)
            self.assertEqual(tuple(loaded["sample_a"]), feature_names)
            self.assertEqual(loaded, features)

    def test_build_quality_cache_parallel_matches_single_worker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples = []
            for index in range(3):
                rgb_path = root / f"rgb_{index}.png"
                ir_path = root / f"ir_{index}.png"
                depth_path = root / f"depth_{index}.png"
                rgb = np.full((8, 12, 3), index * 40, dtype=np.uint8)
                Image.fromarray(rgb).save(rgb_path)
                Image.fromarray(255 - rgb).save(ir_path)
                Image.fromarray(np.full((8, 12), 1000 + index, dtype=np.uint16)).save(depth_path)
                samples.append(
                    MultimodalSample(
                        sample_id=f"sample_{index}",
                        visible_path=rgb_path,
                        infrared_path=ir_path,
                        depth_path=depth_path,
                        label_path=None,
                    )
                )

            single = build_quality_cache(samples, max_side=4, num_workers=1)
            parallel = build_quality_cache(samples, max_side=4, num_workers=2)

            self.assertEqual(parallel, single)


if __name__ == "__main__":
    unittest.main()
