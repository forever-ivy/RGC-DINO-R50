import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from rgc_dino.dino_dataset import MultimodalDinoDataset, MultimodalDinoInferenceDataset
from rgc_dino.quality_features import QUALITY_FEATURE_NAMES


class MultimodalDinoDatasetTest(unittest.TestCase):
    def test_loads_multimodal_sample_with_dino_target_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "train"
            labels = root / "labels"
            for modality in ("visible", "infrared", "depth"):
                (root / modality).mkdir(parents=True)
            labels.mkdir()

            rgb = np.zeros((10, 20, 3), dtype=np.uint8)
            rgb[..., 0] = 255
            infrared = np.full((10, 20, 3), 127, dtype=np.uint8)
            depth = np.full((10, 20), 1200, dtype=np.uint16)
            Image.fromarray(rgb).save(root / "visible" / "sample_a.png")
            Image.fromarray(infrared).save(root / "infrared" / "sample_a.png")
            Image.fromarray(depth).save(root / "depth" / "sample_a.png")
            (labels / "sample_a.txt").write_text("3 0.5 0.5 0.25 0.5\n", encoding="utf-8")

            dataset = MultimodalDinoDataset.from_paths(
                dataset_root=root,
                labels_dir=labels,
                sample_ids=["sample_a"],
                image_max_side=10,
            )

            sample, target = dataset[0]

            self.assertEqual(sample["rgb"].shape, (3, 5, 10))
            self.assertEqual(sample["infrared"].shape, (1, 5, 10))
            self.assertEqual(sample["depth"].shape, (3, 5, 10))
            self.assertTrue(torch.all(sample["depth"][2] == 1.0))
            self.assertEqual(sample["quality"].shape, (24,))
            self.assertEqual(target["labels"].dtype, torch.int64)
            self.assertTrue(torch.equal(target["labels"], torch.tensor([3])))
            self.assertTrue(torch.allclose(target["boxes"], torch.tensor([[0.5, 0.5, 0.25, 0.5]])))
            self.assertTrue(torch.equal(target["orig_size"], torch.tensor([10, 20])))
            self.assertTrue(torch.equal(target["size"], torch.tensor([5, 10])))
            self.assertEqual(int(target["image_id"]), 0)

    def test_accepts_rgb_encoded_depth_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "train"
            labels = root / "labels"
            for modality in ("visible", "infrared", "depth"):
                (root / modality).mkdir(parents=True)
            labels.mkdir()

            rgb = np.zeros((8, 8, 3), dtype=np.uint8)
            infrared = np.zeros((8, 8, 3), dtype=np.uint8)
            depth_rgb = np.full((8, 8, 3), 64, dtype=np.uint8)
            Image.fromarray(rgb).save(root / "visible" / "sample_b.jpg")
            Image.fromarray(infrared).save(root / "infrared" / "sample_b.jpg")
            Image.fromarray(depth_rgb).save(root / "depth" / "sample_b.jpg")
            (labels / "sample_b.txt").write_text("1 0.5 0.5 0.5 0.5\n", encoding="utf-8")

            dataset = MultimodalDinoDataset.from_paths(
                dataset_root=root,
                labels_dir=labels,
                sample_ids=["sample_b"],
                image_max_side=8,
            )

            sample, _target = dataset[0]

            self.assertEqual(sample["depth"].shape, (3, 8, 8))

    def test_depth_tensor_includes_spatial_valid_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "train"
            labels = root / "labels"
            for modality in ("visible", "infrared", "depth"):
                (root / modality).mkdir(parents=True)
            labels.mkdir()

            rgb = np.zeros((2, 2, 3), dtype=np.uint8)
            infrared = np.zeros((2, 2, 3), dtype=np.uint8)
            depth = np.array(
                [
                    [0, 300],
                    [20000, 25000],
                ],
                dtype=np.uint16,
            )
            Image.fromarray(rgb).save(root / "visible" / "sample_c.png")
            Image.fromarray(infrared).save(root / "infrared" / "sample_c.png")
            Image.fromarray(depth).save(root / "depth" / "sample_c.png")
            (labels / "sample_c.txt").write_text("1 0.5 0.5 0.5 0.5\n", encoding="utf-8")

            dataset = MultimodalDinoDataset.from_paths(
                dataset_root=root,
                labels_dir=labels,
                sample_ids=["sample_c"],
                image_max_side=2,
            )

            sample, _target = dataset[0]

            self.assertEqual(sample["depth"].shape, (3, 2, 2))
            log_depth, inverse_depth, valid_mask = sample["depth"]
            self.assertTrue(
                torch.equal(
                    valid_mask,
                    torch.tensor(
                        [
                            [0.0, 1.0],
                            [1.0, 0.0],
                        ]
                    ),
                )
            )
            self.assertEqual(float(log_depth[0, 0]), 0.0)
            self.assertEqual(float(inverse_depth[0, 0]), 0.0)
            self.assertGreater(float(inverse_depth[0, 1]), float(inverse_depth[1, 0]))

    def test_horizontal_flip_is_applied_to_all_modalities_and_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "train"
            labels = root / "labels"
            for modality in ("visible", "infrared", "depth"):
                (root / modality).mkdir(parents=True)
            labels.mkdir()

            rgb = np.zeros((1, 2, 3), dtype=np.uint8)
            rgb[0, 0, 0] = 0
            rgb[0, 1, 0] = 255
            infrared = np.array([[[10, 10, 10], [200, 200, 200]]], dtype=np.uint8)
            depth = np.array([[300, 1000]], dtype=np.uint16)
            Image.fromarray(rgb).save(root / "visible" / "sample_flip.png")
            Image.fromarray(infrared).save(root / "infrared" / "sample_flip.png")
            Image.fromarray(depth).save(root / "depth" / "sample_flip.png")
            (labels / "sample_flip.txt").write_text("2 0.25 0.5 0.2 1.0\n", encoding="utf-8")

            dataset = MultimodalDinoDataset.from_paths(
                dataset_root=root,
                labels_dir=labels,
                sample_ids=["sample_flip"],
                image_max_side=None,
                random_horizontal_flip_prob=1.0,
            )

            sample, target = dataset[0]

            self.assertGreater(float(sample["rgb"][0, 0, 0]), float(sample["rgb"][0, 0, 1]))
            self.assertGreater(float(sample["infrared"][0, 0, 0]), float(sample["infrared"][0, 0, 1]))
            self.assertGreater(float(sample["depth"][0, 0, 0]), float(sample["depth"][0, 0, 1]))
            self.assertTrue(torch.allclose(target["boxes"], torch.tensor([[0.75, 0.5, 0.2, 1.0]])))

    def test_random_image_max_side_is_selected_per_training_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "train"
            labels = root / "labels"
            for modality in ("visible", "infrared", "depth"):
                (root / modality).mkdir(parents=True)
            labels.mkdir()

            rgb = np.zeros((10, 20, 3), dtype=np.uint8)
            infrared = np.zeros((10, 20, 3), dtype=np.uint8)
            depth = np.full((10, 20), 1000, dtype=np.uint16)
            Image.fromarray(rgb).save(root / "visible" / "sample_scale.png")
            Image.fromarray(infrared).save(root / "infrared" / "sample_scale.png")
            Image.fromarray(depth).save(root / "depth" / "sample_scale.png")
            (labels / "sample_scale.txt").write_text("2 0.5 0.5 0.2 0.2\n", encoding="utf-8")

            dataset = MultimodalDinoDataset.from_paths(
                dataset_root=root,
                labels_dir=labels,
                sample_ids=["sample_scale"],
                image_max_sides=(10, 16),
            )

            with mock.patch("rgc_dino.dino_dataset.random.choice", return_value=16):
                sample, target = dataset[0]

            self.assertEqual(sample["rgb"].shape, (3, 8, 16))
            self.assertTrue(torch.equal(target["size"], torch.tensor([8, 16])))

    def test_uses_quality_cache_without_recomputing_features(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "train"
            labels = root / "labels"
            for modality in ("visible", "infrared", "depth"):
                (root / modality).mkdir(parents=True)
            labels.mkdir()

            rgb = np.zeros((4, 4, 3), dtype=np.uint8)
            infrared = np.zeros((4, 4, 3), dtype=np.uint8)
            depth = np.full((4, 4), 1000, dtype=np.uint16)
            Image.fromarray(rgb).save(root / "visible" / "sample_cache.png")
            Image.fromarray(infrared).save(root / "infrared" / "sample_cache.png")
            Image.fromarray(depth).save(root / "depth" / "sample_cache.png")
            (labels / "sample_cache.txt").write_text("1 0.5 0.5 0.5 0.5\n", encoding="utf-8")
            cached = {
                "sample_cache": {
                    name: float(index)
                    for index, name in enumerate(QUALITY_FEATURE_NAMES)
                }
            }

            dataset = MultimodalDinoDataset.from_paths(
                dataset_root=root,
                labels_dir=labels,
                sample_ids=["sample_cache"],
                image_max_side=4,
                quality_cache=cached,
            )

            with mock.patch("rgc_dino.dino_dataset.load_quality_features", side_effect=AssertionError):
                sample, _target = dataset[0]

            self.assertTrue(torch.equal(sample["quality"], torch.arange(24, dtype=torch.float32)))

    def test_inference_dataset_does_not_require_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "test"
            for modality in ("visible", "infrared", "depth"):
                (root / modality).mkdir(parents=True)

            rgb = np.zeros((6, 12, 3), dtype=np.uint8)
            infrared = np.zeros((6, 12, 3), dtype=np.uint8)
            depth = np.full((6, 12), 900, dtype=np.uint16)
            Image.fromarray(rgb).save(root / "visible" / "test_a.png")
            Image.fromarray(infrared).save(root / "infrared" / "test_a.png")
            Image.fromarray(depth).save(root / "depth" / "test_a.png")

            dataset = MultimodalDinoInferenceDataset.from_paths(
                dataset_root=root,
                sample_ids=["test_a"],
                image_max_side=6,
            )

            sample, target = dataset[0]

            self.assertEqual(sample["rgb"].shape, (3, 3, 6))
            self.assertEqual(target["sample_id"], "test_a")
            self.assertTrue(torch.equal(target["orig_size"], torch.tensor([6, 12])))


if __name__ == "__main__":
    unittest.main()
