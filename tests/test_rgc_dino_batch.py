import unittest

import torch

from rgc_dino.dino_batch import NestedTensorBatch, collate_rgc_dino_batch
from rgc_dino.models.rgc_dino_adapter import RgcDinoSamples


class RgcDinoBatchTest(unittest.TestCase):
    def test_collates_multimodal_samples_with_padding_masks(self) -> None:
        sample_a = {
            "rgb": torch.ones(3, 10, 8),
            "infrared": torch.ones(1, 10, 8) * 2,
            "depth": torch.ones(1, 10, 8) * 3,
            "quality": torch.ones(24),
        }
        sample_b = {
            "rgb": torch.ones(3, 6, 12) * 4,
            "infrared": torch.ones(1, 6, 12) * 5,
            "depth": torch.ones(1, 6, 12) * 6,
            "quality": torch.ones(24) * 7,
        }
        target_a = {"image_id": torch.tensor(11)}
        target_b = {"image_id": torch.tensor(12)}

        samples, targets = collate_rgc_dino_batch([(sample_a, target_a), (sample_b, target_b)])

        self.assertIsInstance(samples, RgcDinoSamples)
        self.assertIsInstance(samples.rgb, NestedTensorBatch)
        self.assertEqual(samples.rgb.tensors.shape, (2, 3, 10, 12))
        self.assertEqual(samples.rgb.mask.shape, (2, 10, 12))
        self.assertFalse(bool(samples.rgb.mask[0, :10, :8].any()))
        self.assertTrue(bool(samples.rgb.mask[0, :, 8:].all()))
        self.assertFalse(bool(samples.rgb.mask[1, :6, :12].any()))
        self.assertTrue(bool(samples.rgb.mask[1, 6:, :].all()))
        self.assertEqual(samples.infrared.shape, (2, 1, 10, 12))
        self.assertEqual(samples.depth.shape, (2, 1, 10, 12))
        self.assertEqual(samples.quality.shape, (2, 24))
        self.assertEqual([int(target["image_id"]) for target in targets], [11, 12])

    def test_rejects_non_chw_images(self) -> None:
        with self.assertRaises(ValueError):
            collate_rgc_dino_batch(
                [
                    (
                        {
                            "rgb": torch.ones(10, 8),
                            "infrared": torch.ones(1, 10, 8),
                            "depth": torch.ones(1, 10, 8),
                            "quality": torch.ones(24),
                        },
                        {},
                    )
                ]
            )


if __name__ == "__main__":
    unittest.main()
