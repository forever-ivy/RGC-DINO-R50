import unittest

import torch

from rgc_dino.models.rgc_dino_adapter import ProjectedRgcFeatureFusion


class ProjectedRgcFeatureFusionTest(unittest.TestCase):
    def test_default_initialization_preserves_projected_dino_features(self) -> None:
        torch.manual_seed(19)
        adapter = ProjectedRgcFeatureFusion(
            channels=8,
            quality_dim=24,
            num_levels=2,
            side_base_channels=4,
        )
        rgb_features = [
            torch.randn(2, 8, 8, 10),
            torch.randn(2, 8, 4, 5),
        ]
        infrared = torch.randn(2, 1, 32, 40)
        depth = torch.randn(2, 3, 32, 40)
        quality = torch.randn(2, 24)

        fused = adapter(rgb_features, infrared, depth, quality)

        for fused_level, rgb_level in zip(fused, rgb_features):
            self.assertLessEqual(float((fused_level - rgb_level).detach().abs().max()), 1e-6)

    def test_fuses_projected_dino_features_without_shape_changes(self) -> None:
        torch.manual_seed(7)
        adapter = ProjectedRgcFeatureFusion(
            channels=8,
            quality_dim=24,
            num_levels=4,
            side_base_channels=4,
            gate_min=0.05,
            gate_max=0.50,
        )
        self.assertEqual(adapter.depth_encoder.in_channels, 3)
        rgb_features = [
            torch.randn(2, 8, 32, 40),
            torch.randn(2, 8, 16, 20),
            torch.randn(2, 8, 8, 10),
            torch.randn(2, 8, 4, 5),
        ]
        infrared = torch.randn(2, 3, 128, 160)
        depth = torch.randn(2, 3, 128, 160)
        quality = torch.randn(2, 24)

        fused, gates = adapter(rgb_features, infrared, depth, quality, return_gates=True)

        self.assertEqual([feature.shape for feature in fused], [feature.shape for feature in rgb_features])
        self.assertEqual(len(gates["ir"]), 4)
        self.assertEqual(len(gates["depth"]), 4)
        for modality in ("ir", "depth"):
            for gate in gates[modality]:
                self.assertEqual(gate.shape, (2, 1, 1, 1))
                self.assertGreaterEqual(float(gate.detach().min()), 0.05)
                self.assertLessEqual(float(gate.detach().max()), 0.50)

    def test_accepts_extended_quality_width_for_rdt_feature_set(self) -> None:
        adapter = ProjectedRgcFeatureFusion(channels=8, quality_dim=35, num_levels=2, side_base_channels=4)
        rgb_features = [
            torch.randn(1, 8, 16, 16),
            torch.randn(1, 8, 8, 8),
        ]

        fused = adapter(
            rgb_features,
            torch.randn(1, 1, 64, 64),
            torch.randn(1, 3, 64, 64),
            torch.randn(1, 35),
        )

        self.assertEqual([feature.shape for feature in fused], [feature.shape for feature in rgb_features])

    def test_rejects_wrong_quality_width(self) -> None:
        adapter = ProjectedRgcFeatureFusion(channels=8, quality_dim=24, num_levels=2)
        rgb_features = [
            torch.randn(1, 8, 16, 16),
            torch.randn(1, 8, 8, 8),
        ]

        with self.assertRaises(ValueError):
            adapter(
                rgb_features,
                torch.randn(1, 1, 64, 64),
                torch.randn(1, 3, 64, 64),
                torch.randn(1, 23),
            )


if __name__ == "__main__":
    unittest.main()
