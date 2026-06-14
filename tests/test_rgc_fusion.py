import unittest

import torch

from rgc_dino.models.rgc_fusion import ReliabilityGatedResidualFusion


class RgcFusionTest(unittest.TestCase):
    def test_default_initialization_preserves_rgb_features(self) -> None:
        torch.manual_seed(13)
        fusion = ReliabilityGatedResidualFusion(channels=8, quality_dim=24, num_levels=2)
        rgb = [
            torch.randn(2, 8, 8, 8),
            torch.randn(2, 8, 4, 4),
        ]
        infrared = [torch.randn_like(feature) for feature in rgb]
        depth = [torch.randn_like(feature) for feature in rgb]
        quality = torch.randn(2, 24)

        fused, gates = fusion(rgb, infrared, depth, quality, return_gates=True)

        for fused_level, rgb_level in zip(fused, rgb):
            self.assertLessEqual(float((fused_level - rgb_level).detach().abs().max()), 1e-6)
        for modality in ("ir", "depth"):
            for gate in gates[modality]:
                self.assertGreaterEqual(float(gate.detach().min()), 0.0)

    def test_identity_initialization_still_allows_residual_to_learn(self) -> None:
        torch.manual_seed(17)
        fusion = ReliabilityGatedResidualFusion(channels=4, quality_dim=24, num_levels=1)
        rgb = [torch.randn(2, 4, 4, 4)]
        infrared = [torch.randn_like(rgb[0])]
        depth = [torch.randn_like(rgb[0])]
        quality = torch.randn(2, 24)

        fused = fusion(rgb, infrared, depth, quality)
        (fused[0].sum()).backward()

        final_conv = fusion.residual_blocks[0][-1]
        self.assertIsNotNone(final_conv.weight.grad)
        self.assertGreater(float(final_conv.weight.grad.abs().sum()), 0.0)

    def test_fuses_multiscale_features_without_shape_changes(self) -> None:
        torch.manual_seed(0)
        fusion = ReliabilityGatedResidualFusion(
            channels=8,
            quality_dim=24,
            num_levels=3,
            gate_min=0.05,
            gate_max=0.50,
        )
        rgb = [
            torch.randn(2, 8, 16, 16),
            torch.randn(2, 8, 8, 8),
            torch.randn(2, 8, 4, 4),
        ]
        infrared = [torch.randn_like(feature) for feature in rgb]
        depth = [torch.randn_like(feature) for feature in rgb]
        quality = torch.randn(2, 24)

        fused, gates = fusion(rgb, infrared, depth, quality, return_gates=True)

        self.assertEqual([feature.shape for feature in fused], [feature.shape for feature in rgb])
        self.assertEqual(len(gates["ir"]), 3)
        self.assertEqual(gates["ir"][0].shape, (2, 1, 1, 1))
        for modality in ("ir", "depth"):
            for gate in gates[modality]:
                detached = gate.detach()
                self.assertGreaterEqual(float(detached.min()), 0.05)
                self.assertLessEqual(float(detached.max()), 0.50)

    def test_rejects_mismatched_feature_levels(self) -> None:
        fusion = ReliabilityGatedResidualFusion(channels=8, quality_dim=24, num_levels=2)
        rgb = [torch.randn(1, 8, 4, 4), torch.randn(1, 8, 2, 2)]
        infrared = [torch.randn(1, 8, 4, 4)]
        depth = [torch.randn(1, 8, 4, 4), torch.randn(1, 8, 2, 2)]
        quality = torch.randn(1, 24)

        with self.assertRaises(ValueError):
            fusion(rgb, infrared, depth, quality)


if __name__ == "__main__":
    unittest.main()
