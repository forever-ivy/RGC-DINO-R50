import unittest

import torch

from rgc_dino.models.side_encoder import LightweightSideEncoder


class SideEncoderTest(unittest.TestCase):
    def test_collapses_ir_and_matches_reference_feature_shapes(self) -> None:
        encoder = LightweightSideEncoder(
            in_channels=1,
            channels=8,
            num_levels=3,
            base_channels=4,
            collapse_input_channels=True,
        )
        image = torch.randn(2, 3, 64, 64)
        references = [
            torch.empty(2, 8, 16, 16),
            torch.empty(2, 8, 8, 8),
            torch.empty(2, 8, 4, 4),
        ]

        features = encoder(image, reference_features=references)

        self.assertEqual([feature.shape for feature in features], [feature.shape for feature in references])

    def test_rejects_unexpected_input_channels_when_not_collapsing(self) -> None:
        encoder = LightweightSideEncoder(
            in_channels=1,
            channels=8,
            num_levels=2,
            collapse_input_channels=False,
        )

        with self.assertRaises(ValueError):
            encoder(torch.randn(1, 3, 32, 32))


if __name__ == "__main__":
    unittest.main()
