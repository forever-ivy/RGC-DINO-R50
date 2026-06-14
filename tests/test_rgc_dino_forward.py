import unittest

import torch
from torch import nn

from rgc_dino.models.rgc_dino_adapter import RgcDinoModel, RgcDinoSamples


class FakeNestedTensor:
    def __init__(self, tensors: torch.Tensor, mask: torch.Tensor) -> None:
        self.tensors = tensors
        self.mask = mask

    def decompose(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.tensors, self.mask

    def to(self, device: torch.device | str) -> "FakeNestedTensor":
        return FakeNestedTensor(self.tensors.to(device), self.mask.to(device))


class FakeBackbone(nn.Module):
    num_channels = [8]

    def forward(self, samples: FakeNestedTensor) -> tuple[list[FakeNestedTensor], list[torch.Tensor]]:
        return [samples], [torch.zeros_like(samples.tensors)]


class RecordingTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen_srcs: list[torch.Tensor] = []

    def forward(self, srcs, masks, input_query_bbox, poss, input_query_label, attn_mask):
        self.seen_srcs = [src.detach().clone() for src in srcs]
        batch = srcs[0].shape[0]
        hidden = srcs[0].shape[1]
        num_queries = 2
        hs = srcs[0].new_zeros((1, batch, num_queries, hidden))
        reference = srcs[0].new_full((2, batch, num_queries, 4), 0.5)
        return hs, reference, None, None, None


class AddOneFusion(nn.Module):
    def forward(self, rgb_features, infrared, depth, quality, *, return_gates=False):
        self.seen_quality = quality
        return [feature + 1.0 for feature in rgb_features]


class ZeroBBoxHead(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.new_zeros((*x.shape[:-1], 4))


class FakeDino(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = FakeBackbone()
        self.input_proj = nn.ModuleList([nn.Identity()])
        self.num_feature_levels = 1
        self.dn_number = 0
        self.num_queries = 2
        self.num_classes = 3
        self.hidden_dim = 8
        self.label_enc = nn.Embedding(1, 8)
        self.transformer = RecordingTransformer()
        self.bbox_embed = nn.ModuleList([ZeroBBoxHead()])
        self.class_embed = nn.ModuleList([nn.Linear(8, 3)])
        self.aux_loss = False

    def _set_aux_loss(self, outputs_class, outputs_coord):
        return []


class RgcDinoModelTest(unittest.TestCase):
    def test_compute_fusion_gates_uses_projected_feature_path(self) -> None:
        torch.manual_seed(23)
        base = FakeDino()
        wrapper = RgcDinoModel(base, side_base_channels=4)
        rgb = FakeNestedTensor(
            torch.zeros(2, 8, 4, 4),
            torch.zeros(2, 4, 4, dtype=torch.bool),
        )
        samples = RgcDinoSamples(
            rgb=rgb,
            infrared=torch.randn(2, 1, 16, 16),
            depth=torch.randn(2, 3, 16, 16),
            quality=torch.randn(2, 24),
        )

        gates = wrapper.compute_fusion_gates(samples)

        self.assertEqual(set(gates), {"ir", "depth"})
        self.assertEqual(len(gates["ir"]), 1)
        self.assertEqual(gates["ir"][0].shape, (2, 1, 1, 1))
        self.assertEqual(gates["depth"][0].shape, (2, 1, 1, 1))

    def test_fuses_projected_features_before_transformer(self) -> None:
        base = FakeDino()
        wrapper = RgcDinoModel(base, feature_fusion=AddOneFusion())
        rgb = FakeNestedTensor(
            torch.zeros(2, 8, 4, 4),
            torch.zeros(2, 4, 4, dtype=torch.bool),
        )
        samples = RgcDinoSamples(
            rgb=rgb,
            infrared=torch.randn(2, 1, 16, 16),
            depth=torch.randn(2, 1, 16, 16),
            quality=torch.randn(2, 24),
        )

        outputs = wrapper(samples)

        self.assertTrue(torch.equal(base.transformer.seen_srcs[0], torch.ones(2, 8, 4, 4)))
        self.assertEqual(outputs["pred_logits"].shape, (2, 2, 3))
        self.assertEqual(outputs["pred_boxes"].shape, (2, 2, 4))
        self.assertIsNone(outputs["dn_meta"])

    def test_does_not_inject_fusion_into_masked_padding(self) -> None:
        base = FakeDino()
        wrapper = RgcDinoModel(base, feature_fusion=AddOneFusion())
        mask = torch.zeros(2, 4, 4, dtype=torch.bool)
        mask[1, 2:, :] = True
        mask[1, :, 3:] = True
        rgb = FakeNestedTensor(torch.zeros(2, 8, 4, 4), mask)
        samples = RgcDinoSamples(
            rgb=rgb,
            infrared=torch.randn(2, 1, 16, 16),
            depth=torch.randn(2, 1, 16, 16),
            quality=torch.randn(2, 24),
        )

        wrapper(samples)

        fused = base.transformer.seen_srcs[0]
        self.assertTrue(torch.equal(fused[1, :, 0:2, 0:3], torch.ones(8, 2, 3)))
        self.assertTrue(torch.equal(fused[1, :, 2:, :], torch.zeros(8, 2, 4)))
        self.assertTrue(torch.equal(fused[1, :, :, 3:], torch.zeros(8, 4, 1)))


if __name__ == "__main__":
    unittest.main()
