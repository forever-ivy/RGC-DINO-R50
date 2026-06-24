"""Adapters for inserting RGC fusion into DINO feature streams."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any, Mapping

import torch
from torch.nn import functional as F
from torch import Tensor, nn

from rgc_dino.quality_features import QUALITY_FEATURE_NAMES

from .rgc_fusion import ReliabilityGatedResidualFusion
from .side_encoder import LightweightSideEncoder


@dataclass(frozen=True)
class RgcDinoSamples:
    """Batch container carrying RGB plus auxiliary modalities for RGC-DINO."""

    rgb: Any
    infrared: Tensor
    depth: Tensor
    quality: Tensor

    def to(self, device: torch.device | str) -> "RgcDinoSamples":
        return RgcDinoSamples(
            rgb=_to_device(self.rgb, device),
            infrared=self.infrared.to(device),
            depth=self.depth.to(device),
            quality=self.quality.to(device),
        )


class RgcDinoModel(nn.Module):
    """Wrap an official DINO model and inject RGC fusion before its transformer."""

    def __init__(
        self,
        dino_model: nn.Module,
        *,
        feature_fusion: nn.Module | None = None,
        side_base_channels: int = 32,
        gate_min: float = 0.0,
        gate_max: float = 0.50,
        quality_dim: int = len(QUALITY_FEATURE_NAMES),
    ) -> None:
        super().__init__()
        self.dino_model = dino_model
        self.feature_fusion = feature_fusion or ProjectedRgcFeatureFusion(
            channels=int(dino_model.hidden_dim),
            quality_dim=quality_dim,
            num_levels=int(dino_model.num_feature_levels),
            side_base_channels=side_base_channels,
            gate_min=gate_min,
            gate_max=gate_max,
        )

    def forward(self, samples: RgcDinoSamples | Mapping[str, Any], targets: list[dict[str, Tensor]] | None = None):
        rgc_samples = _coerce_rgc_samples(samples)
        rgb_samples = _ensure_nested_tensor(rgc_samples.rgb)
        features, poss = self.dino_model.backbone(rgb_samples)

        srcs, masks, poss = self._project_dino_features(features, poss, rgb_samples)
        projected_srcs = srcs
        fused_srcs = self.feature_fusion(
            srcs,
            rgc_samples.infrared,
            rgc_samples.depth,
            rgc_samples.quality,
        )
        srcs = _restore_masked_feature_positions(fused_srcs, projected_srcs, masks)
        return self._run_dino_transformer_and_heads(srcs, masks, poss, targets)

    def compute_fusion_gates(self, samples: RgcDinoSamples | Mapping[str, Any]) -> Mapping[str, list[Tensor]]:
        """Return IR/depth gate tensors from the same feature path used for detection."""
        rgc_samples = _coerce_rgc_samples(samples)
        rgb_samples = _ensure_nested_tensor(rgc_samples.rgb)
        features, poss = self.dino_model.backbone(rgb_samples)
        srcs, _masks, _poss = self._project_dino_features(features, poss, rgb_samples)
        _fused, gates = self.feature_fusion(
            srcs,
            rgc_samples.infrared,
            rgc_samples.depth,
            rgc_samples.quality,
            return_gates=True,
        )
        return gates

    def _project_dino_features(self, features: Sequence[Any], poss: list[Tensor], samples: Any):
        srcs: list[Tensor] = []
        masks: list[Tensor] = []
        for level, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.dino_model.input_proj[level](src))
            masks.append(mask)
            if mask is None:
                raise ValueError(f"DINO feature level {level} has no padding mask")

        if self.dino_model.num_feature_levels > len(srcs):
            nested_tensor_cls = _load_nested_tensor_class()
            original_len = len(srcs)
            for level in range(original_len, self.dino_model.num_feature_levels):
                if level == original_len:
                    src = self.dino_model.input_proj[level](features[-1].tensors)
                else:
                    src = self.dino_model.input_proj[level](srcs[-1])
                mask = F.interpolate(samples.mask[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos = self.dino_model.backbone[1](nested_tensor_cls(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos)
        return srcs, masks, poss

    def _run_dino_transformer_and_heads(
        self,
        srcs: Sequence[Tensor],
        masks: Sequence[Tensor],
        poss: Sequence[Tensor],
        targets: list[dict[str, Tensor]] | None,
    ) -> dict[str, Any]:
        model = self.dino_model
        if model.dn_number > 0 or targets is not None:
            prepare_for_cdn = _load_prepare_for_cdn()
            input_query_label, input_query_bbox, attn_mask, dn_meta = prepare_for_cdn(
                dn_args=(targets, model.dn_number, model.dn_label_noise_ratio, model.dn_box_noise_scale),
                training=model.training,
                num_queries=model.num_queries,
                num_classes=model.num_classes,
                hidden_dim=model.hidden_dim,
                label_enc=model.label_enc,
            )
        else:
            input_query_bbox = input_query_label = attn_mask = dn_meta = None

        hs, reference, hs_enc, ref_enc, init_box_proposal = model.transformer(
            srcs,
            masks,
            input_query_bbox,
            poss,
            input_query_label,
            attn_mask,
        )
        hs[0] += model.label_enc.weight[0, 0] * 0.0

        outputs_coord = []
        for layer_ref_sig, layer_bbox_embed, layer_hs in zip(reference[:-1], model.bbox_embed, hs):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + _inverse_sigmoid(layer_ref_sig)
            outputs_coord.append(layer_outputs_unsig.sigmoid())
        outputs_coord_list = torch.stack(outputs_coord)

        outputs_class = torch.stack(
            [layer_cls_embed(layer_hs) for layer_cls_embed, layer_hs in zip(model.class_embed, hs)]
        )
        if model.dn_number > 0 and dn_meta is not None:
            dn_post_process = _load_dn_post_process()
            outputs_class, outputs_coord_list = dn_post_process(
                outputs_class,
                outputs_coord_list,
                dn_meta,
                model.aux_loss,
                model._set_aux_loss,
            )

        out: dict[str, Any] = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord_list[-1],
        }
        if model.aux_loss:
            out["aux_outputs"] = model._set_aux_loss(outputs_class, outputs_coord_list)
        if hs_enc is not None:
            out.update(_encoder_outputs(model, hs_enc, ref_enc, init_box_proposal))
        out["dn_meta"] = dn_meta
        return out


class ProjectedRgcFeatureFusion(nn.Module):
    """Fuse auxiliary modalities into already-projected DINO feature levels.

    Official DINO converts backbone outputs with ``input_proj`` before sending
    them to the transformer. At that point all levels share ``hidden_dim``
    channels, which is the least invasive place to insert RGC fusion.
    """

    def __init__(
        self,
        *,
        channels: int = 256,
        quality_dim: int = len(QUALITY_FEATURE_NAMES),
        num_levels: int = 4,
        side_in_channels: int = 1,
        depth_in_channels: int = 3,
        side_base_channels: int = 32,
        gate_min: float = 0.0,
        gate_max: float = 0.50,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.quality_dim = quality_dim
        self.num_levels = num_levels

        self.infrared_encoder = LightweightSideEncoder(
            in_channels=side_in_channels,
            channels=channels,
            num_levels=num_levels,
            base_channels=side_base_channels,
            collapse_input_channels=True,
        )
        self.depth_encoder = LightweightSideEncoder(
            in_channels=depth_in_channels,
            channels=channels,
            num_levels=num_levels,
            base_channels=side_base_channels,
            collapse_input_channels=False,
        )
        self.fusion = ReliabilityGatedResidualFusion(
            channels=channels,
            quality_dim=quality_dim,
            num_levels=num_levels,
            gate_min=gate_min,
            gate_max=gate_max,
        )

    def forward(
        self,
        rgb_features: Sequence[Tensor],
        infrared: Tensor,
        depth: Tensor,
        quality: Tensor,
        *,
        return_gates: bool = False,
    ) -> list[Tensor] | tuple[list[Tensor], Mapping[str, list[Tensor]]]:
        reference_features = self._prepare_reference_features(rgb_features)
        first_feature = reference_features[0]
        infrared = self._prepare_modality_tensor(infrared, first_feature, name="infrared")
        depth = self._prepare_modality_tensor(depth, first_feature, name="depth")
        quality = quality.to(device=first_feature.device, dtype=first_feature.dtype)

        infrared_features = self.infrared_encoder(infrared, reference_features=reference_features)
        depth_features = self.depth_encoder(depth, reference_features=reference_features)
        return self.fusion(
            reference_features,
            infrared_features,
            depth_features,
            quality,
            return_gates=return_gates,
        )

    def _prepare_reference_features(self, rgb_features: Sequence[Tensor]) -> list[Tensor]:
        if len(rgb_features) != self.num_levels:
            raise ValueError(f"expected {self.num_levels} projected DINO feature levels, got {len(rgb_features)}")
        prepared = list(rgb_features)
        for level, feature in enumerate(prepared):
            if feature.ndim != 4:
                raise ValueError(f"rgb feature level {level} must be BCHW, got {tuple(feature.shape)}")
            if feature.shape[1] != self.channels:
                raise ValueError(
                    f"rgb feature level {level} has {feature.shape[1]} channels, expected {self.channels}"
                )
        return prepared

    @staticmethod
    def _prepare_modality_tensor(image: Tensor, reference: Tensor, *, name: str) -> Tensor:
        if not torch.is_tensor(image):
            raise TypeError(f"{name} must be a torch.Tensor")
        if image.ndim != 4:
            raise ValueError(f"{name} must be BCHW, got {tuple(image.shape)}")
        if image.shape[0] != reference.shape[0]:
            raise ValueError(f"{name} batch {image.shape[0]} does not match rgb batch {reference.shape[0]}")
        return image.to(device=reference.device, dtype=reference.dtype)


def _coerce_rgc_samples(samples: RgcDinoSamples | Mapping[str, Any]) -> RgcDinoSamples:
    if isinstance(samples, RgcDinoSamples):
        return samples
    if isinstance(samples, Mapping):
        return RgcDinoSamples(
            rgb=samples["rgb"],
            infrared=samples["infrared"],
            depth=samples["depth"],
            quality=samples["quality"],
        )
    raise TypeError("RGC-DINO samples must be RgcDinoSamples or a mapping with rgb/infrared/depth/quality")


def _ensure_nested_tensor(rgb_samples: Any) -> Any:
    if isinstance(rgb_samples, (list, torch.Tensor)):
        nested_tensor_from_tensor_list = _load_nested_tensor_from_tensor_list()
        return nested_tensor_from_tensor_list(rgb_samples)
    return rgb_samples


def _to_device(value: Any, device: torch.device | str) -> Any:
    if hasattr(value, "to"):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    return value


def _restore_masked_feature_positions(
    fused_srcs: Sequence[Tensor],
    original_srcs: Sequence[Tensor],
    masks: Sequence[Tensor],
) -> list[Tensor]:
    restored: list[Tensor] = []
    for level, (fused, original, mask) in enumerate(zip(fused_srcs, original_srcs, masks)):
        if fused.shape != original.shape:
            raise ValueError(f"fused feature level {level} shape changed from {tuple(original.shape)} to {tuple(fused.shape)}")
        if mask.shape != fused.shape[0:1] + fused.shape[-2:]:
            raise ValueError(f"mask level {level} shape {tuple(mask.shape)} does not match feature {tuple(fused.shape)}")
        restored.append(torch.where(mask.to(device=fused.device)[:, None], original, fused))
    return restored


def _inverse_sigmoid(x: Tensor, eps: float = 1e-5) -> Tensor:
    x = x.clamp(min=0.0, max=1.0)
    x1 = x.clamp(min=eps)
    x2 = (1.0 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def _encoder_outputs(model: nn.Module, hs_enc: Tensor, ref_enc: Tensor, init_box_proposal: Tensor) -> dict[str, Any]:
    interm_coord = ref_enc[-1]
    interm_class = model.transformer.enc_out_class_embed(hs_enc[-1])
    output: dict[str, Any] = {
        "interm_outputs": {"pred_logits": interm_class, "pred_boxes": interm_coord},
        "interm_outputs_for_matching_pre": {
            "pred_logits": interm_class,
            "pred_boxes": init_box_proposal,
        },
    }
    if hs_enc.shape[0] > 1:
        enc_outputs_coord = []
        enc_outputs_class = []
        for layer_box_embed, layer_class_embed, layer_hs_enc, layer_ref_enc in zip(
            model.enc_bbox_embed,
            model.enc_class_embed,
            hs_enc[:-1],
            ref_enc[:-1],
        ):
            layer_enc_delta_unsig = layer_box_embed(layer_hs_enc)
            layer_enc_outputs_coord = (layer_enc_delta_unsig + _inverse_sigmoid(layer_ref_enc)).sigmoid()
            enc_outputs_coord.append(layer_enc_outputs_coord)
            enc_outputs_class.append(layer_class_embed(layer_hs_enc))

        output["enc_outputs"] = [
            {"pred_logits": pred_logits, "pred_boxes": pred_boxes}
            for pred_logits, pred_boxes in zip(enc_outputs_class, enc_outputs_coord)
        ]
    return output


def _load_nested_tensor_class():
    from util.misc import NestedTensor

    return NestedTensor


def _load_nested_tensor_from_tensor_list():
    from util.misc import nested_tensor_from_tensor_list

    return nested_tensor_from_tensor_list


def _load_prepare_for_cdn():
    from models.dino.dn_components import prepare_for_cdn

    return prepare_for_cdn


def _load_dn_post_process():
    from models.dino.dn_components import dn_post_process

    return dn_post_process
