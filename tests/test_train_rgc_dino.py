import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn

from scripts.train_rgc_dino import (
    _accumulate_gate_stats,
    _build_loaders,
    _build_official_args,
    _finalize_gate_stats,
    _infer_init_checkpoint_scope,
    _quality_median_mad,
    _validate_param_group_coverage,
)
from rgc_dino.quality_features import QUALITY_FEATURE_NAMES


class TrainRgcDinoScriptTest(unittest.TestCase):
    def test_build_official_args_avoids_lr_drop_every_epoch_when_not_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(
                config_file=Path("configs/dino_a0_rgb_4scale.py"),
                output_dir=Path(tmp),
                device="cpu",
                seed=3407,
                num_workers=0,
                amp=False,
                debug=False,
                batch_size=1,
                epochs=12,
                lr_drop=None,
            )

            official_args = _build_official_args(args)

            self.assertEqual(official_args.lr_drop, 11)

    def test_validate_param_group_coverage_rejects_missing_trainable_parameters(self) -> None:
        model = nn.Sequential(
            nn.Linear(2, 3),
            nn.Linear(3, 1),
        )

        with self.assertRaises(ValueError):
            _validate_param_group_coverage(model, [{"params": list(model[0].parameters())}])

    def test_validate_param_group_coverage_accepts_complete_trainable_parameters(self) -> None:
        model = nn.Sequential(
            nn.Linear(2, 3),
            nn.Linear(3, 1),
        )

        _validate_param_group_coverage(model, [{"params": list(model.parameters())}])

    def test_infer_init_checkpoint_scope_detects_rgc_wrapper_state(self) -> None:
        payload = {
            "model": {
                "dino_model.transformer.level_embed": torch.zeros(1),
                "feature_fusion.fusion.alpha_prior": torch.zeros(1),
            }
        }

        self.assertEqual(_infer_init_checkpoint_scope(payload), "rgc")

    def test_infer_init_checkpoint_scope_detects_bare_dino_state(self) -> None:
        payload = {
            "model": {
                "transformer.level_embed": torch.zeros(1),
            }
        }

        self.assertEqual(_infer_init_checkpoint_scope(payload), "dino")

    def test_infer_init_checkpoint_scope_detects_dino_model_payload(self) -> None:
        payload = {
            "dino_model": {
                "transformer.level_embed": torch.zeros(1),
            }
        }

        self.assertEqual(_infer_init_checkpoint_scope(payload), "dino_model_payload")

    def test_gate_stats_are_aggregated_per_modality_and_level(self) -> None:
        collected: dict[str, list[dict[str, float]]] = {}
        counts: dict[str, list[int]] = {}
        _accumulate_gate_stats(
            {
                "ir": [torch.tensor([[[[0.1]]], [[[0.3]]]])],
                "depth": [torch.tensor([[[[0.2]]], [[[0.4]]]])],
            },
            collected,
            counts,
        )
        _accumulate_gate_stats(
            {
                "ir": [torch.tensor([[[[0.5]]]])],
                "depth": [torch.tensor([[[[0.6]]]])],
            },
            collected,
            counts,
        )

        summary = _finalize_gate_stats(collected, counts)

        self.assertAlmostEqual(summary["ir"][0]["mean"], 0.3)
        self.assertAlmostEqual(summary["ir"][0]["min"], 0.1)
        self.assertAlmostEqual(summary["ir"][0]["max"], 0.5)
        self.assertAlmostEqual(summary["depth"][0]["mean"], 0.4)

    def test_quality_median_mad_uses_robust_per_feature_statistics(self) -> None:
        cache = {
            "a": {name: 0.0 for name in QUALITY_FEATURE_NAMES},
            "b": {name: 2.0 for name in QUALITY_FEATURE_NAMES},
            "c": {name: 1000.0 for name in QUALITY_FEATURE_NAMES},
        }

        median, mad = _quality_median_mad(cache)

        self.assertTrue(torch.equal(median, torch.full((24,), 2.0)))
        self.assertTrue(torch.equal(mad, torch.full((24,), 2.0)))

    def test_build_loaders_all_train_mode_allows_empty_validation_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            labels = root / "labels"
            for modality in ("visible", "infrared", "depth"):
                (root / modality).mkdir(parents=True)
            labels.mkdir()
            for sample_id in ("sample_a", "sample_b"):
                Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(root / "visible" / f"{sample_id}.png")
                Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(root / "infrared" / f"{sample_id}.png")
                Image.fromarray(np.zeros((4, 4), dtype=np.uint16)).save(root / "depth" / f"{sample_id}.png")
                (labels / f"{sample_id}.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")

            args = argparse.Namespace(
                dataset_root=root,
                labels=labels,
                assignments=Path(tmp) / "missing.jsonl",
                fold=0,
                train_all=True,
                limit_train=None,
                limit_val=None,
                image_max_side=4,
                train_image_max_sides=None,
                random_horizontal_flip_prob=0.0,
                batch_size=1,
                num_workers=0,
            )

            train_loader, val_loader = _build_loaders(args)

            self.assertEqual(len(train_loader.dataset), 2)
            self.assertIsNone(val_loader)

    def test_build_loaders_uses_multiscale_for_training_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            labels = root / "labels"
            for modality in ("visible", "infrared", "depth"):
                (root / modality).mkdir(parents=True)
            labels.mkdir()
            for sample_id in ("train_a", "val_a"):
                Image.fromarray(np.zeros((4, 8, 3), dtype=np.uint8)).save(root / "visible" / f"{sample_id}.png")
                Image.fromarray(np.zeros((4, 8, 3), dtype=np.uint8)).save(root / "infrared" / f"{sample_id}.png")
                Image.fromarray(np.zeros((4, 8), dtype=np.uint16)).save(root / "depth" / f"{sample_id}.png")
                (labels / f"{sample_id}.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
            assignments = Path(tmp) / "fold_assignments.jsonl"
            assignments.write_text(
                "\n".join(
                    [
                        '{"fold": 0, "sample_id": "train_a", "split": "train"}',
                        '{"fold": 0, "sample_id": "val_a", "split": "val"}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            args = argparse.Namespace(
                dataset_root=root,
                labels=labels,
                assignments=assignments,
                fold=0,
                train_all=False,
                limit_train=None,
                limit_val=None,
                image_max_side=4,
                train_image_max_sides=[4, 8],
                random_horizontal_flip_prob=0.0,
                batch_size=1,
                num_workers=0,
            )

            train_loader, val_loader = _build_loaders(args)

            self.assertEqual(train_loader.dataset.image_max_sides, (4, 8))
            self.assertIsNone(val_loader.dataset.image_max_sides)
            self.assertEqual(val_loader.dataset.image_max_side, 4)


if __name__ == "__main__":
    unittest.main()
