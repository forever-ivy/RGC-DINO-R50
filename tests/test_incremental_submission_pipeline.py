import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_incremental_submission_pipeline.py"


def load_pipeline_module():
    spec = importlib.util.spec_from_file_location("run_incremental_submission_pipeline", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class IncrementalSubmissionPipelineTest(unittest.TestCase):
    def test_select_epochs_to_evaluate_skips_seen_and_min_epoch(self) -> None:
        module = load_pipeline_module()
        with tempfile.TemporaryDirectory() as tmp:
            train_dir = Path(tmp)
            for epoch in [1, 2, 3]:
                (train_dir / f"checkpoint{epoch:04d}.pth").write_bytes(b"weights")
            args = SimpleNamespace(train_dir=train_dir, epochs=None, min_epoch=2)
            state = {"evaluated_epochs": [2]}

            self.assertEqual(module.select_epochs_to_evaluate(args, state), [3])

    def test_should_promote_requires_improvement(self) -> None:
        module = load_pipeline_module()

        self.assertTrue(module.should_promote({"map_50_95": 0.3}, {"best_local_map": None}, 0.0))
        self.assertFalse(module.should_promote({"map_50_95": 0.31}, {"best_local_map": 0.3}, 0.02))
        self.assertTrue(module.should_promote({"map_50_95": 0.33}, {"best_local_map": 0.3}, 0.02))

    def test_run_checkpoint_selection_uses_current_cli(self) -> None:
        module = load_pipeline_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(
                train_dir=root / "train",
                val_ids=root / "val_ids.txt",
                dataset_root_val=root / "train_data",
                labels=root / "labels",
                config_file=root / "config.py",
                device="cpu",
                image_max_side=640,
                side_base_channels=32,
                score_threshold=0.05,
                max_detections=100,
                quality_cache=None,
                nms_iou_threshold=None,
                amp=False,
                dry_run=True,
            )
            eval_dir = root / "eval"

            with patch.object(module, "run_command") as run_command:
                run_command.return_value.returncode = 0
                module.run_checkpoint_selection(args, [1, 2], eval_dir)

            cmd = run_command.call_args.args[0]

        self.assertIn("--train-dir", cmd)
        self.assertIn("--output-dir", cmd)
        self.assertIn("--epochs", cmd)
        self.assertNotIn("--checkpoint-dir", cmd)
        self.assertNotIn("--output-json", cmd)

    def test_load_best_ranking_returns_first_successful_row(self) -> None:
        module = load_pipeline_module()
        with tempfile.TemporaryDirectory() as tmp:
            ranking = Path(tmp) / "ranking.json"
            ranking.write_text(
                json.dumps(
                    [
                        {"checkpoint": "checkpoint0001.pth", "epoch": 1, "map_50_95": None, "error": "bad"},
                        {"checkpoint": "checkpoint0002.pth", "epoch": 2, "map_50_95": 0.4, "map_50": 0.6},
                    ]
                ),
                encoding="utf-8",
            )

            best = module.load_best_ranking(ranking)

        self.assertEqual(best["epoch"], 2)
        self.assertEqual(best["map_50_95"], 0.4)


if __name__ == "__main__":
    unittest.main()
