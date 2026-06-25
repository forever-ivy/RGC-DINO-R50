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

    def test_should_promote_requires_absolute_anchor_and_improvement(self) -> None:
        module = load_pipeline_module()

        self.assertFalse(module.should_promote({"map_50_95": 0.3}, {"best_local_map": None}, 0.0, 0.4379615851682616))
        self.assertTrue(module.should_promote({"map_50_95": 0.438}, {"best_local_map": None}, 0.0, 0.4379615851682616))
        self.assertFalse(module.should_promote({"map_50_95": 0.31}, {"best_local_map": 0.3}, 0.02, 0.0))
        self.assertTrue(module.should_promote({"map_50_95": 0.33}, {"best_local_map": 0.3}, 0.02, 0.0))

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

    def test_run_once_stops_before_inference_below_current_anchor(self) -> None:
        module = load_pipeline_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_dir = root / "train"
            train_dir.mkdir()
            (train_dir / "checkpoint0001.pth").write_bytes(b"weights")
            args = SimpleNamespace(
                train_dir=train_dir,
                output_dir=root / "automation",
                require_improvement_over=0.0,
                baseline_val_map=0.4379615851682616,
                epochs=None,
                min_epoch=None,
            )
            ranking = root / "ranking.json"
            ranking.write_text(
                json.dumps([{"checkpoint": "checkpoint0001.pth", "epoch": 1, "map_50_95": 0.43, "map_50": 0.6}]),
                encoding="utf-8",
            )

            with patch.object(module, "run_checkpoint_selection", return_value=ranking), patch.object(
                module, "run_test_inference"
            ) as run_test_inference:
                result = module.run_once(args)

        self.assertEqual(result["status"], "below_current_anchor")
        self.assertFalse(run_test_inference.called)

    def test_auto_promote_records_leaderboard_baseline(self) -> None:
        module = load_pipeline_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_dir = root / "train"
            train_dir.mkdir()
            checkpoint = train_dir / "checkpoint0001.pth"
            checkpoint.write_bytes(b"weights")
            args = SimpleNamespace(
                train_dir=train_dir,
                output_dir=root / "automation",
                submissions_dir=root / "submissions",
                dataset_root_test=root / "test",
                require_improvement_over=0.0,
                baseline_val_map=0.4379615851682616,
                leaderboard_baseline=50.353,
                auto_promote=True,
                dry_run=False,
                force=False,
                epochs=None,
                min_epoch=None,
                score_threshold=0.05,
                nms_iou_threshold=None,
                image_max_side=640,
                config_file=root / "config.py",
                quality_cache=None,
            )
            ranking = root / "ranking.json"
            ranking.write_text(
                json.dumps([{"checkpoint": "checkpoint0001.pth", "epoch": 1, "map_50_95": 0.438, "map_50": 0.6}]),
                encoding="utf-8",
            )
            zip_path = root / "candidate.zip"
            manifest_path = root / "candidate.manifest.json"

            captured = {}

            class Promoted:
                zip_path = root / "submissions" / "candidate.zip"
                zip_sha256 = "sha"

            def fake_promote(**kwargs):
                captured.update(kwargs)
                return Promoted()

            with patch.object(module, "run_checkpoint_selection", return_value=ranking), patch.object(
                module, "run_optional_sweep", return_value=(0.05, None, None, None)
            ), patch.object(module, "run_test_inference", return_value=(root / "preds", zip_path, manifest_path)), patch.object(
                module, "promote_submission_candidate", fake_promote
            ), patch.object(module, "file_sha256", return_value="checkpoint-sha"):
                result = module.run_once(args)

        self.assertEqual(result["status"], "promoted")
        self.assertEqual(captured["leaderboard_baseline"], 50.353)
        self.assertEqual(captured["candidate_kind"], "checkpoint_best")


if __name__ == "__main__":
    unittest.main()
