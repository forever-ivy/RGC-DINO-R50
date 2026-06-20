import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "monitor_competition.py"


def load_monitor_module():
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("monitor_competition", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["monitor_competition"] = module
    spec.loader.exec_module(module)
    return module


class MonitorCompetitionTest(unittest.TestCase):
    def test_submit_prediction_passes_local_storage_to_submit_script(self) -> None:
        module = load_monitor_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            zip_path = root / "submission.zip"
            zip_path.write_bytes(b"zip")

            monitor = module.CompetitionMonitor(
                output_dir=root / "monitor",
                cookies_file=Path("outputs/cookies.json"),
                user_data_dir=Path("/data1/liuxuan/chrome-competition-profile"),
                local_storage_file=Path("outputs/aicomp_auth.json"),
            )

            calls = []

            class Result:
                returncode = 0

            def fake_run(cmd, timeout, cwd):
                calls.append((cmd, timeout, cwd))
                return Result()

            with patch.object(module.subprocess, "run", fake_run):
                ok = monitor.submit_prediction(zip_path)

        self.assertTrue(ok)
        cmd = calls[0][0]
        self.assertIn("--local-storage", cmd)
        self.assertIn("outputs/aicomp_auth.json", cmd)
        self.assertIn("--user-data-dir", cmd)
        self.assertIn("/data1/liuxuan/chrome-competition-profile", cmd)

    def test_mark_existing_predictions_seen_sets_submission_baseline(self) -> None:
        module = load_monitor_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = root / "submissions"
            predictions.mkdir()
            (predictions / "old.zip").write_bytes(b"zip")

            monitor = module.CompetitionMonitor(output_dir=root / "monitor")
            self.assertIsNone(monitor.state["last_submission"])

            monitor.mark_existing_predictions_seen(predictions)

            state = json.loads((root / "monitor" / "monitor_state.json").read_text())

        self.assertIsNotNone(state["last_submission"])
        datetime.fromisoformat(state["last_submission"])
        self.assertEqual(len(state["ignored_sha256s"]), 1)

    def test_candidate_requires_promotion_sidecar_by_default(self) -> None:
        module = load_monitor_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            zip_path = root / "candidate.zip"
            zip_path.write_bytes(b"zip")
            monitor = module.CompetitionMonitor(output_dir=root / "monitor")

            decision = monitor.evaluate_candidate(zip_path)

        self.assertFalse(decision.eligible)
        self.assertIn("missing .promotion.json", decision.reason)

    def test_candidate_with_sidecar_and_manifest_is_eligible(self) -> None:
        module = load_monitor_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            zip_path = root / "candidate.zip"
            zip_path.write_bytes(b"zip")
            manifest = root / "candidate.manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            sha = module.file_sha256(zip_path)
            zip_path.with_suffix(".promotion.json").write_text(
                json.dumps(
                    {
                        "zip_sha256": sha,
                        "manifest_path": str(manifest),
                        "ready_for_submit": True,
                        "reason": "unit test candidate",
                        "local_map": 0.1,
                        "leaderboard_baseline": 0.0,
                    }
                ),
                encoding="utf-8",
            )
            monitor = module.CompetitionMonitor(output_dir=root / "monitor")

            decision = monitor.evaluate_candidate(zip_path)

        self.assertTrue(decision.eligible)
        self.assertEqual(decision.sha256, sha)

    def test_candidate_skips_submitted_sha_and_cooldown(self) -> None:
        module = load_monitor_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            zip_path = root / "candidate.zip"
            zip_path.write_bytes(b"zip")
            manifest = root / "candidate.manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            sha = module.file_sha256(zip_path)
            zip_path.with_suffix(".promotion.json").write_text(
                json.dumps(
                    {
                        "zip_sha256": sha,
                        "manifest_path": str(manifest),
                        "ready_for_submit": True,
                        "reason": "unit test candidate",
                        "local_map": 0.1,
                        "leaderboard_baseline": 0.0,
                    }
                ),
                encoding="utf-8",
            )
            monitor = module.CompetitionMonitor(output_dir=root / "monitor")
            monitor.state["submitted_sha256s"] = [sha]

            decision = monitor.evaluate_candidate(zip_path)
            self.assertFalse(decision.eligible)
            self.assertIn("already submitted", decision.reason)

            monitor.state["submitted_sha256s"] = []
            monitor.state["cooldown_until"] = (datetime.now() + timedelta(hours=1)).isoformat()
            decision = monitor.evaluate_candidate(zip_path)
            self.assertFalse(decision.eligible)
            self.assertIn("cooldown", decision.reason)


if __name__ == "__main__":
    unittest.main()
