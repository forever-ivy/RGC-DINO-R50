import importlib.util
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "monitor_competition.py"


def load_monitor_module():
    spec = importlib.util.spec_from_file_location("monitor_competition", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
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

            def fake_run(cmd, timeout):
                calls.append((cmd, timeout))
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


if __name__ == "__main__":
    unittest.main()
