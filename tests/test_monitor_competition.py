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


def write_promoted_candidate(
    module,
    root: Path,
    *,
    candidate_kind: str | None = None,
    local_map: float = 0.5,
    leaderboard_baseline: float = 50.353,
    hard_val_map: float | None = None,
    hard_val_status: str | None = None,
) -> Path:
    zip_path = root / "candidate.zip"
    zip_path.write_bytes(b"zip")
    manifest = root / "candidate.manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    payload = {
        "zip_sha256": module.file_sha256(zip_path),
        "manifest_path": str(manifest),
        "ready_for_submit": True,
        "reason": "unit test candidate",
        "local_map": local_map,
        "val_map_50_95": local_map,
        "leaderboard_baseline": leaderboard_baseline,
    }
    if candidate_kind is not None:
        payload["candidate_kind"] = candidate_kind
    if hard_val_map is not None:
        payload["hard_val_map_50_95"] = hard_val_map
    if hard_val_status is not None:
        payload["hard_val_status"] = hard_val_status
    zip_path.with_suffix(".promotion.json").write_text(json.dumps(payload), encoding="utf-8")
    return zip_path


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
            zip_path = write_promoted_candidate(module, root, local_map=0.1, leaderboard_baseline=0.0)
            monitor = module.CompetitionMonitor(output_dir=root / "monitor")

            expected_sha = module.file_sha256(zip_path)
            decision = monitor.evaluate_candidate(zip_path)

        self.assertTrue(decision.eligible)
        self.assertEqual(decision.sha256, expected_sha)

    def test_codetr_candidate_must_beat_current_strict_gate(self) -> None:
        module = load_monitor_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            zip_path = write_promoted_candidate(
                module,
                root,
                candidate_kind="codetr_internimage_l",
                local_map=0.4262677082771047,
            )
            monitor = module.CompetitionMonitor(output_dir=root / "monitor")

            decision = monitor.evaluate_candidate(zip_path)

        self.assertFalse(decision.eligible)
        self.assertIn("does not beat current codetr_internimage_l gate", decision.reason)

    def test_incremental_candidate_kinds_are_gated(self) -> None:
        module = load_monitor_module()
        for candidate_kind in ("inference_sweep_best", "checkpoint_best"):
            with self.subTest(candidate_kind=candidate_kind):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    zip_path = write_promoted_candidate(
                        module,
                        root,
                        candidate_kind=candidate_kind,
                        local_map=0.4370,
                    )
                    monitor = module.CompetitionMonitor(output_dir=root / "monitor")

                    decision = monitor.evaluate_candidate(zip_path)

                self.assertFalse(decision.eligible)
                self.assertIn(f"current {candidate_kind} gate", decision.reason)

    def test_codetr_candidate_rejects_failing_or_regressed_hard_val(self) -> None:
        module = load_monitor_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            zip_path = write_promoted_candidate(
                module,
                root,
                candidate_kind="codetr_internimage_l",
                local_map=0.438,
                hard_val_status="fail",
            )
            monitor = module.CompetitionMonitor(output_dir=root / "monitor")

            decision = monitor.evaluate_candidate(zip_path)

        self.assertFalse(decision.eligible)
        self.assertIn("hard-val status is failing", decision.reason)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            zip_path = write_promoted_candidate(
                module,
                root,
                candidate_kind="codetr_internimage_l",
                local_map=0.438,
                hard_val_map=0.29,
            )
            monitor = module.CompetitionMonitor(output_dir=root / "monitor")

            decision = monitor.evaluate_candidate(zip_path)

        self.assertFalse(decision.eligible)
        self.assertIn("hard-val mAP", decision.reason)

    def test_candidate_skips_submitted_sha_and_cooldown(self) -> None:
        module = load_monitor_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            zip_path = write_promoted_candidate(module, root, local_map=0.1, leaderboard_baseline=0.0)
            sha = module.file_sha256(zip_path)
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
