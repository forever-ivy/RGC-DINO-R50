import unittest
import subprocess
import sys
import tempfile
from pathlib import Path

from rgc_dino.dino_integration import check_dino_tree
from scripts.write_bsub_dino_smoke import render_bsub_script


ROOT = Path(__file__).resolve().parents[1]


class DinoIntegrationConfigTest(unittest.TestCase):
    def test_dino_config_exists_with_required_keys(self) -> None:
        config = ROOT / "configs" / "dino_r50_4scale.yaml"

        text = config.read_text(encoding="utf-8")

        self.assertIn("external_dino_root:", text)
        self.assertIn("official_repo:", text)
        self.assertIn("official_config:", text)
        self.assertIn("num_classes: 12", text)
        self.assertIn("dn_labelbook_size:", text)


class DinoIntegrationStatusTest(unittest.TestCase):
    def test_missing_dino_root_reports_actionable_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "external" / "IDEA-Research-DINO"

            status = check_dino_tree(root)

            self.assertFalse(status.ok)
            self.assertEqual(status.root, root)
            self.assertIn("DINO root not found", status.messages[0])
            self.assertIn("git clone https://github.com/IDEA-Research/DINO", status.clone_hint)

    def test_minimal_dino_tree_passes_required_path_checks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "external" / "IDEA-Research-DINO"
            (root / "models" / "dino" / "ops").mkdir(parents=True)
            (root / "config" / "DINO").mkdir(parents=True)
            (root / "main.py").write_text("", encoding="utf-8")
            (root / "config" / "DINO" / "DINO_4scale.py").write_text("", encoding="utf-8")

            status = check_dino_tree(root)

            self.assertTrue(status.ok)
            self.assertEqual(status.missing_paths, ())


class DinoIntegrationCliTest(unittest.TestCase):
    def test_check_cli_returns_2_for_missing_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "missing-dino"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "check_dino_integration.py"),
                    "--dino-root",
                    str(root),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("DINO root not found", result.stdout)
            self.assertIn("clone_hint:", result.stdout)

    def test_check_cli_returns_0_for_fake_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dino"
            (root / "models" / "dino" / "ops").mkdir(parents=True)
            (root / "config" / "DINO").mkdir(parents=True)
            (root / "main.py").write_text("", encoding="utf-8")
            (root / "config" / "DINO" / "DINO_4scale.py").write_text("", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "check_dino_integration.py"),
                    "--dino-root",
                    str(root),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0)
            self.assertIn("ok: true", result.stdout)


class DinoBsubSmokeTest(unittest.TestCase):
    def test_render_bsub_script_checks_dino_without_submitting(self) -> None:
        script = render_bsub_script(
            job_name="dino-smoke",
            queue="normal",
            gpu=1,
            dino_root=Path("external/IDEA-Research-DINO"),
        )

        self.assertIn("#BSUB -J dino-smoke", script)
        self.assertIn(". /data1/liuxuan/activate-py310.sh", script)
        self.assertIn("python scripts/check_dino_integration.py --dino-root external/IDEA-Research-DINO", script)
        self.assertNotIn("bsub <", script)
        self.assertNotIn("python main.py", script)


class DinoReadmeWorkflowTest(unittest.TestCase):
    def test_readme_documents_dino_integration_workflow(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("DINO 官方工程集成", readme)
        self.assertIn("scripts/check_dino_integration.py", readme)
        self.assertIn("scripts/write_bsub_dino_smoke.py", readme)
        self.assertIn("external/IDEA-Research-DINO", readme)


if __name__ == "__main__":
    unittest.main()
