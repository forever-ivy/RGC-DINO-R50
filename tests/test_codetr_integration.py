import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from rgc_dino.codetr_integration import check_codetr_tree
from scripts.write_bsub_codetr_smoke import render_bsub_script


ROOT = Path(__file__).resolve().parents[1]


class CodetrIntegrationStatusTest(unittest.TestCase):
    def test_missing_codetr_root_reports_actionable_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "external" / "Co-DETR"

            status = check_codetr_tree(root)

            self.assertFalse(status.ok)
            self.assertEqual(status.root, root)
            self.assertIn("Co-DETR root not found", status.messages[0])
            self.assertIn("git clone https://github.com/Sense-X/Co-DETR", status.clone_hint)

    def test_minimal_codetr_tree_passes_required_path_checks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "external" / "Co-DETR"
            (root / "tools").mkdir(parents=True)
            (root / "mmdet").mkdir()
            (root / "configs").mkdir()
            (root / "tools" / "train.py").write_text("", encoding="utf-8")
            (root / "tools" / "test.py").write_text("", encoding="utf-8")
            (root / "tools" / "dist_train.sh").write_text("", encoding="utf-8")

            status = check_codetr_tree(root)

            self.assertTrue(status.ok)
            self.assertEqual(status.missing_paths, ())

    def test_required_missing_weights_make_status_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "external" / "Co-DETR"
            (root / "tools").mkdir(parents=True)
            (root / "mmdet").mkdir()
            (root / "configs").mkdir()
            (root / "tools" / "train.py").write_text("", encoding="utf-8")
            (root / "tools" / "test.py").write_text("", encoding="utf-8")
            (root / "tools" / "dist_train.sh").write_text("", encoding="utf-8")
            missing_weight = Path(tmp) / "missing.pth"

            status = check_codetr_tree(root, weight_paths=[missing_weight], require_weights=True)

            self.assertFalse(status.ok)
            self.assertEqual(status.missing_weight_paths, (missing_weight,))
            self.assertIn("missing required pretrained weight", "\n".join(status.messages))

    def test_internimage_stage_weights_are_hard_gate_when_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "external" / "Co-DETR"
            (root / "tools").mkdir(parents=True)
            (root / "mmdet").mkdir()
            (root / "configs").mkdir()
            (root / "tools" / "train.py").write_text("", encoding="utf-8")
            (root / "tools" / "test.py").write_text("", encoding="utf-8")
            (root / "tools" / "dist_train.sh").write_text("", encoding="utf-8")
            internimage_weight = Path(tmp) / "internimage_l_public_pretrain.pth"
            codetr_weight = Path(tmp) / "codetr_internimage_l_public_pretrain.pth"

            status = check_codetr_tree(
                root,
                weight_paths=[internimage_weight, codetr_weight],
                require_weights=True,
            )

            self.assertFalse(status.ok)
            self.assertEqual(status.missing_weight_paths, (internimage_weight, codetr_weight))
            self.assertIn(str(internimage_weight), "\n".join(status.messages))
            self.assertIn(str(codetr_weight), "\n".join(status.messages))


class CodetrIntegrationCliTest(unittest.TestCase):
    def test_check_cli_returns_2_for_missing_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "missing-codetr"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "check_codetr_integration.py"),
                    "--codetr-root",
                    str(root),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("Co-DETR root not found", result.stdout)
            self.assertIn("clone_hint:", result.stdout)


class CodetrBsubSmokeTest(unittest.TestCase):
    def test_render_bsub_script_checks_codetr_without_training(self) -> None:
        script = render_bsub_script(
            job_name="codetr-smoke",
            queue="normal",
            gpu=1,
            codetr_root=Path("external/Co-DETR"),
            config=Path("configs/codetr_internimage_l_mm_config.py"),
            coco_output=Path("outputs/codetr_coco/fold0"),
            internimage_weights=Path("/data1/liuxuan/checkpoints/internimage/internimage_l_public_pretrain.pth"),
            codetr_weights=Path("/data1/liuxuan/checkpoints/codetr/codetr_internimage_l_public_pretrain.pth"),
            require_weights=True,
            env_prefix=Path("/data1/liuxuan/envs/codetr"),
        )

        self.assertIn("#BSUB -J codetr-smoke", script)
        self.assertIn("conda activate /data1/liuxuan/envs/codetr", script)
        self.assertIn("python scripts/check_codetr_batch_smoke.py", script)
        self.assertIn("--codetr-root external/Co-DETR", script)
        self.assertIn("--config configs/codetr_internimage_l_mm_config.py", script)
        self.assertIn("--coco-root outputs/codetr_coco/fold0", script)
        self.assertIn("--require-cuda", script)
        self.assertIn("--require-weights", script)
        self.assertNotIn("tools/train.py", script)
        self.assertNotIn("bsub <", script)


if __name__ == "__main__":
    unittest.main()
