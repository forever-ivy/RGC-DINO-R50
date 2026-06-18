import unittest
from pathlib import Path

from scripts.write_bsub_codetr_train import render_bsub_script


class WriteBsubCodetrTrainTest(unittest.TestCase):
    def test_render_bsub_script_runs_preflight_export_then_external_training(self) -> None:
        script = render_bsub_script(
            job_name="codetr-train",
            queue="normal",
            gpu=2,
            fold=0,
            codetr_root=Path("external/Co-DETR"),
            config=Path("configs/codetr_internimage_l_mm_config.py"),
            coco_output=Path("outputs/codetr_coco/fold0"),
            work_dir=Path("outputs/codetr/work"),
            internimage_weights=Path("/data1/liuxuan/checkpoints/internimage/internimage_l_public_pretrain.pth"),
            codetr_weights=Path("/data1/liuxuan/checkpoints/codetr/codetr_internimage_l_public_pretrain.pth"),
            require_weights=True,
            num_workers=0,
            env_prefix=Path("/data1/liuxuan/envs/codetr"),
        )

        self.assertIn("#BSUB -J codetr-train", script)
        self.assertIn("python scripts/check_codetr_integration.py", script)
        self.assertIn("--require-weights", script)
        self.assertIn("python scripts/export_codetr_coco.py", script)
        self.assertIn("--clip-labels", script)
        self.assertIn("conda activate /data1/liuxuan/envs/codetr", script)
        self.assertIn("python scripts/check_codetr_environment.py", script)
        self.assertIn("bash external/Co-DETR/tools/dist_train.sh configs/codetr_internimage_l_mm_config.py 2 outputs/codetr/work", script)
        self.assertIn("--cfg-options data.workers_per_gpu=0", script)
        self.assertNotIn("bsub <", script)

    def test_render_bsub_script_uses_single_gpu_train_entry_when_gpu_is_one(self) -> None:
        script = render_bsub_script(
            job_name="codetr-train",
            queue="normal",
            gpu=1,
            fold=0,
            codetr_root=Path("external/Co-DETR"),
            config=Path("configs/codetr_internimage_l_mm_config.py"),
            coco_output=Path("outputs/codetr_coco/fold0"),
            work_dir=Path("outputs/codetr/work"),
            internimage_weights=Path("/data1/liuxuan/checkpoints/internimage/internimage_l_public_pretrain.pth"),
            codetr_weights=Path("/data1/liuxuan/checkpoints/codetr/codetr_internimage_l_public_pretrain.pth"),
            require_weights=False,
            num_workers=0,
            env_prefix=Path("/data1/liuxuan/envs/codetr"),
        )

        self.assertIn("python external/Co-DETR/tools/train.py configs/codetr_internimage_l_mm_config.py", script)
        self.assertIn("--cfg-options data.workers_per_gpu=0", script)
        self.assertNotIn("--require-weights", script)


if __name__ == "__main__":
    unittest.main()
