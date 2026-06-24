import unittest
from pathlib import Path

from scripts.write_bsub_rgc_quality_ablation import render_bsub_script


class WriteBsubRgcQualityAblationTest(unittest.TestCase):
    def test_render_bsub_script_compares_base_and_base_rdt_quality_sets(self) -> None:
        script = render_bsub_script(
            job_name="quality-ablate",
            queue="normal",
            gpu=1,
            fold=0,
            epochs=1,
            lr_drop=1,
            image_max_side=640,
            train_image_max_sides=(480, 640),
            limit_train=32,
            limit_val=16,
            val_batches=4,
            log_gates_batches=2,
            batch_size=1,
            num_workers=0,
            pretrain_dino_weights=Path("/data1/liuxuan/checkpoints/dino/checkpoint0011_4scale.pth"),
            base_quality_cache=Path("outputs/cache/base.json"),
            rdt_output_dir=Path("outputs/cache/rdt"),
            base_rdt_quality_cache=Path("outputs/cache/base_rdt.json"),
            base_output_dir=Path("outputs/base"),
            base_rdt_output_dir=Path("outputs/base_rdt"),
        )

        self.assertIn("scripts/diagnose_rdt_saliency.py", script)
        self.assertIn("--no-write-previews", script)
        self.assertIn("scripts/merge_rdt_quality_cache.py", script)
        self.assertIn("--quality-feature-set \"$FEATURE_SET\"", script)
        self.assertIn("run_ablation base outputs/cache/base.json outputs/base", script)
        self.assertIn("run_ablation base_rdt outputs/cache/base_rdt.json outputs/base_rdt", script)
        self.assertIn("--limit-train 32", script)
        self.assertIn("--limit-val 16", script)


if __name__ == "__main__":
    unittest.main()
