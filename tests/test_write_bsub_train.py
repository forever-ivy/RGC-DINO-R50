import unittest
from pathlib import Path

from scripts.write_bsub_train import render_bsub_script


class WriteBsubTrainTest(unittest.TestCase):
    def test_render_bsub_script_uses_current_rgc_short_train_entry(self) -> None:
        script = render_bsub_script(
            job_name="rgc-short",
            queue="normal",
            gpu=1,
            output_dir=Path("outputs/rgc_dino/short"),
            epochs=3,
            lr_drop=3,
            fold=0,
            image_max_side=640,
            official_dino_checkpoint=Path("outputs/checkpoints/checkpoint0011_4scale.pth"),
            fallback_init_checkpoint=Path("outputs/checkpoints/a0.pth"),
            quality_cache=Path("outputs/cache/quality.json"),
            random_horizontal_flip_prob=0.5,
            log_gates_batches=2,
        )

        self.assertIn("#BSUB -J rgc-short", script)
        self.assertIn("python scripts/cache_quality_features.py", script)
        self.assertIn("python scripts/train_rgc_dino.py", script)
        self.assertIn('INIT_DINO_CHECKPOINT="outputs/checkpoints/checkpoint0011_4scale.pth"', script)
        self.assertIn('INIT_DINO_CHECKPOINT="outputs/checkpoints/a0.pth"', script)
        self.assertIn('--init-dino-checkpoint "$INIT_DINO_CHECKPOINT"', script)
        self.assertIn("--quality-cache outputs/cache/quality.json", script)
        self.assertIn("--random-horizontal-flip-prob 0.5", script)
        self.assertIn("--log-gates-batches 2", script)
        self.assertIn("--amp", script)
        self.assertNotIn("scripts/train_baseline.py", script)
        self.assertNotIn("bsub <", script)


if __name__ == "__main__":
    unittest.main()
