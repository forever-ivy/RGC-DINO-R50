import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from rgc_dino.rdt import compute_rdt_result, load_rdt_result, normalize_depth_to_uint8, write_rdt_preview


class RdtTest(unittest.TestCase):
    def test_normalizes_uint16_depth_and_valid_mask(self) -> None:
        depth = np.array([[0, 299, 300, 1000, 20000, 25000]], dtype=np.uint16)

        normalized, valid = normalize_depth_to_uint8(depth)

        self.assertEqual(normalized.shape, depth.shape)
        self.assertEqual(valid.tolist(), [[False, False, True, True, True, False]])
        self.assertEqual(int(normalized[0, 0]), 0)
        self.assertGreater(int(normalized[0, 3]), 0)

    def test_computes_guided_rgb_and_stats(self) -> None:
        rgb = np.full((4, 4, 3), 100, dtype=np.uint8)
        infrared = np.zeros((4, 4), dtype=np.uint8)
        infrared[1:3, 1:3] = 255
        depth = np.full((4, 4), 1000, dtype=np.uint16)

        result = compute_rdt_result(rgb, infrared, depth)

        self.assertEqual(result.guided_rgb.shape, rgb.shape)
        self.assertIn("rdt_attention_mean", result.stats)
        self.assertIn("rdt_depth_valid_ratio", result.stats)
        self.assertAlmostEqual(result.stats["rdt_depth_valid_ratio"], 1.0)
        self.assertGreaterEqual(float(result.guided_rgb.max()), 85.0)

    def test_loads_from_paths_and_writes_preview(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rgb_path = root / "rgb.png"
            ir_path = root / "ir.png"
            depth_path = root / "depth.png"
            out_path = root / "preview.jpg"
            Image.fromarray(np.full((6, 8, 3), 120, dtype=np.uint8)).save(rgb_path)
            Image.fromarray(np.full((6, 8), 80, dtype=np.uint8)).save(ir_path)
            Image.fromarray(np.full((6, 8), 1000, dtype=np.uint16)).save(depth_path)

            result = load_rdt_result(rgb_path, ir_path, depth_path, max_side=4)
            write_rdt_preview(result, out_path)

            self.assertTrue(out_path.exists())
            self.assertLessEqual(max(result.rgb.shape[:2]), 4)


if __name__ == "__main__":
    unittest.main()
