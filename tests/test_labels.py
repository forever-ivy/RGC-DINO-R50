import unittest
import tempfile
from pathlib import Path

from rgc_dino.labels import load_label_dir, parse_label_line, parse_label_line_clipped


class LabelsTest(unittest.TestCase):
    def test_parse_train_label_line(self) -> None:
        record = parse_label_line("0 0.5 0.5 0.25 0.125")
        self.assertEqual(record.class_id, 0)
        self.assertEqual(record.norm_center_x, 0.5)
        self.assertIsNone(record.confidence)

    def test_parse_submission_line(self) -> None:
        record = parse_label_line("10 0.1 0.2 0.3 0.4 0.9", require_confidence=True)
        self.assertEqual(record.class_id, 10)
        self.assertEqual(record.confidence, 0.9)

    def test_reject_invalid_width(self) -> None:
        with self.assertRaises(ValueError):
            parse_label_line("1 0.5 0.5 0.0 0.1")

    def test_load_label_dir_keys_by_sample_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            labels = Path(tmp)
            (labels / "b.txt").write_text("1 0.1 0.2 0.3 0.4\n", encoding="utf-8")
            (labels / "a.txt").write_text("", encoding="utf-8")

            loaded = load_label_dir(labels)

            self.assertEqual(list(loaded), ["a", "b"])
            self.assertEqual(loaded["a"], [])
            self.assertEqual(loaded["b"][0].class_id, 1)

    def test_parse_label_line_clipped_reports_minor_drift(self) -> None:
        record, was_clipped = parse_label_line_clipped("0 1.003 0.2 0.3 1.001")

        self.assertTrue(was_clipped)
        self.assertEqual(record.norm_center_x, 1.0)
        self.assertEqual(record.norm_h, 1.0)


if __name__ == "__main__":
    unittest.main()
