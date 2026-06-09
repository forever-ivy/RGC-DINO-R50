import unittest

from rgc_dino.labels import parse_label_line


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

if __name__ == "__main__":
    unittest.main()
