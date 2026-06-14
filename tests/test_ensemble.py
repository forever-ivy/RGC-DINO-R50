import tempfile
import unittest
from pathlib import Path

from rgc_dino.ensemble import ensemble_prediction_dirs
from rgc_dino.labels import load_label_file
from rgc_dino.submission import write_submission_files
from rgc_dino.labels import DetectionLabel


class EnsembleTest(unittest.TestCase):
    def test_ensembles_prediction_dirs_with_classwise_nms(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first"
            second = root / "second"
            output = root / "ensemble"
            write_submission_files(
                ["a", "b"],
                {
                    "a": [
                        DetectionLabel(0, 0.5, 0.5, 0.4, 0.4, confidence=0.9),
                    ]
                },
                first,
            )
            write_submission_files(
                ["a", "b"],
                {
                    "a": [
                        DetectionLabel(0, 0.51, 0.5, 0.4, 0.4, confidence=0.8),
                        DetectionLabel(1, 0.51, 0.5, 0.4, 0.4, confidence=0.7),
                    ]
                },
                second,
            )

            summary = ensemble_prediction_dirs(
                [first, second],
                output,
                nms_iou_threshold=0.8,
                max_predictions_per_image=100,
            )

            records = load_label_file(output / "a.txt", require_confidence=True)
            self.assertEqual(summary["files"], 2)
            self.assertEqual(summary["prediction_objects"], 2)
            self.assertEqual([(record.class_id, record.confidence) for record in records], [(0, 0.9), (1, 0.7)])
            self.assertEqual((output / "b.txt").read_text(encoding="utf-8"), "")

    def test_min_model_votes_filters_single_model_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first"
            second = root / "second"
            output = root / "ensemble"
            write_submission_files(
                ["a"],
                {
                    "a": [
                        DetectionLabel(0, 0.5, 0.5, 0.4, 0.4, confidence=0.9),
                        DetectionLabel(1, 0.2, 0.2, 0.1, 0.1, confidence=0.85),
                    ]
                },
                first,
            )
            write_submission_files(
                ["a"],
                {
                    "a": [
                        DetectionLabel(0, 0.51, 0.5, 0.4, 0.4, confidence=0.8),
                    ]
                },
                second,
            )

            ensemble_prediction_dirs(
                [first, second],
                output,
                nms_iou_threshold=0.8,
                min_model_votes=2,
            )

            records = load_label_file(output / "a.txt", require_confidence=True)
            self.assertEqual([(record.class_id, record.confidence) for record in records], [(0, 0.9)])


if __name__ == "__main__":
    unittest.main()
