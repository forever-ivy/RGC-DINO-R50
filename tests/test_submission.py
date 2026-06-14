import tempfile
import unittest
import zipfile
from pathlib import Path

from rgc_dino.labels import DetectionLabel
from rgc_dino.submission import (
    run_no_detection_inference,
    validate_submission_dir,
    write_empty_submission,
    write_submission_files,
    zip_submission_dir,
)


class SubmissionTest(unittest.TestCase):
    def test_writes_required_txt_files_and_limits_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "submission"
            predictions = {
                "a": [
                    DetectionLabel(0, 0.5, 0.5, 0.2, 0.2, confidence=0.1),
                    DetectionLabel(1, 0.4, 0.4, 0.3, 0.3, confidence=0.9),
                ]
            }

            write_submission_files(["a", "b"], predictions, out_dir, max_predictions_per_image=1)

            self.assertEqual((out_dir / "a.txt").read_text(encoding="utf-8").strip(), "1 0.4 0.4 0.3 0.3 0.9")
            self.assertEqual((out_dir / "b.txt").read_text(encoding="utf-8"), "")
            self.assertEqual(validate_submission_dir(["a", "b"], out_dir), [])

    def test_zip_contains_txt_files_at_archive_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "submission"
            zip_path = Path(tmp) / "submission.zip"

            write_submission_files(["a"], {}, out_dir)
            zip_submission_dir(out_dir, zip_path)

            with zipfile.ZipFile(zip_path) as archive:
                self.assertEqual(archive.namelist(), ["a.txt"])

    def test_empty_submission_writes_every_expected_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "empty"

            write_empty_submission(["b", "a"], out_dir)

            self.assertEqual(sorted(path.name for path in out_dir.glob("*.txt")), ["a.txt", "b.txt"])
            self.assertEqual((out_dir / "a.txt").read_text(encoding="utf-8"), "")

    def test_no_detection_inference_uses_sample_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "predictions"
            out_dir.mkdir()
            (out_dir / "stale.txt").write_text("0 0.1 0.1 0.2 0.2 0.9\n", encoding="utf-8")

            run_no_detection_inference(["x", "y"], out_dir)

            self.assertEqual(sorted(path.name for path in out_dir.glob("*.txt")), ["x.txt", "y.txt"])
            self.assertEqual(validate_submission_dir(["x", "y"], out_dir), [])


if __name__ == "__main__":
    unittest.main()
