import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from rgc_dino.submission_promotion import promote_submission_candidate


class SubmissionPromotionTest(unittest.TestCase):
    def test_promote_submission_candidate_copies_zip_and_writes_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "rgc_fold0_ep3_map0210000_20260615.zip"
            with zipfile.ZipFile(candidate, "w") as archive:
                archive.writestr("sample_0001.txt", "0 0.9 0.5 0.5 0.1 0.1\n")
            submissions = root / "submissions"

            result = promote_submission_candidate(
                candidate_zip=candidate,
                submissions_dir=submissions,
                reason="local val improved after full-plan gate fix",
                local_map=0.21,
                leaderboard_baseline=34.579,
            )

            self.assertEqual(result.zip_path, submissions / candidate.name)
            self.assertTrue(result.zip_path.exists())
            sidecar = json.loads(result.metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(sidecar["reason"], "local val improved after full-plan gate fix")
            self.assertEqual(sidecar["local_map"], 0.21)
            self.assertEqual(sidecar["leaderboard_baseline"], 34.579)
            self.assertEqual(sidecar["zip_sha256"], result.zip_sha256)

    def test_promote_submission_candidate_rejects_validation_or_debug_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "rgc_fold0_val_debug.zip"
            with zipfile.ZipFile(candidate, "w") as archive:
                archive.writestr("sample_0001.txt", "0 0.9 0.5 0.5 0.1 0.1\n")

            with self.assertRaises(ValueError):
                promote_submission_candidate(
                    candidate_zip=candidate,
                    submissions_dir=root / "submissions",
                    reason="bad candidate",
                )

    def test_promote_submission_candidate_rejects_existing_target_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "rgc_fold0_ep3.zip"
            with zipfile.ZipFile(candidate, "w") as archive:
                archive.writestr("sample_0001.txt", "0 0.9 0.5 0.5 0.1 0.1\n")
            submissions = root / "submissions"
            submissions.mkdir()
            (submissions / candidate.name).write_bytes(b"existing")

            with self.assertRaises(FileExistsError):
                promote_submission_candidate(
                    candidate_zip=candidate,
                    submissions_dir=submissions,
                    reason="duplicate",
                )

    def test_promote_submission_candidate_requires_text_files_in_zip_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "rgc_fold0_ep3.zip"
            with zipfile.ZipFile(candidate, "w") as archive:
                archive.writestr("nested/sample_0001.txt", "0 0.9 0.5 0.5 0.1 0.1\n")

            with self.assertRaises(ValueError):
                promote_submission_candidate(
                    candidate_zip=candidate,
                    submissions_dir=root / "submissions",
                    reason="no root txt files",
                )


if __name__ == "__main__":
    unittest.main()
