import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from rgc_dino.submission_manifest import file_sha256
from rgc_dino.submission_promotion import promote_submission_candidate


def write_candidate_zip(path: Path, sample_ids=("sample_0001",)) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for sample_id in sample_ids:
            archive.writestr(f"{sample_id}.txt", "0 0.9 0.5 0.5 0.1 0.1\n")


def write_manifest_for(zip_path: Path) -> Path:
    manifest = zip_path.with_suffix(".manifest.json")
    manifest.write_text(json.dumps({"zip_sha256": file_sha256(zip_path)}), encoding="utf-8")
    return manifest


class SubmissionPromotionTest(unittest.TestCase):
    def test_promote_submission_candidate_copies_zip_and_writes_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "rgc_fold0_ep3_map0210000_20260615.zip"
            write_candidate_zip(candidate)
            manifest = write_manifest_for(candidate)
            submissions = root / "submissions"

            result = promote_submission_candidate(
                candidate_zip=candidate,
                submissions_dir=submissions,
                reason="local val improved after full-plan gate fix",
                local_map=0.21,
                leaderboard_baseline=34.579,
                manifest_path=manifest,
                expected_ids=["sample_0001"],
                checkpoint_path=root / "missing_checkpoint.pth",
                epoch=3,
                score_threshold=0.05,
            )

            self.assertEqual(result.zip_path, submissions / candidate.name)
            self.assertTrue(result.zip_path.exists())
            sidecar = json.loads(result.metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(sidecar["reason"], "local val improved after full-plan gate fix")
            self.assertEqual(sidecar["local_map"], 0.21)
            self.assertEqual(sidecar["leaderboard_baseline"], 34.579)
            self.assertEqual(sidecar["zip_sha256"], result.zip_sha256)
            self.assertEqual(sidecar["manifest_path"], str(manifest))
            self.assertTrue(sidecar["ready_for_submit"])
            self.assertEqual(sidecar["epoch"], 3)
            self.assertEqual(sidecar["score_threshold"], 0.05)
            self.assertEqual(sidecar["submission_guard"]["expected_txt_count"], 1)

    def test_promote_submission_candidate_rejects_validation_or_debug_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "rgc_fold0_val_debug.zip"
            write_candidate_zip(candidate)
            write_manifest_for(candidate)

            with self.assertRaises(ValueError):
                promote_submission_candidate(
                    candidate_zip=candidate,
                    submissions_dir=root / "submissions",
                    reason="bad candidate",
                    expected_ids=["sample_0001"],
                )

    def test_promote_submission_candidate_rejects_existing_target_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "rgc_fold0_ep3.zip"
            write_candidate_zip(candidate)
            write_manifest_for(candidate)
            submissions = root / "submissions"
            submissions.mkdir()
            (submissions / candidate.name).write_bytes(b"existing")

            with self.assertRaises(FileExistsError):
                promote_submission_candidate(
                    candidate_zip=candidate,
                    submissions_dir=submissions,
                    reason="duplicate",
                    expected_ids=["sample_0001"],
                )

    def test_promote_submission_candidate_requires_text_files_in_zip_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "rgc_fold0_ep3.zip"
            with zipfile.ZipFile(candidate, "w") as archive:
                archive.writestr("nested/sample_0001.txt", "0 0.9 0.5 0.5 0.1 0.1\n")
            write_manifest_for(candidate)

            with self.assertRaises(ValueError):
                promote_submission_candidate(
                    candidate_zip=candidate,
                    submissions_dir=root / "submissions",
                    reason="no root txt files",
                    expected_ids=["sample_0001"],
                )

    def test_promote_submission_candidate_rejects_missing_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "rgc_fold0_ep3.zip"
            write_candidate_zip(candidate)

            with self.assertRaises(FileNotFoundError):
                promote_submission_candidate(
                    candidate_zip=candidate,
                    submissions_dir=root / "submissions",
                    reason="missing manifest",
                    expected_ids=["sample_0001"],
                )

    def test_promote_submission_candidate_rejects_partial_zip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "rgc_fold0_ep3.zip"
            write_candidate_zip(candidate, sample_ids=("sample_0001",))
            write_manifest_for(candidate)

            with self.assertRaises(ValueError):
                promote_submission_candidate(
                    candidate_zip=candidate,
                    submissions_dir=root / "submissions",
                    reason="partial candidate",
                    expected_ids=["sample_0001", "sample_0002"],
                )

    def test_promote_submission_candidate_rejects_manifest_sha_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "rgc_fold0_ep3.zip"
            write_candidate_zip(candidate)
            candidate.with_suffix(".manifest.json").write_text(
                json.dumps({"zip_sha256": "not-the-real-sha"}),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                promote_submission_candidate(
                    candidate_zip=candidate,
                    submissions_dir=root / "submissions",
                    reason="bad manifest",
                    expected_ids=["sample_0001"],
                )


if __name__ == "__main__":
    unittest.main()
