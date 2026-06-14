import hashlib
import tempfile
import unittest
from pathlib import Path

from rgc_dino.submission_manifest import SubmissionManifest, build_submission_manifest, write_submission_manifest


class SubmissionManifestTest(unittest.TestCase):
    def test_build_submission_manifest_records_hashes_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            zip_path = root / "submission.zip"
            zip_path.write_bytes(b"zip")
            checkpoint = root / "checkpoint.pth"
            checkpoint.write_bytes(b"weights")
            split_manifest = root / "split_manifest.json"
            split_manifest.write_text("{}", encoding="utf-8")

            manifest = build_submission_manifest(
                zip_path=zip_path,
                checkpoint_path=checkpoint,
                git_commit="abc123",
                split_manifest_path=split_manifest,
                calibrator_version="v1",
                config_path=root / "config.yaml",
            )

            self.assertIsInstance(manifest, SubmissionManifest)
            self.assertEqual(manifest.zip_sha256, hashlib.sha256(b"zip").hexdigest())
            self.assertEqual(manifest.checkpoint_sha256, hashlib.sha256(b"weights").hexdigest())
            self.assertEqual(manifest.git_commit, "abc123")
            self.assertEqual(manifest.calibrator_version, "v1")

    def test_write_submission_manifest_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.json"
            zip_path = root / "submission.zip"
            zip_path.write_bytes(b"zip")
            checkpoint = root / "checkpoint.pth"
            checkpoint.write_bytes(b"weights")
            split_manifest = root / "split_manifest.json"
            split_manifest.write_text("{}", encoding="utf-8")

            manifest = build_submission_manifest(
                zip_path=zip_path,
                checkpoint_path=checkpoint,
                git_commit="abc123",
                split_manifest_path=split_manifest,
                calibrator_version="v1",
                config_path=root / "config.yaml",
            )
            write_submission_manifest(manifest_path, manifest)

            self.assertTrue(manifest_path.exists())
            self.assertIn("abc123", manifest_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
