import tempfile
import unittest
import json
from pathlib import Path

from rgc_dino.dataset import (
    discover_aligned_samples,
    summarize_multimodal_dataset,
    write_manifest_jsonl,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


class DatasetTest(unittest.TestCase):
    def test_discovers_only_three_modality_aligned_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "images"
            labels = Path(tmp) / "labels"

            for modality in ("visible", "infrared", "depth"):
                _touch(root / modality / "0001.png")
                _touch(root / modality / "0002.jpg")
            _touch(root / "visible" / "visible_only.png")
            _touch(labels / "0001.txt")
            _touch(labels / "extra_label.txt")

            samples = discover_aligned_samples(root, labels_dir=labels)

            self.assertEqual([sample.sample_id for sample in samples], ["0001", "0002"])
            self.assertEqual(samples[0].label_path, labels / "0001.txt")
            self.assertIsNone(samples[1].label_path)

    def test_require_labels_filters_unlabeled_aligned_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "images"
            labels = Path(tmp) / "labels"

            for modality in ("visible", "infrared", "depth"):
                _touch(root / modality / "labeled.png")
                _touch(root / modality / "unlabeled.png")
            _touch(labels / "labeled.txt")

            samples = discover_aligned_samples(root, labels_dir=labels, require_labels=True)

            self.assertEqual([sample.sample_id for sample in samples], ["labeled"])

    def test_summary_reports_missing_modalities_and_extra_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "images"
            labels = Path(tmp) / "labels"

            _touch(root / "visible" / "a.png")
            _touch(root / "infrared" / "a.png")
            _touch(root / "depth" / "a.png")
            _touch(root / "visible" / "b.png")
            _touch(labels / "a.txt")
            _touch(labels / "c.txt")

            summary = summarize_multimodal_dataset(root, labels)

            self.assertEqual(summary.aligned_count, 1)
            self.assertEqual(summary.modality_counts["visible"], 2)
            self.assertEqual(summary.missing_by_modality["infrared"], ["b"])
            self.assertEqual(summary.extra_label_ids, ["c"])

    def test_write_manifest_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "images"
            manifest = Path(tmp) / "manifest.jsonl"
            for modality in ("visible", "infrared", "depth"):
                _touch(root / modality / "a.png")

            samples = discover_aligned_samples(root)
            write_manifest_jsonl(samples, manifest)

            rows = [
                json.loads(line)
                for line in manifest.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(rows[0]["sample_id"], "a")
            self.assertIn("visible", rows[0]["modalities"])
            self.assertIsNone(rows[0]["label_path"])


if __name__ == "__main__":
    unittest.main()
