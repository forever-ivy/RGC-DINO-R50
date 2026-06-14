import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from rgc_dino.splits import build_grouped_stratified_splits, write_split_manifest


class SplitsTest(unittest.TestCase):
    def test_build_grouped_stratified_splits_keeps_groups_together(self) -> None:
        labels = {
            "seq1_a": [0, 0],
            "seq1_b": [1],
            "seq2_a": [1, 1],
            "seq3_a": [2],
        }

        splits = build_grouped_stratified_splits(labels, folds=2)

        self.assertEqual(len(splits), 2)
        fold_by_sample = {
            sample_id: fold.fold_index
            for fold in splits
            for sample_id in fold.val_ids
        }
        self.assertEqual(fold_by_sample["seq1_a"], fold_by_sample["seq1_b"])
        self.assertEqual(set(fold_by_sample), set(labels))
        for fold in splits:
            self.assertTrue(fold.train_ids)
            self.assertTrue(fold.val_ids)

    def test_write_split_manifest_writes_json_and_jsonl(self) -> None:
        labels = {"a": [0], "b": [1], "c": [0]}
        splits = build_grouped_stratified_splits(labels, folds=2)

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            manifest_path = write_split_manifest(splits, output_dir)

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            rows = [
                json.loads(line)
                for line in (output_dir / "fold_assignments.jsonl").read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual(manifest["folds"], 2)
            self.assertEqual(len(rows), 6)
            self.assertEqual({row["split"] for row in rows}, {"train", "val"})
            self.assertEqual(
                Counter(row["sample_id"] for row in rows if row["split"] == "val"),
                Counter({"a": 1, "b": 1, "c": 1}),
            )


if __name__ == "__main__":
    unittest.main()
