import tempfile
import unittest
from pathlib import Path

from PIL import Image

from rgc_dino.training_splits import select_train_val_ids


class TrainingSplitsTest(unittest.TestCase):
    def test_train_all_uses_every_labeled_aligned_sample_and_no_validation_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            labels = root / "labels"
            for modality in ("visible", "infrared", "depth"):
                (root / modality).mkdir(parents=True)
            labels.mkdir()
            for sample_id in ("sample_b", "sample_a"):
                for modality in ("visible", "infrared", "depth"):
                    Image.new("RGB", (4, 4), color=(1, 2, 3)).save(root / modality / f"{sample_id}.png")
                (labels / f"{sample_id}.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")

            train_ids, val_ids = select_train_val_ids(
                dataset_root=root,
                labels_dir=labels,
                assignments_path=Path(tmp) / "missing.jsonl",
                fold=0,
                train_all=True,
            )

            self.assertEqual(train_ids, ["sample_a", "sample_b"])
            self.assertEqual(val_ids, [])

    def test_fold_mode_uses_assignment_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            assignments = Path(tmp) / "fold_assignments.jsonl"
            assignments.write_text(
                "\n".join(
                    [
                        '{"fold": 0, "sample_id": "train_b", "split": "train"}',
                        '{"fold": 0, "sample_id": "val_a", "split": "val"}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            train_ids, val_ids = select_train_val_ids(
                dataset_root=Path(tmp) / "unused",
                labels_dir=Path(tmp) / "unused" / "labels",
                assignments_path=assignments,
                fold=0,
                train_all=False,
            )

            self.assertEqual(train_ids, ["train_b"])
            self.assertEqual(val_ids, ["val_a"])


if __name__ == "__main__":
    unittest.main()
