import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from rgc_dino.coco_export import load_split_ids, write_coco_rgb_dataset


class CocoExportTest(unittest.TestCase):
    def test_exports_visible_images_and_yolo_labels_to_coco(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            labels = root / "labels"
            output = Path(tmp) / "coco"
            (root / "visible").mkdir(parents=True)
            labels.mkdir()
            Image.new("RGB", (100, 50), color=(10, 20, 30)).save(root / "visible" / "sample_a.jpg")
            Image.new("RGB", (80, 40), color=(20, 30, 40)).save(root / "visible" / "sample_b.jpg")
            (labels / "sample_a.txt").write_text("1 0.5 0.5 0.2 0.4\n", encoding="utf-8")
            (labels / "sample_b.txt").write_text("0 1.003 0.5 0.5 0.5\n", encoding="utf-8")

            write_coco_rgb_dataset(
                dataset_root=root,
                labels_dir=labels,
                output_root=output,
                train_ids=["sample_a"],
                val_ids=["sample_b"],
                clip_labels=True,
            )

            train = json.loads((output / "annotations" / "instances_train2017.json").read_text())
            val = json.loads((output / "annotations" / "instances_val2017.json").read_text())

            self.assertTrue((output / "train2017" / "sample_a.jpg").is_symlink())
            self.assertEqual(train["images"][0]["width"], 100)
            self.assertEqual(train["images"][0]["height"], 50)
            self.assertEqual(train["annotations"][0]["category_id"], 1)
            self.assertEqual(train["annotations"][0]["bbox"], [40.0, 15.0, 20.0, 20.0])
            self.assertEqual(train["annotations"][0]["area"], 400.0)
            self.assertEqual(val["annotations"][0]["bbox"], [60.0, 10.0, 20.0, 20.0])
            self.assertEqual(len(train["categories"]), 12)

    def test_load_split_ids_selects_requested_fold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            assignments = Path(tmp) / "fold_assignments.jsonl"
            assignments.write_text(
                "\n".join(
                    [
                        '{"fold": 0, "sample_id": "b", "split": "train"}',
                        '{"fold": 0, "sample_id": "a", "split": "val"}',
                        '{"fold": 1, "sample_id": "c", "split": "train"}',
                        '{"fold": 1, "sample_id": "d", "split": "val"}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            train_ids, val_ids = load_split_ids(assignments, fold=0)

            self.assertEqual(train_ids, ["b"])
            self.assertEqual(val_ids, ["a"])


if __name__ == "__main__":
    unittest.main()
