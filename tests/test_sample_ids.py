import tempfile
import unittest
from pathlib import Path

from rgc_dino.sample_ids import load_sample_ids_file, restrict_mapping_to_sample_ids


class SampleIdsTest(unittest.TestCase):
    def test_load_sample_ids_file_strips_comments_blanks_and_preserves_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ids.txt"
            path.write_text(
                "\n".join(
                    [
                        "# fold0 validation ids",
                        "sample_b",
                        "",
                        "sample_a  ",
                        "sample_b",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            self.assertEqual(load_sample_ids_file(path), ["sample_b", "sample_a"])

    def test_restrict_mapping_to_sample_ids_keeps_requested_ids_only(self) -> None:
        values = {"sample_a": [1], "sample_b": [2], "sample_c": [3]}

        self.assertEqual(
            restrict_mapping_to_sample_ids(values, ["sample_c", "missing", "sample_a"]),
            {"sample_c": [3], "sample_a": [1]},
        )


if __name__ == "__main__":
    unittest.main()
