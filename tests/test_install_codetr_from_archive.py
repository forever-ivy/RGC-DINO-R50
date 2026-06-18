import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.install_codetr_from_archive import _find_codetr_root


class InstallCodetrFromArchiveTest(unittest.TestCase):
    def test_find_codetr_root_inside_extracted_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "Co-DETR-main"
            (root / "tools").mkdir(parents=True)
            (root / "mmdet").mkdir()
            (root / "tools" / "train.py").write_text("", encoding="utf-8")
            (root / "tools" / "test.py").write_text("", encoding="utf-8")

            self.assertEqual(_find_codetr_root(Path(tmp)), root)

    def test_archive_fixture_shape_is_supported_by_zipfile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "codetr.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("Co-DETR-main/tools/train.py", "")
                handle.writestr("Co-DETR-main/tools/test.py", "")
                handle.writestr("Co-DETR-main/mmdet/__init__.py", "")
            extract_root = Path(tmp) / "extract"
            with zipfile.ZipFile(archive) as handle:
                handle.extractall(extract_root)

            self.assertEqual(_find_codetr_root(extract_root), extract_root / "Co-DETR-main")


if __name__ == "__main__":
    unittest.main()
