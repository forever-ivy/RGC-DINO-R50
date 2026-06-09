import unittest

from rgc_dino.constants import CLASS_NAMES, NUM_CLASSES


class ConstantsTest(unittest.TestCase):
    def test_class_count_matches_official_spec(self) -> None:
        self.assertEqual(NUM_CLASSES, 12)
        self.assertEqual(len(CLASS_NAMES), NUM_CLASSES)

    def test_required_classes_are_present(self) -> None:
        self.assertEqual(CLASS_NAMES[0], "person")
        self.assertEqual(CLASS_NAMES[10], "uav")
        self.assertEqual(CLASS_NAMES[11], "tricycle")


if __name__ == "__main__":
    unittest.main()
