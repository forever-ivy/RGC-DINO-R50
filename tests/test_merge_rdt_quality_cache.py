import tempfile
import unittest
from pathlib import Path

from rgc_dino.quality_features import QUALITY_FEATURE_NAMES, RDT_QUALITY_FEATURE_NAMES, load_quality_feature_cache
from scripts.merge_rdt_quality_cache import merge_quality_and_rdt


class MergeRdtQualityCacheTest(unittest.TestCase):
    def test_merges_base_and_rdt_features_in_base_rdt_order(self) -> None:
        base = {
            "sample_a": {name: float(index) for index, name in enumerate(QUALITY_FEATURE_NAMES)},
        }
        rdt = {
            "sample_a": {name: float(index + 100) for index, name in enumerate(RDT_QUALITY_FEATURE_NAMES)},
        }

        merged = merge_quality_and_rdt(base, rdt)

        self.assertEqual(tuple(merged["sample_a"]), QUALITY_FEATURE_NAMES + RDT_QUALITY_FEATURE_NAMES)
        self.assertEqual(len(merged["sample_a"]), 35)
        self.assertEqual(merged["sample_a"][QUALITY_FEATURE_NAMES[-1]], 23.0)
        self.assertEqual(merged["sample_a"][RDT_QUALITY_FEATURE_NAMES[0]], 100.0)

    def test_merged_cache_loads_as_base_rdt_feature_set(self) -> None:
        base = {
            "sample_a": {name: float(index) for index, name in enumerate(QUALITY_FEATURE_NAMES)},
        }
        rdt = {
            "sample_a": {name: float(index + 100) for index, name in enumerate(RDT_QUALITY_FEATURE_NAMES)},
        }
        merged = merge_quality_and_rdt(base, rdt)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "merged.json"
            from rgc_dino.quality_features import write_quality_feature_cache

            write_quality_feature_cache(path, merged, feature_set="base_rdt")
            loaded = load_quality_feature_cache(path, feature_set="base_rdt")

        self.assertEqual(loaded, merged)


if __name__ == "__main__":
    unittest.main()
