import tempfile
import unittest
from pathlib import Path

from rgc_dino.labels import DetectionLabel
from rgc_dino.postprocess import (
    ClassAllocationConfig,
    apply_class_score_thresholds,
    apply_class_score_weights,
    apply_classwise_nms,
    cap_predictions_per_image,
    cap_predictions_per_image_class_aware,
    coerce_class_score_thresholds,
    load_class_score_thresholds,
    score_histograms,
    summarize_predictions,
    topk_truncation_report,
)


class PostprocessTest(unittest.TestCase):
    def test_coerces_thresholds_from_list_and_mapping(self) -> None:
        self.assertEqual(coerce_class_score_thresholds([0.1] * 12), tuple([0.1] * 12))
        thresholds = coerce_class_score_thresholds({"person": 0.003, "4": 0.02})
        self.assertAlmostEqual(thresholds[0], 0.003)
        self.assertAlmostEqual(thresholds[4], 0.02)
        self.assertAlmostEqual(thresholds[1], 0.0)

    def test_loads_thresholds_from_wrapped_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "thresholds.json"
            path.write_text('{"class_conf": {"0": 0.003, "sign": 0.02}}', encoding="utf-8")
            thresholds = load_class_score_thresholds(path)

        self.assertAlmostEqual(thresholds[0], 0.003)
        self.assertAlmostEqual(thresholds[4], 0.02)

    def test_apply_class_score_thresholds_filters_by_class(self) -> None:
        predictions = {
            "a": [
                DetectionLabel(0, 0.5, 0.5, 0.2, 0.2, 0.002),
                DetectionLabel(0, 0.4, 0.4, 0.2, 0.2, 0.004),
                DetectionLabel(1, 0.3, 0.3, 0.2, 0.2, 0.001),
            ]
        }
        thresholds = [0.0] * 12
        thresholds[0] = 0.003

        filtered = apply_class_score_thresholds(predictions, thresholds)

        self.assertEqual([record.confidence for record in filtered["a"]], [0.004, 0.001])

    def test_classwise_nms_keeps_different_classes(self) -> None:
        predictions = {
            "a": [
                DetectionLabel(0, 0.5, 0.5, 0.4, 0.4, 0.9),
                DetectionLabel(0, 0.5, 0.5, 0.38, 0.38, 0.8),
                DetectionLabel(1, 0.5, 0.5, 0.38, 0.38, 0.7),
            ]
        }

        filtered = apply_classwise_nms(predictions, iou_threshold=0.8)

        self.assertEqual([(record.class_id, record.confidence) for record in filtered["a"]], [(0, 0.9), (1, 0.7)])

    def test_cap_and_summary(self) -> None:
        predictions = {
            "a": [
                DetectionLabel(0, 0.5, 0.5, 0.2, 0.2, 0.1),
                DetectionLabel(1, 0.5, 0.5, 0.2, 0.2, 0.9),
            ],
            "b": [],
        }

        capped = cap_predictions_per_image(predictions, max_detections=1)
        summary = summarize_predictions(capped, image_ids=["a", "b"])

        self.assertEqual(capped["a"][0].class_id, 1)
        self.assertEqual(summary["prediction_objects"], 1)
        self.assertEqual(summary["non_empty_images"], 1)
        self.assertEqual(summary["per_class_counts"]["1"], 1)
        self.assertEqual(summary["per_image_count_quantiles"]["max"], 1.0)

    def test_score_histograms_and_topk_report(self) -> None:
        before = {
            "a": [
                DetectionLabel(0, 0.5, 0.5, 0.2, 0.2, 0.9),
                DetectionLabel(1, 0.5, 0.5, 0.2, 0.2, 0.4),
                DetectionLabel(1, 0.5, 0.5, 0.2, 0.2, 0.001),
            ]
        }
        after = cap_predictions_per_image(before, max_detections=2)

        histogram = score_histograms(before, bins=[0.0, 0.01, 0.5, 1.0])
        report = topk_truncation_report(before, after, image_ids=["a"], max_detections=2)

        self.assertEqual(histogram["by_class"]["1"]["[0,0.01)"], 1)
        self.assertEqual(histogram["by_class"]["1"]["[0.01,0.5)"], 1)
        self.assertEqual(report["saturated_image_count"], 1)
        self.assertEqual(report["dropped_prediction_objects"], 1)
        self.assertEqual(report["dropped_by_class"]["1"], 1)

    def test_class_aware_cap_defaults_to_global_topk(self) -> None:
        predictions = {
            "a": [
                DetectionLabel(0, 0.5, 0.5, 0.2, 0.2, 0.3),
                DetectionLabel(1, 0.5, 0.5, 0.2, 0.2, 0.9),
                DetectionLabel(2, 0.5, 0.5, 0.2, 0.2, 0.4),
            ]
        }

        global_cap = cap_predictions_per_image(predictions, max_detections=2)
        class_aware = cap_predictions_per_image_class_aware(predictions, max_detections=2)

        self.assertEqual(
            [(record.class_id, record.confidence) for record in class_aware["a"]],
            [(record.class_id, record.confidence) for record in global_cap["a"]],
        )

    def test_class_score_weights_can_change_topk_order(self) -> None:
        predictions = {
            "a": [
                DetectionLabel(0, 0.5, 0.5, 0.2, 0.2, 0.6),
                DetectionLabel(1, 0.5, 0.5, 0.2, 0.2, 0.5),
            ]
        }
        weights = [1.0] * 12
        weights[1] = 1.3

        weighted = apply_class_score_weights(predictions, weights=weights)
        capped = cap_predictions_per_image_class_aware(
            predictions,
            max_detections=1,
            allocation=ClassAllocationConfig(score_weights=tuple(weights)),
        )

        self.assertAlmostEqual(weighted["a"][1].confidence, 0.65)
        self.assertEqual(capped["a"][0].class_id, 1)
        self.assertLessEqual(len(capped["a"]), 1)

    def test_soft_cap_decays_same_class_tail(self) -> None:
        predictions = {
            "a": [
                DetectionLabel(0, 0.5, 0.5, 0.2, 0.2, 0.9),
                DetectionLabel(0, 0.5, 0.5, 0.2, 0.2, 0.8),
                DetectionLabel(1, 0.5, 0.5, 0.2, 0.2, 0.75),
            ]
        }
        soft_caps = [None] * 12
        soft_caps[0] = 1

        capped = cap_predictions_per_image_class_aware(
            predictions,
            max_detections=2,
            allocation=ClassAllocationConfig(soft_caps=tuple(soft_caps), soft_cap_decay=0.5),
        )

        self.assertEqual([record.class_id for record in capped["a"]], [0, 1])
        self.assertLessEqual(len(capped["a"]), 2)

    def test_reserved_quota_is_opportunistic_and_legal(self) -> None:
        predictions = {
            "a": [
                DetectionLabel(0, 0.5, 0.5, 0.2, 0.2, 0.9),
                DetectionLabel(0, 0.5, 0.5, 0.2, 0.2, 0.8),
                DetectionLabel(1, 0.5, 0.5, 0.2, 0.2, 0.1),
            ],
            "b": [DetectionLabel(0, 0.5, 0.5, 0.2, 0.2, 0.9)],
        }
        quotas = [0] * 12
        quotas[1] = 1
        quotas[2] = 1

        capped = cap_predictions_per_image_class_aware(
            predictions,
            max_detections=2,
            allocation=ClassAllocationConfig(reserved_quotas=tuple(quotas)),
        )

        self.assertEqual([record.class_id for record in capped["a"]], [0, 1])
        self.assertEqual([record.class_id for record in capped["b"]], [0])
        self.assertTrue(all(len(records) <= 2 for records in capped.values()))


if __name__ == "__main__":
    unittest.main()
