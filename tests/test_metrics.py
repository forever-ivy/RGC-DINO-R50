import unittest

from rgc_dino.labels import DetectionLabel
from rgc_dino.metrics import box_iou_xyxy, evaluate_detection_map, map_summary, xywh_to_xyxy


class MetricsTest(unittest.TestCase):
    def test_xywh_to_xyxy_and_iou(self) -> None:
        box = xywh_to_xyxy(DetectionLabel(0, 0.5, 0.5, 0.2, 0.4))

        self.assertEqual(box, (0.4, 0.3, 0.6, 0.7))
        self.assertAlmostEqual(box_iou_xyxy(box, box), 1.0)

    def test_perfect_prediction_scores_one(self) -> None:
        ground_truths = {"img1": [DetectionLabel(0, 0.5, 0.5, 0.2, 0.2)]}
        predictions = {"img1": [DetectionLabel(0, 0.5, 0.5, 0.2, 0.2, confidence=0.9)]}

        result = evaluate_detection_map(ground_truths, predictions, iou_thresholds=(0.5, 0.75))

        self.assertAlmostEqual(result.map, 1.0)
        self.assertEqual(result.ground_truth_count, 1)
        self.assertEqual(result.prediction_count, 1)

    def test_wrong_class_prediction_scores_zero(self) -> None:
        ground_truths = {"img1": [DetectionLabel(0, 0.5, 0.5, 0.2, 0.2)]}
        predictions = {"img1": [DetectionLabel(1, 0.5, 0.5, 0.2, 0.2, confidence=0.9)]}

        result = evaluate_detection_map(ground_truths, predictions, iou_thresholds=(0.5,))

        self.assertAlmostEqual(result.map, 0.0)

    def test_missing_confidence_is_rejected(self) -> None:
        ground_truths = {"img1": [DetectionLabel(0, 0.5, 0.5, 0.2, 0.2)]}
        predictions = {"img1": [DetectionLabel(0, 0.5, 0.5, 0.2, 0.2)]}

        with self.assertRaises(ValueError):
            evaluate_detection_map(ground_truths, predictions)

    def test_map_summary_exposes_ap75_ap90_and_per_class(self) -> None:
        ground_truths = {"img1": [DetectionLabel(0, 0.5, 0.5, 0.2, 0.2)]}
        predictions = {"img1": [DetectionLabel(0, 0.5, 0.5, 0.2, 0.2, confidence=0.9)]}

        result = evaluate_detection_map(ground_truths, predictions)
        summary = map_summary(result)

        self.assertAlmostEqual(summary["map_75"], 1.0)
        self.assertAlmostEqual(summary["map_90"], 1.0)
        class0 = summary["per_class_ap"][0]
        self.assertEqual(class0["class_id"], 0)
        self.assertAlmostEqual(class0["ap_50_95"], 1.0)


if __name__ == "__main__":
    unittest.main()
