import unittest

import torch

from rgc_dino.dino_inference import ClasswiseScoreCalibrator, dino_result_to_detection_labels


class DinoInferenceTest(unittest.TestCase):
    def test_converts_postprocessed_xyxy_boxes_to_submission_labels(self) -> None:
        result = {
            "scores": torch.tensor([0.9, 0.1]),
            "labels": torch.tensor([2, 3]),
            "boxes": torch.tensor(
                [
                    [10.0, 20.0, 30.0, 60.0],
                    [0.0, 0.0, 10.0, 10.0],
                ]
            ),
        }

        labels = dino_result_to_detection_labels(
            result,
            orig_height=100,
            orig_width=200,
            score_threshold=0.5,
            max_detections=100,
        )

        self.assertEqual(len(labels), 1)
        self.assertEqual(labels[0].class_id, 2)
        self.assertAlmostEqual(labels[0].norm_center_x, 0.1)
        self.assertAlmostEqual(labels[0].norm_center_y, 0.4)
        self.assertAlmostEqual(labels[0].norm_w, 0.1)
        self.assertAlmostEqual(labels[0].norm_h, 0.4)
        self.assertAlmostEqual(labels[0].confidence or 0.0, 0.9)

    def test_optional_nms_suppresses_same_class_overlaps_only(self) -> None:
        result = {
            "scores": torch.tensor([0.95, 0.90, 0.85, 0.80]),
            "labels": torch.tensor([2, 2, 3, 2]),
            "boxes": torch.tensor(
                [
                    [10.0, 10.0, 50.0, 50.0],
                    [12.0, 12.0, 48.0, 48.0],
                    [12.0, 12.0, 48.0, 48.0],
                    [70.0, 70.0, 90.0, 90.0],
                ]
            ),
        }

        labels = dino_result_to_detection_labels(
            result,
            orig_height=100,
            orig_width=100,
            score_threshold=0.5,
            max_detections=100,
            nms_iou_threshold=0.8,
        )

        self.assertEqual([label.class_id for label in labels], [2, 3, 2])
        self.assertEqual([round(label.confidence or 0.0, 2) for label in labels], [0.95, 0.85, 0.80])

    def test_class_score_thresholds_filter_after_calibration(self) -> None:
        result = {
            "scores": torch.tensor([0.8, 0.7, 0.6]),
            "labels": torch.tensor([1, 2, 1]),
            "boxes": torch.tensor(
                [
                    [0.0, 0.0, 10.0, 10.0],
                    [10.0, 10.0, 20.0, 20.0],
                    [20.0, 20.0, 30.0, 30.0],
                ]
            ),
        }
        calibrator = ClasswiseScoreCalibrator.from_mapping({"1": {"scale": 0.5}, "2": {"bias": 0.2}})
        thresholds = [0.0] * 12
        thresholds[1] = 0.5
        thresholds[2] = 0.85

        labels = dino_result_to_detection_labels(
            result,
            orig_height=100,
            orig_width=100,
            score_threshold=0.0,
            max_detections=100,
            score_calibrator=calibrator,
            class_score_thresholds=tuple(thresholds),
        )

        self.assertEqual([label.class_id for label in labels], [2])
        self.assertAlmostEqual(labels[0].confidence or 0.0, 0.9)

    def test_classwise_calibrator_adjusts_scores_before_topk_sorting(self) -> None:
        result = {
            "scores": torch.tensor([0.8, 0.7, 0.6]),
            "labels": torch.tensor([1, 2, 1]),
            "boxes": torch.tensor(
                [
                    [0.0, 0.0, 10.0, 10.0],
                    [10.0, 10.0, 20.0, 20.0],
                    [20.0, 20.0, 30.0, 30.0],
                ]
            ),
        }
        calibrator = ClasswiseScoreCalibrator.from_mapping({"1": {"scale": 0.5}, "2": {"bias": 0.2}})

        labels = dino_result_to_detection_labels(
            result,
            orig_height=100,
            orig_width=100,
            score_threshold=0.0,
            max_detections=2,
            score_calibrator=calibrator,
        )

        self.assertEqual([label.class_id for label in labels], [2, 1])
        self.assertAlmostEqual(labels[0].confidence or 0.0, 0.9)
        self.assertAlmostEqual(labels[1].confidence or 0.0, 0.4)


if __name__ == "__main__":
    unittest.main()
