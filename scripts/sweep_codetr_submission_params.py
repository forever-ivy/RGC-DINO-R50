#!/usr/bin/env python
"""Sweep Co-DETR outputs under the exact competition TXT submission contract.

MMDetection's native COCO evaluation can score raw Co-DETR outputs before the
repository's final TXT conversion and top-K truncation.  This script closes that
loop: it converts a ``results.pkl``/bbox JSON to competition TXT files for each
threshold/top-K setting, reloads those TXT files, and evaluates them with the
same YOLO-style metric helper used by submission candidates.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from codetr_results_to_submission import (  # noqa: E402
    _load_coco_images,
    _predictions_from_bbox_json,
    _predictions_from_results_pkl,
)
from rgc_dino.constants import MAX_PREDICTIONS_PER_IMAGE  # noqa: E402
from rgc_dino.labels import DetectionLabel, load_label_dir, load_label_file  # noqa: E402
from rgc_dino.metrics import evaluate_detection_map  # noqa: E402
from rgc_dino.sample_ids import load_sample_ids_file, restrict_mapping_to_sample_ids  # noqa: E402
from rgc_dino.submission import write_submission_files  # noqa: E402


@dataclass(frozen=True)
class SweepResult:
    score_threshold: float
    max_detections: int
    map_50_95: float
    map_50: float
    ground_truth_objects: int
    raw_prediction_objects: int
    written_prediction_objects: int
    mean_predictions_per_image: float
    non_empty_files: int
    prediction_dir: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--results-pkl", type=Path, help="MMDetection results.pkl from tools/test.py")
    source.add_argument("--bbox-json", type=Path, help="COCO bbox JSON from format_results")
    parser.add_argument("--coco-ann", type=Path, required=True, help="COCO annotation used to create the outputs")
    parser.add_argument("--labels", type=Path, default=ROOT / "source" / "训练集" / "labels")
    parser.add_argument("--sample-ids-file", type=Path, help="optional newline-delimited sample IDs to evaluate")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2],
        help="score thresholds to try before writing TXT files",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        nargs="+",
        default=[50, 80, MAX_PREDICTIONS_PER_IMAGE],
        help="per-image caps to try after confidence sorting",
    )
    parser.add_argument("--clip-labels", action="store_true", default=True)
    parser.add_argument("--no-clip-labels", dest="clip_labels", action="store_false")
    parser.add_argument("--top-k", type=int, default=10, help="number of ranked rows to print")
    parser.add_argument("--keep-predictions", action="store_true", default=True)
    parser.add_argument("--no-keep-predictions", dest="keep_predictions", action="store_false")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    images, id_to_image = _load_coco_images(args.coco_ann)
    coco_ids = [Path(str(image["file_name"])).stem for image in images]
    if args.sample_ids_file is not None:
        requested_ids = load_sample_ids_file(args.sample_ids_file)
        requested_set = set(requested_ids)
        missing_ids = sorted(requested_set - set(coco_ids))
        if missing_ids:
            print(
                f"sample IDs are not present in --coco-ann images; first missing IDs: {missing_ids[:10]}",
                file=sys.stderr,
            )
            print(
                "Use a validation COCO/results pair for val sweeps; test results cannot be scored against fold val IDs.",
                file=sys.stderr,
            )
            return 2
        image_ids = [sample_id for sample_id in coco_ids if sample_id in requested_set]
    else:
        requested_ids = coco_ids
        image_ids = coco_ids

    if not image_ids:
        print("no COCO images left after applying --sample-ids-file", file=sys.stderr)
        return 2

    ground_truths = load_label_dir(args.labels, clip=args.clip_labels)
    ground_truths = restrict_mapping_to_sample_ids(ground_truths, image_ids)
    if not ground_truths:
        print("no ground-truth labels available for requested COCO images", file=sys.stderr)
        return 2

    raw_predictions_by_threshold: dict[float, dict[str, list[DetectionLabel]]] = {}
    results: list[SweepResult] = []
    for threshold in args.thresholds:
        predictions = _load_predictions(args, images, id_to_image, threshold)
        if args.sample_ids_file is not None:
            predictions = {key: value for key, value in predictions.items() if key in set(requested_ids)}
        raw_predictions_by_threshold[threshold] = predictions
        raw_prediction_objects = sum(len(records) for records in predictions.values())

        for max_det in args.max_detections:
            tag = _tag(threshold, max_det)
            pred_dir = args.output_dir / tag
            write_submission_files(image_ids, predictions, pred_dir, max_predictions_per_image=max_det)
            reloaded = _load_written_predictions(image_ids, pred_dir)
            metric = evaluate_detection_map(ground_truths, reloaded)
            written_prediction_objects = metric.prediction_count
            non_empty = sum(1 for records in reloaded.values() if records)
            results.append(
                SweepResult(
                    score_threshold=threshold,
                    max_detections=max_det,
                    map_50_95=metric.map,
                    map_50=metric.map50,
                    ground_truth_objects=metric.ground_truth_count,
                    raw_prediction_objects=raw_prediction_objects,
                    written_prediction_objects=written_prediction_objects,
                    mean_predictions_per_image=written_prediction_objects / max(1, len(image_ids)),
                    non_empty_files=non_empty,
                    prediction_dir=str(pred_dir),
                )
            )
            if not args.keep_predictions:
                for path in pred_dir.glob("*.txt"):
                    path.unlink()
                pred_dir.rmdir()

    ranked = sorted(results, key=lambda item: item.map_50_95, reverse=True)
    ranking_path = args.output_dir / "submission_param_ranking.json"
    ranking_path.write_text(
        json.dumps([asdict(item) for item in ranked], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("=" * 96)
    print("Co-DETR final-TXT submission-contract sweep")
    print("=" * 96)
    print(f"images: {len(image_ids)}")
    print(f"ground_truth_objects: {sum(len(records) for records in ground_truths.values())}")
    print(f"ranking: {ranking_path}")
    print()
    print(f"{'rank':<5}{'mAP@50:95':<12}{'mAP@50':<10}{'thr':<10}{'topK':<7}{'raw':<9}{'written':<9}{'mean/img':<9}{'dir'}")
    for index, item in enumerate(ranked[: args.top_k], start=1):
        print(
            f"{index:<5}{item.map_50_95:<12.6f}{item.map_50:<10.6f}"
            f"{item.score_threshold:<10g}{item.max_detections:<7}"
            f"{item.raw_prediction_objects:<9}{item.written_prediction_objects:<9}"
            f"{item.mean_predictions_per_image:<9.2f}{item.prediction_dir}"
        )
    return 0


def _load_predictions(
    args: argparse.Namespace,
    images: list[dict[str, Any]],
    id_to_image: dict[int, dict[str, Any]],
    threshold: float,
) -> dict[str, list[DetectionLabel]]:
    if args.bbox_json is not None:
        return _predictions_from_bbox_json(
            args.bbox_json,
            id_to_image=id_to_image,
            score_threshold=threshold,
        )
    return _predictions_from_results_pkl(
        args.results_pkl,
        images=images,
        score_threshold=threshold,
    )


def _load_written_predictions(image_ids: list[str], pred_dir: Path) -> dict[str, list[DetectionLabel]]:
    predictions: dict[str, list[DetectionLabel]] = {}
    for sample_id in image_ids:
        path = pred_dir / f"{sample_id}.txt"
        predictions[sample_id] = load_label_file(path, require_confidence=True) if path.exists() else []
    return predictions


def _tag(threshold: float, max_det: int) -> str:
    threshold_tag = f"{threshold:.4f}".rstrip("0").rstrip(".").replace(".", "p")
    if not threshold_tag:
        threshold_tag = "0"
    return f"th{threshold_tag}_top{max_det}"


if __name__ == "__main__":
    raise SystemExit(main())
