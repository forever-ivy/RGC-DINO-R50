#!/usr/bin/env python
"""Greedy per-class threshold sweep for Co-DETR final-TXT validation outputs.

This script consumes an existing MMDetection ``results.pkl`` or COCO bbox JSON,
keeps a low global candidate threshold, and searches class-wise confidence
thresholds under the repository's final TXT submission contract.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from codetr_results_to_submission import (  # noqa: E402
    _load_coco_images,
    _predictions_from_bbox_json,
    _predictions_from_results_pkl,
)
from rgc_dino.constants import CLASS_NAMES, MAX_PREDICTIONS_PER_IMAGE, NUM_CLASSES  # noqa: E402
from rgc_dino.labels import DetectionLabel, load_label_dir  # noqa: E402
from rgc_dino.metrics import evaluate_detection_map, map_summary  # noqa: E402
from rgc_dino.postprocess import (  # noqa: E402
    apply_class_score_thresholds,
    apply_classwise_nms,
    cap_predictions_per_image,
    coerce_class_score_thresholds,
    score_histograms,
    summarize_predictions,
    topk_truncation_report,
)
from rgc_dino.sample_ids import load_sample_ids_file, restrict_mapping_to_sample_ids  # noqa: E402
from rgc_dino.submission import write_submission_files  # noqa: E402


@dataclass(frozen=True)
class ClassThresholdSweepRow:
    stage: str
    class_id: int | None
    class_name: str | None
    threshold: float | None
    map_50_95: float
    map_50: float
    map_75: float
    map_90: float
    objective: float
    prediction_objects: int
    mean_predictions_per_image: float
    thresholds: list[float]
    prediction_summary: dict[str, Any]
    prediction_dir: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--results-pkl", type=Path, help="MMDetection results.pkl from tools/test.py")
    source.add_argument("--bbox-json", type=Path, help="COCO bbox JSON from format_results")
    source.add_argument("--prediction-cache", type=Path, help="JSON cache from scripts/cache_codetr_predictions.py")
    parser.add_argument("--coco-ann", type=Path, required=True, help="COCO annotation used to create the outputs")
    parser.add_argument("--labels", type=Path, default=ROOT / "source" / "训练集" / "labels")
    parser.add_argument("--sample-ids-file", type=Path, help="optional newline-delimited sample IDs to evaluate")
    parser.add_argument("--hard-val-sample-ids-file", type=Path, help="optional hard-val sample IDs to score alongside normal val")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-threshold", type=float, default=0.0, help="global low candidate score threshold")
    parser.add_argument("--init-threshold", type=float, default=0.0, help="initial threshold for every class")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.0, 0.0005, 0.001, 0.0015, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05],
        help="per-class threshold candidates",
    )
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--target-classes", type=int, nargs="+", help="optional class IDs to tune; defaults to all classes")
    parser.add_argument("--pre-limit-per-image", type=int, help="optional fast-mode cap applied to raw candidates before sweeping")
    parser.add_argument("--nms-iou-threshold", type=float, help="optional classwise NMS IoU before top-k")
    parser.add_argument("--max-detections", type=int, default=MAX_PREDICTIONS_PER_IMAGE)
    parser.add_argument("--box-penalty", type=float, default=0.0, help="subtract penalty * boxes_per_image from objective")
    parser.add_argument("--initial-thresholds-json", type=Path, help="optional JSON initial class thresholds")
    parser.add_argument("--clip-labels", action="store_true", default=True)
    parser.add_argument("--no-clip-labels", dest="clip_labels", action="store_false")
    parser.add_argument("--write-best-predictions", action="store_true", default=True)
    parser.add_argument("--no-write-best-predictions", dest="write_best_predictions", action="store_false")
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
            return 2
        image_ids = [sample_id for sample_id in coco_ids if sample_id in requested_set]
    else:
        requested_set = set(coco_ids)
        image_ids = coco_ids

    hard_val_ids: list[str] = []
    if args.hard_val_sample_ids_file is not None:
        hard_val_ids = load_sample_ids_file(args.hard_val_sample_ids_file)
        missing_hard_ids = sorted(set(hard_val_ids) - set(image_ids))
        if missing_hard_ids:
            print(
                f"hard-val sample IDs are not in the evaluated image IDs; first missing IDs: {missing_hard_ids[:10]}",
                file=sys.stderr,
            )
            return 2

    ground_truths = load_label_dir(args.labels, clip=args.clip_labels)
    ground_truths = restrict_mapping_to_sample_ids(ground_truths, image_ids)
    if not ground_truths:
        print("no ground-truth labels available for requested COCO images", file=sys.stderr)
        return 2

    raw_predictions = _load_predictions(args, images, id_to_image, args.candidate_threshold)
    raw_predictions = {key: value for key, value in raw_predictions.items() if key in requested_set}
    if args.pre_limit_per_image is not None:
        raw_predictions = cap_predictions_per_image(raw_predictions, max_detections=args.pre_limit_per_image)
    raw_summary = summarize_predictions(raw_predictions, image_ids=image_ids)

    if args.initial_thresholds_json is not None:
        thresholds = list(coerce_class_score_thresholds(json.loads(args.initial_thresholds_json.read_text(encoding="utf-8"))))
    else:
        thresholds = [float(args.init_threshold)] * NUM_CLASSES
    candidate_values = sorted(set(float(value) for value in args.thresholds))

    rows: list[ClassThresholdSweepRow] = []

    def evaluate(stage: str, class_id: int | None, threshold: float | None, candidate_thresholds: list[float]) -> tuple[float, float, dict[str, Any]]:
        stages = _postprocess_stages(
            raw_predictions,
            thresholds=candidate_thresholds,
            nms_iou_threshold=args.nms_iou_threshold,
            max_detections=args.max_detections,
        )
        predictions = stages["after_topk"]
        metric = evaluate_detection_map(ground_truths, predictions)
        metric_summary = map_summary(metric)
        summary = summarize_predictions(predictions, image_ids=image_ids)
        boxes_per_image = summary["mean_predictions_per_image"]
        obj = metric.map - args.box_penalty * boxes_per_image
        rows.append(
            ClassThresholdSweepRow(
                stage=stage,
                class_id=class_id,
                class_name=CLASS_NAMES[class_id] if class_id is not None else None,
                threshold=threshold,
                map_50_95=metric.map,
                map_50=metric.map50,
                map_75=float(metric_summary["map_75"]),
                map_90=float(metric_summary["map_90"]),
                objective=obj,
                prediction_objects=metric.prediction_count,
                mean_predictions_per_image=boxes_per_image,
                thresholds=list(candidate_thresholds),
                prediction_summary=summary,
            )
        )
        return metric.map, obj, summary

    best_map, best_obj, _summary = evaluate("initial", None, None, thresholds)
    target_classes = args.target_classes if args.target_classes is not None else list(range(NUM_CLASSES))
    for class_id in target_classes:
        if not 0 <= class_id < NUM_CLASSES:
            raise ValueError(f"target class {class_id} outside [0, {NUM_CLASSES - 1}]")

    for round_index in range(args.rounds):
        improved = False
        for class_id in target_classes:
            best_for_class = thresholds[class_id]
            best_for_class_map = best_map
            best_for_class_obj = best_obj
            for candidate in candidate_values:
                candidate_thresholds = list(thresholds)
                candidate_thresholds[class_id] = candidate
                map_value, obj, _ = evaluate(
                    f"round_{round_index + 1}",
                    class_id,
                    candidate,
                    candidate_thresholds,
                )
                if obj > best_for_class_obj + 1e-12:
                    best_for_class = candidate
                    best_for_class_map = map_value
                    best_for_class_obj = obj
            if best_for_class != thresholds[class_id]:
                thresholds[class_id] = best_for_class
                best_map = best_for_class_map
                best_obj = best_for_class_obj
                improved = True
                evaluate(f"accepted_round_{round_index + 1}", class_id, best_for_class, thresholds)
        if not improved:
            break

    best_stages = _postprocess_stages(
        raw_predictions,
        thresholds=thresholds,
        nms_iou_threshold=args.nms_iou_threshold,
        max_detections=args.max_detections,
    )
    best_predictions = best_stages["after_topk"]
    best_metric = evaluate_detection_map(ground_truths, best_predictions)
    best_metric_summary = map_summary(best_metric)
    stage_summaries = {
        name: summarize_predictions(stage_predictions, image_ids=image_ids)
        for name, stage_predictions in best_stages.items()
    }
    best_summary = stage_summaries["after_topk"]
    topk_report = topk_truncation_report(
        best_stages["after_nms"],
        best_predictions,
        image_ids=image_ids,
        max_detections=args.max_detections,
    )
    hard_val_report = _hard_val_report(
        hard_val_ids,
        all_image_ids=image_ids,
        ground_truths=ground_truths,
        stages=best_stages,
        max_detections=args.max_detections,
    )

    best_dir: Path | None = None
    if args.write_best_predictions:
        best_dir = args.output_dir / "best_predictions"
        write_submission_files(image_ids, best_predictions, best_dir, max_predictions_per_image=args.max_detections)
        rows.append(
            ClassThresholdSweepRow(
                stage="best_written",
                class_id=None,
                class_name=None,
                threshold=None,
                map_50_95=best_metric.map,
                map_50=best_metric.map50,
                map_75=float(best_metric_summary["map_75"]),
                map_90=float(best_metric_summary["map_90"]),
                objective=best_metric.map - args.box_penalty * best_summary["mean_predictions_per_image"],
                prediction_objects=best_metric.prediction_count,
                mean_predictions_per_image=best_summary["mean_predictions_per_image"],
                thresholds=list(thresholds),
                prediction_summary=best_summary,
                prediction_dir=str(best_dir),
            )
        )

    diagnostics = {
        "metric_summary": best_metric_summary,
        "stage_summaries": stage_summaries,
        "score_histograms": score_histograms(best_predictions),
        "topk_truncation_report": topk_report,
        "hard_val": hard_val_report,
    }
    result_payload = {
        "candidate_threshold": args.candidate_threshold,
        "nms_iou_threshold": args.nms_iou_threshold,
        "max_detections": args.max_detections,
        "pre_limit_per_image": args.pre_limit_per_image,
        "target_classes": target_classes,
        "box_penalty": args.box_penalty,
        "thresholds": thresholds,
        "thresholds_by_class": {str(index): {"name": CLASS_NAMES[index], "threshold": thresholds[index]} for index in range(NUM_CLASSES)},
        "map_50_95": best_metric.map,
        "map_50": best_metric.map50,
        "map_75": best_metric_summary["map_75"],
        "map_90": best_metric_summary["map_90"],
        "raw_prediction_summary": raw_summary,
        "stage_summaries": stage_summaries,
        "best_prediction_summary": best_summary,
        "per_class_ap": best_metric_summary["per_class_ap"],
        "topk_truncation_report": topk_report,
        "hard_val": hard_val_report,
        "prediction_dir": str(best_dir) if best_dir is not None else None,
    }
    (args.output_dir / "class_threshold_sweep.json").write_text(
        json.dumps(result_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "class_threshold_sweep_rows.json").write_text(
        json.dumps([asdict(row) for row in rows], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "prediction_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "per_class_ap.json").write_text(
        json.dumps(best_metric_summary["per_class_ap"], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "topk_truncation_report.json").write_text(
        json.dumps(topk_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if hard_val_report is not None:
        (args.output_dir / "hard_val_report.json").write_text(
            json.dumps(hard_val_report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    _write_report(args.output_dir / "class_threshold_sweep_report.md", result_payload, rows)
    _write_diagnostics_report(args.output_dir / "prediction_diagnostics_report.md", result_payload, diagnostics)

    print("=" * 96)
    print("Co-DETR class-wise threshold sweep")
    print("=" * 96)
    print(f"images: {len(image_ids)}")
    print(f"candidate_threshold: {args.candidate_threshold:g}")
    print(f"nms_iou_threshold: {args.nms_iou_threshold}")
    print(f"max_detections: {args.max_detections}")
    print(f"mAP@50:95: {best_metric.map:.6f}")
    print(f"mAP@50: {best_metric.map50:.6f}")
    print(f"mAP@75: {float(best_metric_summary['map_75']):.6f}")
    print(f"mAP@90: {float(best_metric_summary['map_90']):.6f}")
    if hard_val_report is not None:
        print(f"hard-val mAP@50:95: {hard_val_report['after_topk']['metric_summary']['map_50_95']:.6f}")
    print(f"prediction_objects: {best_metric.prediction_count}")
    print(f"thresholds: {thresholds}")
    print(f"report: {args.output_dir / 'class_threshold_sweep_report.md'}")
    return 0


def _load_predictions(
    args: argparse.Namespace,
    images: list[dict[str, Any]],
    id_to_image: dict[int, dict[str, Any]],
    threshold: float,
) -> dict[str, list[DetectionLabel]]:
    if args.prediction_cache is not None:
        return _predictions_from_cache(args.prediction_cache, score_threshold=threshold)
    if args.bbox_json is not None:
        return _predictions_from_bbox_json(args.bbox_json, id_to_image=id_to_image, score_threshold=threshold)
    return _predictions_from_results_pkl(args.results_pkl, images=images, score_threshold=threshold)


def _predictions_from_cache(path: Path, *, score_threshold: float) -> dict[str, list[DetectionLabel]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("predictions", payload)
    predictions: dict[str, list[DetectionLabel]] = {}
    if isinstance(rows, dict):
        iterable = []
        for sample_id, records in rows.items():
            for record in records:
                item = dict(record)
                item.setdefault("sample_id", sample_id)
                iterable.append(item)
    else:
        iterable = rows
    for row in iterable:
        score = float(row["confidence"] if "confidence" in row else row["score"])
        if score < score_threshold:
            continue
        sample_id = str(row["sample_id"])
        label = DetectionLabel(
            class_id=int(row["class_id"]),
            norm_center_x=float(row["norm_center_x"]),
            norm_center_y=float(row["norm_center_y"]),
            norm_w=float(row["norm_w"]),
            norm_h=float(row["norm_h"]),
            confidence=score,
        )
        predictions.setdefault(sample_id, []).append(label)
    return predictions


def _postprocess_stages(
    predictions: Mapping[str, Sequence[DetectionLabel]],
    *,
    thresholds: Sequence[float],
    nms_iou_threshold: float | None,
    max_detections: int,
) -> dict[str, dict[str, list[DetectionLabel]]]:
    raw = {sample_id: list(records) for sample_id, records in predictions.items()}
    after_class_threshold = apply_class_score_thresholds(raw, thresholds)
    after_nms = apply_classwise_nms(after_class_threshold, iou_threshold=nms_iou_threshold)
    after_topk = cap_predictions_per_image(after_nms, max_detections=max_detections)
    return {
        "raw": raw,
        "after_class_threshold": after_class_threshold,
        "after_nms": after_nms,
        "after_topk": after_topk,
    }


def _finalize_predictions(
    predictions: dict[str, list[DetectionLabel]],
    *,
    thresholds: list[float],
    nms_iou_threshold: float | None,
    max_detections: int,
) -> dict[str, list[DetectionLabel]]:
    return _postprocess_stages(
        predictions,
        thresholds=thresholds,
        nms_iou_threshold=nms_iou_threshold,
        max_detections=max_detections,
    )["after_topk"]


def _subset_predictions(
    predictions: Mapping[str, Sequence[DetectionLabel]],
    sample_ids: Sequence[str],
) -> dict[str, list[DetectionLabel]]:
    return {sample_id: list(predictions.get(sample_id, [])) for sample_id in sample_ids}


def _hard_val_report(
    hard_val_ids: Sequence[str],
    *,
    all_image_ids: Sequence[str],
    ground_truths: Mapping[str, Sequence[DetectionLabel]],
    stages: Mapping[str, Mapping[str, Sequence[DetectionLabel]]],
    max_detections: int,
) -> dict[str, Any] | None:
    if not hard_val_ids:
        return None
    hard_gt = restrict_mapping_to_sample_ids(ground_truths, hard_val_ids)
    if not hard_gt:
        return {"sample_ids": list(hard_val_ids), "error": "no hard-val ground truths"}
    remainder_ids = [sample_id for sample_id in all_image_ids if sample_id not in set(hard_val_ids)]
    report: dict[str, Any] = {"sample_ids": list(hard_val_ids), "image_count": len(hard_val_ids)}
    for name, predictions in stages.items():
        hard_predictions = _subset_predictions(predictions, hard_val_ids)
        hard_metric = evaluate_detection_map(hard_gt, hard_predictions)
        report[name] = {
            "metric_summary": map_summary(hard_metric),
            "prediction_summary": summarize_predictions(hard_predictions, image_ids=hard_val_ids),
        }
    if remainder_ids:
        remainder_gt = restrict_mapping_to_sample_ids(ground_truths, remainder_ids)
        if remainder_gt:
            remainder_predictions = _subset_predictions(stages["after_topk"], remainder_ids)
            remainder_metric = evaluate_detection_map(remainder_gt, remainder_predictions)
            report["remainder_after_topk"] = {
                "image_count": len(remainder_ids),
                "metric_summary": map_summary(remainder_metric),
                "prediction_summary": summarize_predictions(remainder_predictions, image_ids=remainder_ids),
            }
    report["topk_truncation_report"] = topk_truncation_report(
        _subset_predictions(stages["after_nms"], hard_val_ids),
        _subset_predictions(stages["after_topk"], hard_val_ids),
        image_ids=hard_val_ids,
        max_detections=max_detections,
    )
    return report


def _write_report(path: Path, result: dict[str, Any], rows: list[ClassThresholdSweepRow]) -> None:
    hard_val = result.get("hard_val")
    lines = [
        "# Co-DETR class-wise threshold sweep",
        "",
        f"- mAP@50:95: `{result['map_50_95']:.6f}`",
        f"- mAP@50: `{result['map_50']:.6f}`",
        f"- mAP@75: `{float(result['map_75']):.6f}`",
        f"- mAP@90: `{float(result['map_90']):.6f}`",
        f"- candidate threshold: `{result['candidate_threshold']}`",
        f"- NMS IoU: `{result['nms_iou_threshold']}`",
        f"- max detections: `{result['max_detections']}`",
        f"- prediction objects: `{result['best_prediction_summary']['prediction_objects']}`",
        f"- mean predictions/image: `{result['best_prediction_summary']['mean_predictions_per_image']:.4f}`",
    ]
    if hard_val is not None and "after_topk" in hard_val:
        hard_metric = hard_val["after_topk"]["metric_summary"]
        lines.append(f"- hard-val mAP@50:95: `{hard_metric['map_50_95']:.6f}`")
    lines.extend([
        "",
        "## Thresholds by class",
        "",
        "| class_id | name | threshold | predictions | AP50:95 | AP50 | AP75 | AP90 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ])
    per_class_counts = result["best_prediction_summary"]["per_class_counts"]
    per_class_ap = {str(item["class_id"]): item for item in result["per_class_ap"]}
    for class_id, item in result["thresholds_by_class"].items():
        ap = per_class_ap.get(class_id, {})
        lines.append(
            f"| {class_id} | {item['name']} | {item['threshold']:.6g} | {per_class_counts.get(class_id, 0)} | "
            f"{float(ap.get('ap_50_95', 0.0)):.6f} | {float(ap.get('ap_50', 0.0)):.6f} | "
            f"{float(ap.get('ap_75', 0.0)):.6f} | {float(ap.get('ap_90', 0.0)):.6f} |"
        )
    lines.extend(["", "## Accepted steps", "", "| stage | class | threshold | mAP@50:95 | AP75 | AP90 | objective | predictions |", "|---|---|---:|---:|---:|---:|---:|---:|"])
    for row in rows:
        if not row.stage.startswith("accepted") and row.stage != "best_written":
            continue
        class_label = "" if row.class_id is None else f"{row.class_id}:{row.class_name}"
        threshold = "" if row.threshold is None else f"{row.threshold:.6g}"
        lines.append(
            f"| {row.stage} | {class_label} | {threshold} | {row.map_50_95:.6f} | {row.map_75:.6f} | {row.map_90:.6f} | {row.objective:.6f} | {row.prediction_objects} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_diagnostics_report(path: Path, result: dict[str, Any], diagnostics: dict[str, Any]) -> None:
    topk = diagnostics["topk_truncation_report"]
    stage_summaries = diagnostics["stage_summaries"]
    lines = [
        "# Prediction diagnostics",
        "",
        "## Stage summaries",
        "",
        "| stage | predictions | mean/img | max/img | non-empty |",
        "|---|---:|---:|---:|---:|",
    ]
    for stage in ("raw", "after_class_threshold", "after_nms", "after_topk"):
        summary = stage_summaries[stage]
        lines.append(
            f"| {stage} | {summary['prediction_objects']} | {summary['mean_predictions_per_image']:.4f} | {summary['max_predictions_per_image']} | {summary['non_empty_images']} |"
        )
    lines.extend(
        [
            "",
            "## TopK truncation",
            "",
            f"- max detections: `{topk['max_detections']}`",
            f"- saturated images: `{topk['saturated_image_count']}`",
            f"- dropped prediction objects: `{topk['dropped_prediction_objects']}`",
            "",
            "## Output files",
            "",
            "- `prediction_diagnostics.json`",
            "- `per_class_ap.json`",
            "- `topk_truncation_report.json`",
        ]
    )
    if result.get("hard_val") is not None:
        lines.append("- `hard_val_report.json`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
