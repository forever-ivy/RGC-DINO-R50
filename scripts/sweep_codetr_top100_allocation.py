#!/usr/bin/env python
"""Sweep class-aware legal top-100 allocation for Co-DETR validation outputs.

This script keeps the official per-image cap fixed at 100 by default.  It uses
existing Co-DETR ``results.pkl`` / bbox JSON / prediction cache outputs and
searches lightweight class-aware re-ranking policies that can move useful boxes
from the diagnostic top120/top150 tail into the legal top100.
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
    ClassAllocationConfig,
    apply_class_score_thresholds,
    apply_classwise_nms,
    cap_predictions_per_image,
    cap_predictions_per_image_class_aware,
    coerce_class_allocation_config,
    coerce_class_score_thresholds,
    score_histograms,
    summarize_predictions,
    topk_truncation_report,
)
from rgc_dino.sample_ids import load_sample_ids_file, restrict_mapping_to_sample_ids  # noqa: E402
from rgc_dino.submission import write_submission_files  # noqa: E402


@dataclass(frozen=True)
class AllocationSweepRow:
    name: str
    mode: str
    class_id: int | None
    class_name: str | None
    weight: float | None
    config: dict[str, Any]
    map_50_95: float
    map_50: float
    map_75: float
    map_90: float
    hard_val_map_50_95: float | None
    prediction_objects: int
    mean_predictions_per_image: float
    prediction_summary: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--results-pkl", type=Path, help="MMDetection results.pkl from tools/test.py")
    source.add_argument("--bbox-json", type=Path, help="COCO bbox JSON from format_results")
    source.add_argument("--prediction-cache", type=Path, help="JSON cache from scripts/cache_codetr_predictions.py")
    parser.add_argument("--coco-ann", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=ROOT / "source" / "训练集" / "labels")
    parser.add_argument("--sample-ids-file", type=Path, help="optional newline-delimited sample IDs to evaluate")
    parser.add_argument("--hard-val-sample-ids-file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-threshold", type=float, default=0.0)
    parser.add_argument("--initial-thresholds-json", type=Path, help="JSON class score thresholds to apply before allocation")
    parser.add_argument("--nms-iou-threshold", type=float)
    parser.add_argument("--max-detections", type=int, default=MAX_PREDICTIONS_PER_IMAGE)
    parser.add_argument(
        "--mode",
        choices=("config-eval", "single-class-weight", "greedy-weight"),
        default="single-class-weight",
    )
    parser.add_argument("--class-allocation-config", type=Path, help="evaluate one allocation JSON config")
    parser.add_argument("--target-classes", type=int, nargs="+", default=[0, 1, 2, 4, 5, 7, 8, 11])
    parser.add_argument("--weights", type=float, nargs="+", default=[0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3])
    parser.add_argument("--greedy-rounds", type=int, default=2)
    parser.add_argument("--clip-labels", action="store_true", default=True)
    parser.add_argument("--no-clip-labels", dest="clip_labels", action="store_false")
    parser.add_argument("--write-best-predictions", action="store_true", default=True)
    parser.add_argument("--no-write-best-predictions", dest="write_best_predictions", action="store_false")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_detections > MAX_PREDICTIONS_PER_IMAGE:
        print(
            f"refusing max_detections={args.max_detections}; official cap is {MAX_PREDICTIONS_PER_IMAGE}",
            file=sys.stderr,
        )
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)

    images, id_to_image = _load_coco_images(args.coco_ann)
    coco_ids = [Path(str(image["file_name"])).stem for image in images]
    if args.sample_ids_file is not None:
        requested_ids = load_sample_ids_file(args.sample_ids_file)
        requested_set = set(requested_ids)
        missing_ids = sorted(requested_set - set(coco_ids))
        if missing_ids:
            print(f"sample IDs missing from COCO ann: {missing_ids[:10]}", file=sys.stderr)
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
            print(f"hard-val IDs missing from evaluated IDs: {missing_hard_ids[:10]}", file=sys.stderr)
            return 2

    ground_truths = load_label_dir(args.labels, clip=args.clip_labels)
    ground_truths = restrict_mapping_to_sample_ids(ground_truths, image_ids)
    if not ground_truths:
        print("no ground-truth labels available for requested COCO images", file=sys.stderr)
        return 2

    raw_predictions = _load_predictions(args, images, id_to_image, args.candidate_threshold)
    raw_predictions = {key: value for key, value in raw_predictions.items() if key in requested_set}

    if args.initial_thresholds_json is not None:
        thresholds = list(coerce_class_score_thresholds(json.loads(args.initial_thresholds_json.read_text(encoding="utf-8"))))
    else:
        thresholds = [0.0] * NUM_CLASSES

    pre_allocation = apply_class_score_thresholds(raw_predictions, thresholds)
    pre_allocation = apply_classwise_nms(pre_allocation, iou_threshold=args.nms_iou_threshold)

    rows = _run_sweep(args, ground_truths, pre_allocation, image_ids, hard_val_ids)
    rows = sorted(rows, key=lambda item: item.map_50_95, reverse=True)
    best = rows[0]
    best_config = coerce_class_allocation_config(best.config)
    best_predictions = cap_predictions_per_image_class_aware(
        pre_allocation,
        max_detections=args.max_detections,
        allocation=best_config,
    )
    best_metric = evaluate_detection_map(ground_truths, best_predictions)
    best_summary = map_summary(best_metric)
    hard_val_report = _hard_val_report(
        hard_val_ids,
        all_image_ids=image_ids,
        ground_truths=ground_truths,
        before_allocation=pre_allocation,
        predictions=best_predictions,
        max_detections=args.max_detections,
    )
    topk_report = topk_truncation_report(
        pre_allocation,
        best_predictions,
        image_ids=image_ids,
        max_detections=args.max_detections,
    )
    prediction_summary = summarize_predictions(best_predictions, image_ids=image_ids)
    raw_summary = summarize_predictions(raw_predictions, image_ids=image_ids)
    pre_summary = summarize_predictions(pre_allocation, image_ids=image_ids)

    best_prediction_dir: Path | None = None
    if args.write_best_predictions:
        best_prediction_dir = args.output_dir / "best_predictions"
        write_submission_files(image_ids, best_predictions, best_prediction_dir, max_predictions_per_image=args.max_detections)

    payload = {
        "mode": args.mode,
        "candidate_threshold": args.candidate_threshold,
        "nms_iou_threshold": args.nms_iou_threshold,
        "max_detections": args.max_detections,
        "target_classes": args.target_classes,
        "weights": args.weights,
        "best": asdict(best),
        "best_allocation_config": best.config,
        "metric_summary": best_summary,
        "raw_prediction_summary": raw_summary,
        "pre_allocation_summary": pre_summary,
        "best_prediction_summary": prediction_summary,
        "topk_truncation_report": topk_report,
        "hard_val": hard_val_report,
        "prediction_dir": str(best_prediction_dir) if best_prediction_dir is not None else None,
    }
    (args.output_dir / "allocation_sweep_summary.json").write_text(
        json.dumps([asdict(row) for row in rows], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "best_allocation_config.json").write_text(
        json.dumps(best.config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "prediction_diagnostics.json").write_text(
        json.dumps(
            {
                "metric_summary": best_summary,
                "score_histograms": score_histograms(best_predictions),
                "raw_prediction_summary": raw_summary,
                "pre_allocation_summary": pre_summary,
                "best_prediction_summary": prediction_summary,
                "topk_truncation_report": topk_report,
                "hard_val": hard_val_report,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "per_class_ap.json").write_text(
        json.dumps(best_summary["per_class_ap"], indent=2, ensure_ascii=False) + "\n",
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
    _write_report(args.output_dir / "allocation_sweep_summary.md", payload, rows)

    print("=" * 96)
    print("Co-DETR class-aware legal top100 allocation sweep")
    print("=" * 96)
    print(f"images: {len(image_ids)}")
    print(f"mode: {args.mode}")
    print(f"mAP@50:95: {best.map_50_95:.9f}")
    print(f"mAP@50: {best.map_50:.9f}")
    print(f"hard-val mAP@50:95: {best.hard_val_map_50_95}")
    print(f"best: {best.name}")
    print(f"config: {best.config}")
    print(f"report: {args.output_dir / 'allocation_sweep_summary.md'}")
    return 0


def _run_sweep(
    args: argparse.Namespace,
    ground_truths: Mapping[str, Sequence[DetectionLabel]],
    pre_allocation: Mapping[str, Sequence[DetectionLabel]],
    image_ids: Sequence[str],
    hard_val_ids: Sequence[str],
) -> list[AllocationSweepRow]:
    rows: list[AllocationSweepRow] = []

    def evaluate(name: str, mode: str, class_id: int | None, weight: float | None, config: dict[str, Any]) -> AllocationSweepRow:
        allocation = coerce_class_allocation_config(config)
        predictions = cap_predictions_per_image_class_aware(
            pre_allocation,
            max_detections=args.max_detections,
            allocation=allocation,
        )
        metric = evaluate_detection_map(ground_truths, predictions)
        metric_summary = map_summary(metric)
        prediction_summary = summarize_predictions(predictions, image_ids=image_ids)
        hard_map = _hard_val_map(hard_val_ids, ground_truths=ground_truths, predictions=predictions)
        return AllocationSweepRow(
            name=name,
            mode=mode,
            class_id=class_id,
            class_name=CLASS_NAMES[class_id] if class_id is not None else None,
            weight=weight,
            config=config,
            map_50_95=metric.map,
            map_50=metric.map50,
            map_75=float(metric_summary["map_75"]),
            map_90=float(metric_summary["map_90"]),
            hard_val_map_50_95=hard_map,
            prediction_objects=metric.prediction_count,
            mean_predictions_per_image=prediction_summary["mean_predictions_per_image"],
            prediction_summary=prediction_summary,
        )

    rows.append(evaluate("baseline", "baseline", None, None, {}))

    if args.mode == "config-eval":
        if args.class_allocation_config is None:
            raise ValueError("--class-allocation-config is required for config-eval")
        config = json.loads(args.class_allocation_config.read_text(encoding="utf-8"))
        rows.append(evaluate("config_eval", "config-eval", None, None, config))
        return rows

    if args.mode == "single-class-weight":
        for class_id in args.target_classes:
            _validate_class_id(class_id)
            for weight in args.weights:
                weights = [1.0] * NUM_CLASSES
                weights[class_id] = float(weight)
                config = {"score_weights": weights}
                rows.append(
                    evaluate(
                        f"class{class_id}_{CLASS_NAMES[class_id]}_weight{weight:g}",
                        "single-class-weight",
                        class_id,
                        float(weight),
                        config,
                    )
                )
        return rows

    if args.mode == "greedy-weight":
        weights = [1.0] * NUM_CLASSES
        best = evaluate("greedy_initial", "greedy-weight", None, None, {"score_weights": weights})
        rows.append(best)
        for round_index in range(args.greedy_rounds):
            improved = False
            for class_id in args.target_classes:
                _validate_class_id(class_id)
                best_for_class = best
                best_weight = weights[class_id]
                for weight in args.weights:
                    candidate_weights = list(weights)
                    candidate_weights[class_id] = float(weight)
                    row = evaluate(
                        f"greedy_r{round_index + 1}_class{class_id}_{CLASS_NAMES[class_id]}_weight{weight:g}",
                        "greedy-weight",
                        class_id,
                        float(weight),
                        {"score_weights": candidate_weights},
                    )
                    rows.append(row)
                    if row.map_50_95 > best_for_class.map_50_95 + 1e-12:
                        best_for_class = row
                        best_weight = float(weight)
                if best_for_class.map_50_95 > best.map_50_95 + 1e-12:
                    weights[class_id] = best_weight
                    best = evaluate(
                        f"greedy_accept_r{round_index + 1}_class{class_id}_{CLASS_NAMES[class_id]}_weight{best_weight:g}",
                        "greedy-weight",
                        class_id,
                        best_weight,
                        {"score_weights": list(weights)},
                    )
                    rows.append(best)
                    improved = True
            if not improved:
                break
        return rows

    raise ValueError(f"unsupported mode: {args.mode}")


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
        predictions.setdefault(str(row["sample_id"]), []).append(
            DetectionLabel(
                class_id=int(row["class_id"]),
                norm_center_x=float(row["norm_center_x"]),
                norm_center_y=float(row["norm_center_y"]),
                norm_w=float(row["norm_w"]),
                norm_h=float(row["norm_h"]),
                confidence=score,
            )
        )
    return predictions


def _hard_val_map(
    hard_val_ids: Sequence[str],
    *,
    ground_truths: Mapping[str, Sequence[DetectionLabel]],
    predictions: Mapping[str, Sequence[DetectionLabel]],
) -> float | None:
    if not hard_val_ids:
        return None
    hard_gt = restrict_mapping_to_sample_ids(ground_truths, hard_val_ids)
    if not hard_gt:
        return None
    hard_predictions = {sample_id: list(predictions.get(sample_id, [])) for sample_id in hard_val_ids}
    return evaluate_detection_map(hard_gt, hard_predictions).map


def _hard_val_report(
    hard_val_ids: Sequence[str],
    *,
    all_image_ids: Sequence[str],
    ground_truths: Mapping[str, Sequence[DetectionLabel]],
    before_allocation: Mapping[str, Sequence[DetectionLabel]],
    predictions: Mapping[str, Sequence[DetectionLabel]],
    max_detections: int,
) -> dict[str, Any] | None:
    if not hard_val_ids:
        return None
    hard_gt = restrict_mapping_to_sample_ids(ground_truths, hard_val_ids)
    if not hard_gt:
        return {"sample_ids": list(hard_val_ids), "error": "no hard-val ground truths"}
    hard_predictions = {sample_id: list(predictions.get(sample_id, [])) for sample_id in hard_val_ids}
    hard_metric = evaluate_detection_map(hard_gt, hard_predictions)
    remainder_ids = [sample_id for sample_id in all_image_ids if sample_id not in set(hard_val_ids)]
    report: dict[str, Any] = {
        "sample_ids": list(hard_val_ids),
        "image_count": len(hard_val_ids),
        "after_topk": {
            "metric_summary": map_summary(hard_metric),
            "prediction_summary": summarize_predictions(hard_predictions, image_ids=hard_val_ids),
        },
        "topk_truncation_report": topk_truncation_report(
            {sample_id: list(before_allocation.get(sample_id, [])) for sample_id in hard_val_ids},
            hard_predictions,
            image_ids=hard_val_ids,
            max_detections=max_detections,
        ),
    }
    remainder_gt = restrict_mapping_to_sample_ids(ground_truths, remainder_ids)
    if remainder_gt:
        remainder_predictions = {sample_id: list(predictions.get(sample_id, [])) for sample_id in remainder_ids}
        remainder_metric = evaluate_detection_map(remainder_gt, remainder_predictions)
        report["remainder_after_topk"] = {
            "image_count": len(remainder_ids),
            "metric_summary": map_summary(remainder_metric),
            "prediction_summary": summarize_predictions(remainder_predictions, image_ids=remainder_ids),
        }
    return report


def _write_report(path: Path, payload: dict[str, Any], rows: Sequence[AllocationSweepRow]) -> None:
    best = payload["best"]
    lines = [
        "# Co-DETR class-aware legal top100 allocation sweep",
        "",
        f"- mode: `{payload['mode']}`",
        f"- candidate threshold: `{payload['candidate_threshold']}`",
        f"- NMS IoU: `{payload['nms_iou_threshold']}`",
        f"- max detections: `{payload['max_detections']}`",
        f"- best: `{best['name']}`",
        f"- mAP@50:95: `{best['map_50_95']:.9f}`",
        f"- mAP@50: `{best['map_50']:.9f}`",
        f"- mAP@75: `{best['map_75']:.9f}`",
        f"- mAP@90: `{best['map_90']:.9f}`",
        f"- hard-val mAP@50:95: `{best['hard_val_map_50_95']}`",
        f"- prediction objects: `{best['prediction_objects']}`",
        "",
        "## Top rows",
        "",
        "| rank | name | class | weight | mAP@50:95 | hard-val | mAP50 | mAP75 | mAP90 | preds | mean/img |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(rows[:30], start=1):
        class_label = "" if row.class_id is None else f"{row.class_id}:{row.class_name}"
        weight = "" if row.weight is None else f"{row.weight:.6g}"
        hard = "" if row.hard_val_map_50_95 is None else f"{row.hard_val_map_50_95:.9f}"
        lines.append(
            f"| {index} | {row.name} | {class_label} | {weight} | {row.map_50_95:.9f} | "
            f"{hard} | {row.map_50:.9f} | {row.map_75:.9f} | {row.map_90:.9f} | "
            f"{row.prediction_objects} | {row.mean_predictions_per_image:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Output files",
            "",
            "- `allocation_sweep_summary.json`",
            "- `best_allocation_config.json`",
            "- `prediction_diagnostics.json`",
            "- `per_class_ap.json`",
            "- `topk_truncation_report.json`",
            "- `hard_val_report.json` (when hard-val IDs are supplied)",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _validate_class_id(class_id: int) -> None:
    if not 0 <= class_id < NUM_CLASSES:
        raise ValueError(f"class ID {class_id} outside [0, {NUM_CLASSES - 1}]")


if __name__ == "__main__":
    raise SystemExit(main())
