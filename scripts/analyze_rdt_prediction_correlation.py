#!/usr/bin/env python
"""Correlate RGB-guided-RDT diagnostics with validation prediction behavior.

This is a lightweight analysis step: it does not run model inference or training.
It joins per-image RDT saliency stats with an existing Co-DETR prediction cache,
final-TXT predictions, and labels, then writes JSON/Markdown diagnostics.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rgc_dino.labels import DetectionLabel, load_label_dir, load_label_file  # noqa: E402
from rgc_dino.metrics import box_iou_xyxy, xywh_to_xyxy  # noqa: E402
from rgc_dino.sample_ids import load_sample_ids_file, restrict_mapping_to_sample_ids  # noqa: E402


DEFAULT_RDT_STATS = ROOT / "outputs" / "diagnostics" / "rdt_phase35_fold0_val" / "rdt_stats.json"
DEFAULT_PREDICTION_CACHE = (
    ROOT / "outputs" / "codetr" / "fasttrack_20260620_0940" / "classwise_phase35" / "prediction_cache.json"
)
DEFAULT_FINAL_PREDICTIONS = (
    ROOT / "outputs" / "codetr" / "fasttrack_20260620_0940" / "submission_contract_sweep" / "th0_top100"
)
DEFAULT_LABELS = ROOT / "source" / "训练集" / "labels"
DEFAULT_VAL_IDS = ROOT / "outputs" / "splits" / "fold0_val_ids.txt"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "diagnostics" / "rdt_phase35_fold0_val_prediction_correlation"

RDT_FEATURE_PREFIX = "rdt_"
TARGET_METRICS = (
    "gt_count",
    "raw_prediction_count",
    "raw_count_score_ge_001",
    "raw_count_score_ge_005",
    "final_prediction_count",
    "final_class_diversity",
    "top100_saturated",
    "tp_count",
    "fp_count",
    "fn_count",
    "precision_iou50",
    "recall_iou50",
)
FOCUS_FEATURES = (
    "rdt_depth_valid_ratio",
    "rdt_attention_hot_ratio",
    "rdt_attention_mean",
    "rdt_attention_std",
    "rdt_attention_top10_mean",
    "rdt_depth_attention_mean",
    "rdt_ir_attention_std",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rdt-stats", type=Path, default=DEFAULT_RDT_STATS)
    parser.add_argument("--prediction-cache", type=Path, default=DEFAULT_PREDICTION_CACHE)
    parser.add_argument("--final-predictions", type=Path, default=DEFAULT_FINAL_PREDICTIONS)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--sample-ids-file", type=Path, default=DEFAULT_VAL_IDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--clip-labels", action="store_true", default=True)
    parser.add_argument("--no-clip-labels", dest="clip_labels", action="store_false")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_ids = load_sample_ids_file(args.sample_ids_file)
    rdt_stats = _load_rdt_stats(args.rdt_stats)
    missing_stats = [sample_id for sample_id in image_ids if sample_id not in rdt_stats]
    if missing_stats:
        print(f"RDT stats missing sample IDs; first missing: {missing_stats[:10]}", file=sys.stderr)
        return 2

    raw_predictions = _load_prediction_cache(args.prediction_cache)
    final_predictions = _load_final_predictions(args.final_predictions, image_ids)
    ground_truths = load_label_dir(args.labels, clip=args.clip_labels)
    ground_truths = restrict_mapping_to_sample_ids(ground_truths, image_ids)

    rows = _build_joined_rows(
        image_ids=image_ids,
        rdt_stats=rdt_stats,
        raw_predictions=raw_predictions,
        final_predictions=final_predictions,
        ground_truths=ground_truths,
        iou_threshold=args.iou_threshold,
    )
    correlations = _correlations(rows)
    quartiles = _quartile_diagnostics(rows)
    feature_summary = _feature_summary(rows)
    extremes = _extreme_examples(rows)

    result = {
        "inputs": {
            "rdt_stats": str(args.rdt_stats),
            "prediction_cache": str(args.prediction_cache),
            "final_predictions": str(args.final_predictions),
            "labels": str(args.labels),
            "sample_ids_file": str(args.sample_ids_file),
            "iou_threshold": args.iou_threshold,
        },
        "summary": {
            "images": len(rows),
            "missing_raw_prediction_images": sum(1 for row in rows if row["raw_prediction_count"] == 0),
            "missing_final_prediction_images": sum(1 for row in rows if row["final_prediction_count"] == 0),
            "total_ground_truth_objects": int(sum(row["gt_count"] for row in rows)),
            "total_final_predictions": int(sum(row["final_prediction_count"] for row in rows)),
            "total_iou50_tp": int(sum(row["tp_count"] for row in rows)),
            "total_iou50_fp": int(sum(row["fp_count"] for row in rows)),
            "total_iou50_fn": int(sum(row["fn_count"] for row in rows)),
            "mean_precision_iou50": _mean([row["precision_iou50"] for row in rows]),
            "mean_recall_iou50": _mean([row["recall_iou50"] for row in rows]),
        },
        "rdt_feature_summary": feature_summary,
        "correlations": correlations,
        "quartiles": quartiles,
        "extreme_examples": extremes,
    }

    _write_json(args.output_dir / "joined_rows.json", rows)
    _write_json(args.output_dir / "rdt_prediction_correlation.json", result)
    _write_report(args.output_dir / "rdt_prediction_correlation_report.md", result)

    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    print(f"report: {args.output_dir / 'rdt_prediction_correlation_report.md'}")
    return 0


def _load_rdt_stats(path: Path) -> dict[str, dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("RDT stats must be a sample_id -> stats mapping")
    return {str(sample_id): {str(key): float(value) for key, value in stats.items()} for sample_id, stats in payload.items()}


def _load_prediction_cache(path: Path) -> dict[str, list[DetectionLabel]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("predictions", payload) if isinstance(payload, dict) else payload
    predictions: dict[str, list[DetectionLabel]] = {}
    if isinstance(rows, dict):
        iterable: list[dict[str, Any]] = []
        for sample_id, records in rows.items():
            for record in records:
                item = dict(record)
                item.setdefault("sample_id", sample_id)
                iterable.append(item)
    else:
        iterable = list(rows)
    for row in iterable:
        sample_id = str(row["sample_id"])
        predictions.setdefault(sample_id, []).append(_row_to_detection_label(row))
    return predictions


def _row_to_detection_label(row: Mapping[str, Any]) -> DetectionLabel:
    score = row["confidence"] if "confidence" in row else row["score"]
    return DetectionLabel(
        class_id=int(row["class_id"]),
        norm_center_x=float(row["norm_center_x"]),
        norm_center_y=float(row["norm_center_y"]),
        norm_w=float(row["norm_w"]),
        norm_h=float(row["norm_h"]),
        confidence=float(score),
    )


def _load_final_predictions(prediction_dir: Path, image_ids: Sequence[str]) -> dict[str, list[DetectionLabel]]:
    predictions: dict[str, list[DetectionLabel]] = {}
    for sample_id in image_ids:
        path = prediction_dir / f"{sample_id}.txt"
        predictions[sample_id] = load_label_file(path, require_confidence=True) if path.exists() else []
    return predictions


def _build_joined_rows(
    *,
    image_ids: Sequence[str],
    rdt_stats: Mapping[str, Mapping[str, float]],
    raw_predictions: Mapping[str, Sequence[DetectionLabel]],
    final_predictions: Mapping[str, Sequence[DetectionLabel]],
    ground_truths: Mapping[str, Sequence[DetectionLabel]],
    iou_threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_id in image_ids:
        raw_records = list(raw_predictions.get(sample_id, []))
        final_records = list(final_predictions.get(sample_id, []))
        gt_records = list(ground_truths.get(sample_id, []))
        match = _match_one_image(gt_records, final_records, iou_threshold=iou_threshold)
        row: dict[str, Any] = {"sample_id": sample_id}
        row.update({key: _json_float(value) for key, value in rdt_stats[sample_id].items()})
        row.update(_prediction_summary(raw_records, prefix="raw"))
        row.update(_prediction_summary(final_records, prefix="final"))
        row["gt_count"] = len(gt_records)
        row.update(match)
        row["top100_saturated"] = 1 if len(final_records) >= 100 else 0
        rows.append(row)
    return rows


def _prediction_summary(records: Sequence[DetectionLabel], *, prefix: str) -> dict[str, Any]:
    scores = np.asarray([record.confidence or 0.0 for record in records], dtype=np.float64)
    class_ids = {record.class_id for record in records}
    return {
        f"{prefix}_prediction_count": len(records),
        f"{prefix}_class_diversity": len(class_ids),
        f"{prefix}_score_mean": _json_float(float(np.mean(scores)) if scores.size else 0.0),
        f"{prefix}_score_max": _json_float(float(np.max(scores)) if scores.size else 0.0),
        f"{prefix}_score_p90": _json_float(float(np.quantile(scores, 0.9)) if scores.size else 0.0),
        f"{prefix}_count_score_ge_001": int(np.sum(scores >= 0.001)),
        f"{prefix}_count_score_ge_005": int(np.sum(scores >= 0.005)),
        f"{prefix}_count_score_ge_010": int(np.sum(scores >= 0.010)),
        f"{prefix}_count_score_ge_050": int(np.sum(scores >= 0.050)),
    }


def _match_one_image(
    ground_truths: Sequence[DetectionLabel],
    predictions: Sequence[DetectionLabel],
    *,
    iou_threshold: float,
) -> dict[str, Any]:
    matched: set[int] = set()
    tp = 0
    fp = 0
    gt_boxes = [xywh_to_xyxy(record) for record in ground_truths]
    pred_sorted = sorted(predictions, key=lambda item: item.confidence or 0.0, reverse=True)
    for prediction in pred_sorted:
        pred_box = xywh_to_xyxy(prediction)
        best_iou = 0.0
        best_index = -1
        for index, target in enumerate(ground_truths):
            if index in matched or target.class_id != prediction.class_id:
                continue
            iou = box_iou_xyxy(pred_box, gt_boxes[index])
            if iou > best_iou:
                best_iou = iou
                best_index = index
        if best_index >= 0 and best_iou >= iou_threshold:
            matched.add(best_index)
            tp += 1
        else:
            fp += 1
    fn = max(0, len(ground_truths) - len(matched))
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / len(ground_truths) if ground_truths else 0.0
    return {
        "tp_count": tp,
        "fp_count": fp,
        "fn_count": fn,
        "precision_iou50": _json_float(precision),
        "recall_iou50": _json_float(recall),
    }


def _correlations(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    rdt_features = [key for key in rows[0] if key.startswith(RDT_FEATURE_PREFIX)]
    results: list[dict[str, Any]] = []
    for feature in sorted(rdt_features):
        for target in TARGET_METRICS:
            if target not in rows[0]:
                continue
            xs = [float(row[feature]) for row in rows]
            ys = [float(row[target]) for row in rows]
            pearson = _pearson(xs, ys)
            spearman = _spearman(xs, ys)
            results.append(
                {
                    "feature": feature,
                    "target": target,
                    "pearson": _json_float(pearson),
                    "spearman": _json_float(spearman),
                    "abs_pearson": _json_float(abs(pearson)),
                    "abs_spearman": _json_float(abs(spearman)),
                }
            )
    results.sort(key=lambda item: max(item["abs_pearson"], item["abs_spearman"]), reverse=True)
    return results


def _quartile_diagnostics(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for feature in FOCUS_FEATURES:
        if not rows or feature not in rows[0]:
            continue
        ordered = sorted(rows, key=lambda row: float(row[feature]))
        bins: list[dict[str, Any]] = []
        for index, chunk in enumerate(_split_evenly(ordered, 4), start=1):
            if not chunk:
                continue
            bins.append(
                {
                    "bin": index,
                    "count": len(chunk),
                    "feature_min": _json_float(min(float(row[feature]) for row in chunk)),
                    "feature_max": _json_float(max(float(row[feature]) for row in chunk)),
                    "gt_count_mean": _mean(row["gt_count"] for row in chunk),
                    "final_prediction_count_mean": _mean(row["final_prediction_count"] for row in chunk),
                    "fp_count_mean": _mean(row["fp_count"] for row in chunk),
                    "tp_count_mean": _mean(row["tp_count"] for row in chunk),
                    "fn_count_mean": _mean(row["fn_count"] for row in chunk),
                    "precision_iou50_mean": _mean(row["precision_iou50"] for row in chunk),
                    "recall_iou50_mean": _mean(row["recall_iou50"] for row in chunk),
                    "top100_saturation_rate": _mean(row["top100_saturated"] for row in chunk),
                }
            )
        result[feature] = bins
    return result


def _feature_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    if not rows:
        return {}
    features = [key for key in rows[0] if key.startswith(RDT_FEATURE_PREFIX)]
    summary: dict[str, dict[str, float]] = {}
    for feature in sorted(features):
        values = np.asarray([float(row[feature]) for row in rows], dtype=np.float64)
        summary[feature] = {
            "min": _json_float(float(np.min(values))),
            "q25": _json_float(float(np.quantile(values, 0.25))),
            "median": _json_float(float(np.median(values))),
            "mean": _json_float(float(np.mean(values))),
            "q75": _json_float(float(np.quantile(values, 0.75))),
            "max": _json_float(float(np.max(values))),
        }
    return summary


def _extreme_examples(rows: Sequence[Mapping[str, Any]], *, limit: int = 12) -> dict[str, list[dict[str, Any]]]:
    fields = [
        "sample_id",
        "gt_count",
        "final_prediction_count",
        "tp_count",
        "fp_count",
        "fn_count",
        "precision_iou50",
        "recall_iou50",
        "rdt_depth_valid_ratio",
        "rdt_attention_hot_ratio",
        "rdt_attention_mean",
        "rdt_attention_std",
    ]

    def pick(sorted_rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [{field: row[field] for field in fields if field in row} for row in sorted_rows[:limit]]

    return {
        "highest_fp": pick(sorted(rows, key=lambda row: (row["fp_count"], row["gt_count"]), reverse=True)),
        "lowest_depth_valid_ratio": pick(sorted(rows, key=lambda row: row["rdt_depth_valid_ratio"])),
        "highest_attention_hot_ratio": pick(sorted(rows, key=lambda row: row["rdt_attention_hot_ratio"], reverse=True)),
        "lowest_precision": pick(sorted(rows, key=lambda row: (row["precision_iou50"], -row["gt_count"]))),
    }


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return 0.0
    return _pearson(_rank_average(x), _rank_average(y))


def _rank_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = (start + end - 1) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def _split_evenly(items: Sequence[Mapping[str, Any]], bins: int) -> list[list[Mapping[str, Any]]]:
    return [list(items[round(index * len(items) / bins) : round((index + 1) * len(items) / bins)]) for index in range(bins)]


def _write_report(path: Path, result: Mapping[str, Any]) -> None:
    summary = result["summary"]
    correlations = list(result["correlations"])
    focus_pairs = [
        item
        for item in correlations
        if item["target"] in {"fp_count", "precision_iou50", "recall_iou50", "final_prediction_count", "top100_saturated"}
    ][:20]
    lines = [
        "# RDT prediction correlation analysis",
        "",
        "This is a validation-only diagnostic report. It does not prove a training gain by itself.",
        "",
        "## Inputs",
        "",
    ]
    for key, value in result["inputs"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Overall prediction/error summary",
            "",
            f"- images: `{summary['images']}`",
            f"- total ground-truth objects: `{summary['total_ground_truth_objects']}`",
            f"- total final predictions: `{summary['total_final_predictions']}`",
            f"- IoU@0.50 TP / FP / FN: `{summary['total_iou50_tp']} / {summary['total_iou50_fp']} / {summary['total_iou50_fn']}`",
            f"- mean per-image precision@0.50: `{summary['mean_precision_iou50']:.6f}`",
            f"- mean per-image recall@0.50: `{summary['mean_recall_iou50']:.6f}`",
            "",
            "## Strongest RDT correlations",
            "",
            "| feature | target | Pearson | Spearman |",
            "|---|---|---:|---:|",
        ]
    )
    for item in correlations[:20]:
        lines.append(
            f"| {item['feature']} | {item['target']} | {item['pearson']:.4f} | {item['spearman']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Error-focused correlations",
            "",
            "| feature | target | Pearson | Spearman |",
            "|---|---|---:|---:|",
        ]
    )
    for item in focus_pairs:
        lines.append(
            f"| {item['feature']} | {item['target']} | {item['pearson']:.4f} | {item['spearman']:.4f} |"
        )
    lines.extend(["", "## Quartile diagnostics", ""])
    for feature, bins in result["quartiles"].items():
        lines.extend(
            [
                f"### {feature}",
                "",
                "| bin | feature range | images | FP mean | TP mean | FN mean | precision | recall | top100 sat |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for item in bins:
            lines.append(
                f"| {item['bin']} | {item['feature_min']:.4f}–{item['feature_max']:.4f} | {item['count']} | "
                f"{item['fp_count_mean']:.3f} | {item['tp_count_mean']:.3f} | {item['fn_count_mean']:.3f} | "
                f"{item['precision_iou50_mean']:.4f} | {item['recall_iou50_mean']:.4f} | "
                f"{item['top100_saturation_rate']:.4f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Decision notes",
            "",
            "- Treat `|correlation| >= 0.20` as a potentially useful image-level signal; below that is weak/noisy.",
            "- A good RGC quality feature should correlate with FP/precision/recall or identify a clear low-reliability subset.",
            "- If correlations are weak, keep RDT as visualization/diagnostics rather than starting a heavy training run.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _mean(values: Sequence[float] | Any) -> float:
    seq = list(values)
    if not seq:
        return 0.0
    return _json_float(float(sum(float(value) for value in seq) / len(seq)))


def _json_float(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())
