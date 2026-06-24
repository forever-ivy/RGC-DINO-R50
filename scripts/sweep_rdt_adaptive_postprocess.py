#!/usr/bin/env python
"""Sweep RDT-aware image-adaptive postprocessing on cached predictions.

This uses RDT diagnostics as validation-only image quality signals. It does not
run inference or training; it filters/caps an existing prediction cache and
scores final TXT-contract outputs against labels.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rgc_dino.constants import MAX_PREDICTIONS_PER_IMAGE  # noqa: E402
from rgc_dino.labels import DetectionLabel, load_label_dir  # noqa: E402
from rgc_dino.metrics import evaluate_detection_map  # noqa: E402
from rgc_dino.sample_ids import load_sample_ids_file, restrict_mapping_to_sample_ids  # noqa: E402
from rgc_dino.submission import write_submission_files  # noqa: E402


@dataclass(frozen=True)
class AdaptiveRow:
    strategy: str
    feature: str | None
    direction: str | None
    quantile: float | None
    selected_images: int
    selected_top_k: int | None
    selected_threshold: float | None
    default_top_k: int
    default_threshold: float
    map_50_95: float
    map_50: float
    prediction_objects: int
    mean_predictions_per_image: float
    prediction_dir: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-cache", type=Path, required=True)
    parser.add_argument("--rdt-stats", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=ROOT / "source" / "训练集" / "labels")
    parser.add_argument("--sample-ids-file", type=Path, default=ROOT / "outputs" / "splits" / "fold0_val_ids.txt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--features", nargs="+", default=["rdt_depth_valid_ratio", "rdt_attention_mean", "rdt_attention_std", "rdt_depth_attention_mean"])
    parser.add_argument("--quantiles", type=float, nargs="+", default=[0.15, 0.25, 0.35, 0.50])
    parser.add_argument("--default-top-k", type=int, default=MAX_PREDICTIONS_PER_IMAGE)
    parser.add_argument("--selected-top-k", type=int, nargs="+", default=[40, 50, 60, 70, 80, 90])
    parser.add_argument("--default-threshold", type=float, default=0.0)
    parser.add_argument("--selected-threshold", type=float, nargs="+", default=[0.001, 0.003, 0.005, 0.01, 0.02, 0.03])
    parser.add_argument("--clip-labels", action="store_true", default=True)
    parser.add_argument("--no-clip-labels", dest="clip_labels", action="store_false")
    parser.add_argument("--write-best-predictions", action="store_true", default=True)
    parser.add_argument("--top-k-report", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_ids = load_sample_ids_file(args.sample_ids_file)
    predictions = _load_prediction_cache(args.prediction_cache)
    rdt_stats = _load_rdt_stats(args.rdt_stats)
    missing = [sample_id for sample_id in image_ids if sample_id not in rdt_stats]
    if missing:
        print(f"RDT stats missing IDs: {missing[:10]}", file=sys.stderr)
        return 2
    ground_truths = restrict_mapping_to_sample_ids(load_label_dir(args.labels, clip=args.clip_labels), image_ids)

    rows: list[AdaptiveRow] = []
    baseline = _finalize(
        predictions,
        image_ids=image_ids,
        selected_ids=set(),
        default_top_k=args.default_top_k,
        selected_top_k=args.default_top_k,
        default_threshold=args.default_threshold,
        selected_threshold=args.default_threshold,
    )
    base_metric = evaluate_detection_map(ground_truths, baseline)
    rows.append(
        AdaptiveRow(
            strategy="baseline",
            feature=None,
            direction=None,
            quantile=None,
            selected_images=0,
            selected_top_k=None,
            selected_threshold=None,
            default_top_k=args.default_top_k,
            default_threshold=args.default_threshold,
            map_50_95=base_metric.map,
            map_50=base_metric.map50,
            prediction_objects=base_metric.prediction_count,
            mean_predictions_per_image=base_metric.prediction_count / max(1, len(image_ids)),
            prediction_dir=None,
        )
    )

    for feature in args.features:
        values = np.asarray([float(rdt_stats[sample_id][feature]) for sample_id in image_ids], dtype=np.float64)
        for quantile in args.quantiles:
            threshold_value = float(np.quantile(values, quantile))
            high_value = float(np.quantile(values, 1.0 - quantile))
            selections = [
                ("low", {sample_id for sample_id in image_ids if float(rdt_stats[sample_id][feature]) <= threshold_value}),
                ("high", {sample_id for sample_id in image_ids if float(rdt_stats[sample_id][feature]) >= high_value}),
            ]
            for direction, selected_ids in selections:
                for top_k in args.selected_top_k:
                    finalized = _finalize(
                        predictions,
                        image_ids=image_ids,
                        selected_ids=selected_ids,
                        default_top_k=args.default_top_k,
                        selected_top_k=top_k,
                        default_threshold=args.default_threshold,
                        selected_threshold=args.default_threshold,
                    )
                    metric = evaluate_detection_map(ground_truths, finalized)
                    rows.append(
                        AdaptiveRow(
                            strategy="adaptive_top_k",
                            feature=feature,
                            direction=direction,
                            quantile=quantile,
                            selected_images=len(selected_ids),
                            selected_top_k=top_k,
                            selected_threshold=None,
                            default_top_k=args.default_top_k,
                            default_threshold=args.default_threshold,
                            map_50_95=metric.map,
                            map_50=metric.map50,
                            prediction_objects=metric.prediction_count,
                            mean_predictions_per_image=metric.prediction_count / max(1, len(image_ids)),
                            prediction_dir=None,
                        )
                    )
                for score_threshold in args.selected_threshold:
                    finalized = _finalize(
                        predictions,
                        image_ids=image_ids,
                        selected_ids=selected_ids,
                        default_top_k=args.default_top_k,
                        selected_top_k=args.default_top_k,
                        default_threshold=args.default_threshold,
                        selected_threshold=score_threshold,
                    )
                    metric = evaluate_detection_map(ground_truths, finalized)
                    rows.append(
                        AdaptiveRow(
                            strategy="adaptive_threshold",
                            feature=feature,
                            direction=direction,
                            quantile=quantile,
                            selected_images=len(selected_ids),
                            selected_top_k=None,
                            selected_threshold=score_threshold,
                            default_top_k=args.default_top_k,
                            default_threshold=args.default_threshold,
                            map_50_95=metric.map,
                            map_50=metric.map50,
                            prediction_objects=metric.prediction_count,
                            mean_predictions_per_image=metric.prediction_count / max(1, len(image_ids)),
                            prediction_dir=None,
                        )
                    )

    ranked = sorted(rows, key=lambda row: row.map_50_95, reverse=True)
    best = ranked[0]
    if args.write_best_predictions and best.strategy != "baseline":
        selected_ids = _selected_ids_for_row(best, image_ids=image_ids, rdt_stats=rdt_stats)
        best_predictions = _finalize(
            predictions,
            image_ids=image_ids,
            selected_ids=selected_ids,
            default_top_k=best.default_top_k,
            selected_top_k=best.selected_top_k or best.default_top_k,
            default_threshold=best.default_threshold,
            selected_threshold=best.selected_threshold if best.selected_threshold is not None else best.default_threshold,
        )
        best_dir = args.output_dir / "best_predictions"
        write_submission_files(image_ids, best_predictions, best_dir, max_predictions_per_image=MAX_PREDICTIONS_PER_IMAGE)
        best = AdaptiveRow(**{**asdict(best), "prediction_dir": str(best_dir)})
        ranked = [best if row is ranked[0] else row for row in ranked]

    payload = {
        "inputs": {
            "prediction_cache": str(args.prediction_cache),
            "rdt_stats": str(args.rdt_stats),
            "sample_ids_file": str(args.sample_ids_file),
            "labels": str(args.labels),
        },
        "baseline_map_50_95": base_metric.map,
        "baseline_map_50": base_metric.map50,
        "best": asdict(best),
        "rows": [asdict(row) for row in ranked],
    }
    (args.output_dir / "rdt_adaptive_postprocess_sweep.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_report(args.output_dir / "rdt_adaptive_postprocess_sweep_report.md", payload, ranked[: args.top_k_report])

    print("=" * 96)
    print("RDT-aware adaptive postprocess sweep")
    print("=" * 96)
    print(f"baseline mAP@50:95: {base_metric.map:.6f}")
    print(f"best mAP@50:95: {best.map_50_95:.6f}")
    print(f"best: {best}")
    print(f"report: {args.output_dir / 'rdt_adaptive_postprocess_sweep_report.md'}")
    return 0


def _load_prediction_cache(path: Path) -> dict[str, list[DetectionLabel]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("predictions", payload) if isinstance(payload, dict) else payload
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
        label = DetectionLabel(
            class_id=int(row["class_id"]),
            norm_center_x=float(row["norm_center_x"]),
            norm_center_y=float(row["norm_center_y"]),
            norm_w=float(row["norm_w"]),
            norm_h=float(row["norm_h"]),
            confidence=score,
        )
        predictions.setdefault(str(row["sample_id"]), []).append(label)
    return predictions


def _load_rdt_stats(path: Path) -> dict[str, dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(sample_id): {str(key): float(value) for key, value in stats.items()} for sample_id, stats in payload.items()}


def _finalize(
    predictions: dict[str, list[DetectionLabel]],
    *,
    image_ids: list[str],
    selected_ids: set[str],
    default_top_k: int,
    selected_top_k: int,
    default_threshold: float,
    selected_threshold: float,
) -> dict[str, list[DetectionLabel]]:
    finalized: dict[str, list[DetectionLabel]] = {}
    for sample_id in image_ids:
        threshold = selected_threshold if sample_id in selected_ids else default_threshold
        top_k = selected_top_k if sample_id in selected_ids else default_top_k
        records = [record for record in predictions.get(sample_id, []) if (record.confidence or 0.0) >= threshold]
        records.sort(key=lambda record: record.confidence or 0.0, reverse=True)
        finalized[sample_id] = records[:top_k]
    return finalized


def _selected_ids_for_row(
    row: AdaptiveRow,
    *,
    image_ids: list[str],
    rdt_stats: dict[str, dict[str, float]],
) -> set[str]:
    if row.feature is None or row.direction is None or row.quantile is None:
        return set()
    values = np.asarray([float(rdt_stats[sample_id][row.feature]) for sample_id in image_ids], dtype=np.float64)
    if row.direction == "low":
        threshold_value = float(np.quantile(values, row.quantile))
        return {sample_id for sample_id in image_ids if float(rdt_stats[sample_id][row.feature]) <= threshold_value}
    high_value = float(np.quantile(values, 1.0 - row.quantile))
    return {sample_id for sample_id in image_ids if float(rdt_stats[sample_id][row.feature]) >= high_value}


def _write_report(path: Path, payload: dict[str, Any], top_rows: list[AdaptiveRow]) -> None:
    lines = [
        "# RDT-aware adaptive postprocess sweep",
        "",
        f"- baseline mAP@50:95: `{payload['baseline_map_50_95']:.6f}`",
        f"- baseline mAP@50: `{payload['baseline_map_50']:.6f}`",
        f"- best mAP@50:95: `{payload['best']['map_50_95']:.6f}`",
        f"- best mAP@50: `{payload['best']['map_50']:.6f}`",
        f"- best strategy: `{payload['best']['strategy']}`",
        "",
        "## Top rows",
        "",
        "| rank | strategy | feature | direction | q | selected | topK | thr | mAP@50:95 | mAP@50 | preds |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(top_rows, start=1):
        lines.append(
            f"| {rank} | {row.strategy} | {row.feature or ''} | {row.direction or ''} | "
            f"{'' if row.quantile is None else f'{row.quantile:.2f}'} | {row.selected_images} | "
            f"{'' if row.selected_top_k is None else row.selected_top_k} | "
            f"{'' if row.selected_threshold is None else f'{row.selected_threshold:.4g}'} | "
            f"{row.map_50_95:.6f} | {row.map_50:.6f} | {row.prediction_objects} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
