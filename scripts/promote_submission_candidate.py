#!/usr/bin/env python
"""Promote a deliberate test-set prediction ZIP into outputs/submissions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rgc_dino.postprocess import load_class_score_thresholds  # noqa: E402
from rgc_dino.submission_promotion import promote_submission_candidate  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_zip", type=Path)
    parser.add_argument("--submissions-dir", type=Path, default=ROOT / "outputs" / "submissions")
    parser.add_argument("--reason", required=True)
    parser.add_argument("--local-map", type=float)
    parser.add_argument("--leaderboard-baseline", type=float)
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "source" / "AIC2026_PHASE_1_1000")
    parser.add_argument("--candidate-kind", default="manual")
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--train-dir", type=Path)
    parser.add_argument("--val-map-50-95", type=float)
    parser.add_argument("--val-map-50", type=float)
    parser.add_argument("--prediction-objects", type=int)
    parser.add_argument("--score-threshold", type=float)
    parser.add_argument("--class-score-thresholds", type=Path)
    parser.add_argument("--candidate-score-threshold", type=float)
    parser.add_argument("--nms-iou-threshold", type=float)
    parser.add_argument("--pre-limit-per-image", type=int)
    parser.add_argument("--max-detections", type=int)
    parser.add_argument("--image-max-side", type=int)
    parser.add_argument("--config-path", type=Path)
    parser.add_argument("--quality-cache", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--git-commit")
    parser.add_argument("--source-ranking-json", type=Path)
    parser.add_argument("--source-sweep-ranking-json", type=Path)
    parser.add_argument("--threshold-sweep-json", type=Path)
    parser.add_argument("--diagnostics-path", type=Path)
    parser.add_argument("--hard-val-report-path", type=Path)
    parser.add_argument("--conversion-summary-path", type=Path)
    parser.add_argument("--raw-prediction-summary-json", type=Path)
    parser.add_argument("--after-class-threshold-summary-json", type=Path)
    parser.add_argument("--after-nms-summary-json", type=Path)
    parser.add_argument("--after-topk-summary-json", type=Path)
    parser.add_argument("--hard-val-status")
    parser.add_argument("--hard-val-map-50-95", type=float)
    parser.add_argument("--not-ready", dest="ready_for_submit", action="store_false")
    parser.add_argument("--force", action="store_true")
    parser.set_defaults(ready_for_submit=True)
    return parser.parse_args()


def _load_json_arg(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    conversion_summary = _load_json_arg(args.conversion_summary_path)
    class_thresholds = (
        load_class_score_thresholds(args.class_score_thresholds)
        if args.class_score_thresholds is not None
        else (conversion_summary or {}).get("class_score_thresholds")
    )
    class_thresholds_path = args.class_score_thresholds
    if class_thresholds_path is None and conversion_summary is not None:
        raw_path = conversion_summary.get("class_score_thresholds_path")
        class_thresholds_path = Path(raw_path) if raw_path else None
    score_threshold = args.score_threshold
    if score_threshold is None and conversion_summary is not None:
        score_threshold = conversion_summary.get("score_threshold")
    candidate_score_threshold = args.candidate_score_threshold
    if candidate_score_threshold is None:
        candidate_score_threshold = score_threshold
    max_detections = args.max_detections
    if max_detections is None and conversion_summary is not None:
        max_detections = conversion_summary.get("max_detections")
    nms_iou_threshold = args.nms_iou_threshold
    if nms_iou_threshold is None and conversion_summary is not None:
        nms_iou_threshold = conversion_summary.get("nms_iou_threshold")
    raw_prediction_summary = _load_json_arg(args.raw_prediction_summary_json)
    after_class_threshold_summary = _load_json_arg(args.after_class_threshold_summary_json)
    after_nms_summary = _load_json_arg(args.after_nms_summary_json)
    after_topk_summary = _load_json_arg(args.after_topk_summary_json)
    if conversion_summary is not None:
        raw_prediction_summary = raw_prediction_summary or conversion_summary.get("raw_prediction_summary")
        after_class_threshold_summary = after_class_threshold_summary or conversion_summary.get("after_class_threshold_summary")
        after_nms_summary = after_nms_summary or conversion_summary.get("after_nms_summary")
        after_topk_summary = after_topk_summary or conversion_summary.get("after_topk_summary")
    result = promote_submission_candidate(
        candidate_zip=args.candidate_zip,
        submissions_dir=args.submissions_dir,
        reason=args.reason,
        local_map=args.local_map,
        leaderboard_baseline=args.leaderboard_baseline,
        force=args.force,
        manifest_path=args.manifest_path,
        dataset_root=args.dataset_root,
        candidate_kind=args.candidate_kind,
        checkpoint_path=args.checkpoint_path,
        epoch=args.epoch,
        train_dir=args.train_dir,
        val_map_50_95=args.val_map_50_95,
        val_map_50=args.val_map_50,
        prediction_objects=args.prediction_objects,
        score_threshold=score_threshold,
        class_score_thresholds_path=class_thresholds_path,
        class_score_thresholds=class_thresholds,
        candidate_score_threshold=candidate_score_threshold,
        nms_iou_threshold=nms_iou_threshold,
        pre_limit_per_image=args.pre_limit_per_image,
        max_detections=max_detections,
        image_max_side=args.image_max_side,
        config_path=args.config_path,
        quality_cache=args.quality_cache,
        split_manifest=args.split_manifest,
        git_commit=args.git_commit,
        source_ranking_json=args.source_ranking_json,
        source_sweep_ranking_json=args.source_sweep_ranking_json,
        threshold_sweep_json=args.threshold_sweep_json,
        diagnostics_path=args.diagnostics_path,
        hard_val_report_path=args.hard_val_report_path,
        conversion_summary_path=args.conversion_summary_path,
        raw_prediction_summary=raw_prediction_summary,
        after_class_threshold_summary=after_class_threshold_summary,
        after_nms_summary=after_nms_summary,
        after_topk_summary=after_topk_summary,
        hard_val_status=args.hard_val_status,
        hard_val_map_50_95=args.hard_val_map_50_95,
        ready_for_submit=args.ready_for_submit,
    )
    print(
        json.dumps(
            {
                "promoted_zip": str(result.zip_path),
                "metadata_path": str(result.metadata_path),
                "zip_sha256": result.zip_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
