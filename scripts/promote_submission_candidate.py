#!/usr/bin/env python
"""Promote a deliberate test-set prediction ZIP into outputs/submissions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

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
    parser.add_argument("--nms-iou-threshold", type=float)
    parser.add_argument("--image-max-side", type=int)
    parser.add_argument("--config-path", type=Path)
    parser.add_argument("--quality-cache", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--git-commit")
    parser.add_argument("--source-ranking-json", type=Path)
    parser.add_argument("--source-sweep-ranking-json", type=Path)
    parser.add_argument("--not-ready", dest="ready_for_submit", action="store_false")
    parser.add_argument("--force", action="store_true")
    parser.set_defaults(ready_for_submit=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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
        score_threshold=args.score_threshold,
        nms_iou_threshold=args.nms_iou_threshold,
        image_max_side=args.image_max_side,
        config_path=args.config_path,
        quality_cache=args.quality_cache,
        split_manifest=args.split_manifest,
        git_commit=args.git_commit,
        source_ranking_json=args.source_ranking_json,
        source_sweep_ranking_json=args.source_sweep_ranking_json,
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
