#!/usr/bin/env python
"""Evaluate new checkpoints, generate evidence-backed ZIPs, and promote candidates.

This script is the Python replacement for the older incremental_submit.sh /
auto_submit_best.sh / monitor_and_submit.sh shell logic. It deliberately stops at
promotion; real platform submission is handled by monitor_competition.py.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rgc_dino.submission_manifest import file_sha256  # noqa: E402
from rgc_dino.submission_promotion import promote_submission_candidate  # noqa: E402

SELECT_BEST = ROOT / "scripts" / "select_best_checkpoint.py"
SWEEP = ROOT / "scripts" / "sweep_inference_params.py"
INFER = ROOT / "scripts" / "infer_rgc_dino.py"
DEFAULT_TRAIN_ROOT = ROOT / "source" / "训练集"
DEFAULT_TEST_ROOT = ROOT / "source" / "AIC2026_PHASE_1_1000"
DEFAULT_LABELS = DEFAULT_TRAIN_ROOT / "labels"
CURRENT_CODETR_STRICT_MAP = 0.4379615851682616
CURRENT_LEADERBOARD_BASELINE = 50.353


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--val-ids", type=Path, default=ROOT / "outputs" / "splits" / "fold0_val_ids.txt")
    parser.add_argument("--dataset-root-val", type=Path, default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--dataset-root-test", type=Path, default=DEFAULT_TEST_ROOT)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--quality-cache", type=Path)
    parser.add_argument("--config-file", type=Path, default=ROOT / "configs" / "dino_a0_rgb_4scale.py")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "automation")
    parser.add_argument("--submissions-dir", type=Path, default=ROOT / "outputs" / "submissions")
    parser.add_argument("--epochs", type=int, nargs="+")
    parser.add_argument("--min-epoch", type=int)
    parser.add_argument("--include-latest", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-max-side", type=int, default=640)
    parser.add_argument("--side-base-channels", type=int, default=32)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--nms-iou-threshold", type=float)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--sweep-thresholds", type=float, nargs="+")
    parser.add_argument("--sweep-nms", type=float, nargs="+")
    parser.add_argument("--sweep-top-k", type=int, default=5)
    parser.add_argument("--skip-sweep", action="store_true")
    parser.add_argument("--require-improvement-over", type=float, default=0.0)
    parser.add_argument("--baseline-val-map", type=float, default=CURRENT_CODETR_STRICT_MAP)
    parser.add_argument("--leaderboard-baseline", type=float, default=CURRENT_LEADERBOARD_BASELINE)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-interval", type=int, default=600)
    parser.add_argument("--auto-promote", action="store_true", help="Promote newly generated best ZIP into submissions dir")
    parser.add_argument("--dry-run", action="store_true", help="Print planned heavy commands without running inference/eval")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def discover_checkpoint_epochs(train_dir: Path) -> list[int]:
    epochs: list[int] = []
    for path in sorted(train_dir.glob("checkpoint[0-9][0-9][0-9][0-9].pth")):
        match = re.search(r"checkpoint(\d+)\.pth$", path.name)
        if match:
            epochs.append(int(match.group(1)))
    return epochs


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "evaluated_epochs": [],
            "best_epoch": None,
            "best_local_map": None,
            "last_promoted_epoch": None,
            "promoted_sha256s": [],
            "skipped_epochs": [],
        }
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def select_epochs_to_evaluate(args: argparse.Namespace, state: dict[str, Any]) -> list[int]:
    available = args.epochs if args.epochs is not None else discover_checkpoint_epochs(args.train_dir)
    evaluated = set(int(epoch) for epoch in state.get("evaluated_epochs", []))
    selected = []
    for epoch in available:
        if args.min_epoch is not None and epoch < args.min_epoch:
            continue
        if epoch in evaluated:
            continue
        selected.append(epoch)
    return sorted(selected)


def run_command(cmd: list[str], *, dry_run: bool) -> subprocess.CompletedProcess[str]:
    print("$ " + " ".join(cmd), flush=True)
    if dry_run:
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)


def run_checkpoint_selection(args: argparse.Namespace, epochs: list[int], eval_dir: Path) -> Path:
    cmd = [
        sys.executable, str(SELECT_BEST),
        "--train-dir", str(args.train_dir),
        "--val-ids", str(args.val_ids),
        "--dataset-root", str(args.dataset_root_val),
        "--labels", str(args.labels),
        "--config-file", str(args.config_file),
        "--output-dir", str(eval_dir),
        "--device", args.device,
        "--image-max-side", str(args.image_max_side),
        "--side-base-channels", str(args.side_base_channels),
        "--score-threshold", str(args.score_threshold),
        "--max-detections", str(args.max_detections),
        "--epochs", *[str(epoch) for epoch in epochs],
    ]
    if args.quality_cache:
        cmd += ["--quality-cache", str(args.quality_cache)]
    if args.nms_iou_threshold is not None:
        cmd += ["--nms-iou-threshold", str(args.nms_iou_threshold)]
    if args.amp:
        cmd += ["--amp"]
    result = run_command(cmd, dry_run=args.dry_run)
    if result.returncode != 0:
        raise RuntimeError(f"checkpoint selection failed: {result.stderr[-800:]}")
    return eval_dir / "ranking.json"


def load_best_ranking(ranking_path: Path) -> dict[str, Any] | None:
    if not ranking_path.exists():
        return None
    rows = json.loads(ranking_path.read_text(encoding="utf-8"))
    return next((row for row in rows if row.get("map_50_95") is not None and not row.get("error")), None)


def run_optional_sweep(args: argparse.Namespace, checkpoint: Path, output_dir: Path) -> tuple[float, float | None, Path | None, dict[str, Any] | None]:
    if args.skip_sweep or (not args.sweep_thresholds and not args.sweep_nms):
        return args.score_threshold, args.nms_iou_threshold, None, None
    cmd = [
        sys.executable, str(SWEEP),
        "--checkpoint", str(checkpoint),
        "--val-ids", str(args.val_ids),
        "--dataset-root", str(args.dataset_root_val),
        "--labels", str(args.labels),
        "--output-dir", str(output_dir),
        "--device", args.device,
        "--top-k", str(args.sweep_top_k),
    ]
    if args.quality_cache:
        cmd += ["--quality-cache", str(args.quality_cache)]
    if args.sweep_thresholds:
        cmd += ["--thresholds", *[str(value) for value in args.sweep_thresholds]]
    if args.sweep_nms:
        cmd += ["--nms-iou", *[str(value) for value in args.sweep_nms]]
    if args.amp:
        cmd += ["--amp"]
    result = run_command(cmd, dry_run=args.dry_run)
    if result.returncode != 0:
        raise RuntimeError(f"sweep failed: {result.stderr[-800:]}")
    ranking_path = output_dir / "sweep_ranking.json"
    if args.dry_run or not ranking_path.exists():
        return args.score_threshold, args.nms_iou_threshold, ranking_path, None
    rows = json.loads(ranking_path.read_text(encoding="utf-8"))
    best = next((row for row in rows if not row.get("error")), None)
    if not best:
        return args.score_threshold, args.nms_iou_threshold, ranking_path, None
    config = best.get("config", {})
    return config.get("score_threshold", args.score_threshold), config.get("nms_iou_threshold"), ranking_path, best


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def run_test_inference(
    args: argparse.Namespace,
    *,
    checkpoint: Path,
    epoch: int,
    score_threshold: float,
    nms_iou_threshold: float | None,
    run_dir: Path,
) -> tuple[Path, Path, Path]:
    tag_parts = [args.train_dir.name, f"ep{epoch:04d}", f"map{score_threshold:.3f}".replace(".", "")]
    if nms_iou_threshold is not None:
        tag_parts.append(f"nms{int(nms_iou_threshold * 100):02d}")
    tag = "_".join(tag_parts)
    pred_dir = run_dir / "test_predictions" / tag
    zip_path = run_dir / "candidates" / f"{tag}.zip"
    manifest_path = zip_path.with_suffix(".manifest.json")
    cmd = [
        sys.executable, str(INFER),
        "--config-file", str(args.config_file),
        "--dataset-root", str(args.dataset_root_test),
        "--checkpoint", str(checkpoint),
        "--model-mode", "rgc",
        "--checkpoint-scope", "auto",
        "--output-dir", str(pred_dir),
        "--zip-path", str(zip_path),
        "--manifest-path", str(manifest_path),
        "--device", args.device,
        "--image-max-side", str(args.image_max_side),
        "--side-base-channels", str(args.side_base_channels),
        "--score-threshold", str(score_threshold),
        "--max-detections", str(args.max_detections),
    ]
    if args.quality_cache:
        cmd += ["--quality-cache", str(args.quality_cache)]
    if nms_iou_threshold is not None:
        cmd += ["--nms-iou-threshold", str(nms_iou_threshold)]
    if args.amp:
        cmd += ["--amp"]
    result = run_command(cmd, dry_run=args.dry_run)
    if result.returncode != 0:
        raise RuntimeError(f"test inference failed: {result.stderr[-800:]}")
    return pred_dir, zip_path, manifest_path


def should_promote(
    best: dict[str, Any],
    state: dict[str, Any],
    required_improvement: float,
    baseline_val_map: float = 0.0,
) -> bool:
    current_map = best.get("map_50_95")
    previous = state.get("best_local_map")
    if current_map is None:
        return False
    required_baseline = float(baseline_val_map)
    if previous is not None:
        required_baseline = max(required_baseline, float(previous) + required_improvement)
    return float(current_map) > required_baseline


def run_once(args: argparse.Namespace) -> dict[str, Any]:
    state_path = args.train_dir / "submission_pipeline_state.json"
    state = load_state(state_path)
    epochs = select_epochs_to_evaluate(args, state)
    if not epochs:
        print("No new checkpoints to evaluate")
        return {"status": "no_new_checkpoints", "state_path": str(state_path)}

    run_dir = args.output_dir / args.train_dir.name
    eval_dir = run_dir / "val_eval"
    ranking_path = run_checkpoint_selection(args, epochs, eval_dir)
    best = load_best_ranking(ranking_path)
    state["evaluated_epochs"] = sorted(set(state.get("evaluated_epochs", [])) | set(epochs))
    state["last_pipeline_run"] = datetime.now().isoformat()

    if best is None:
        state.setdefault("skipped_epochs", []).append({"epochs": epochs, "reason": "no successful ranking row"})
        save_state(state_path, state)
        return {"status": "no_successful_checkpoint", "ranking_path": str(ranking_path)}

    if not should_promote(best, state, args.require_improvement_over, args.baseline_val_map):
        save_state(state_path, state)
        previous = state.get('best_local_map')
        required_baseline = float(args.baseline_val_map)
        if previous is not None:
            required_baseline = max(required_baseline, float(previous) + args.require_improvement_over)
        print(
            f"Best new mAP {best.get('map_50_95')} did not exceed required baseline "
            f"{required_baseline} (previous={previous}, absolute_anchor={args.baseline_val_map})"
        )
        status = "below_current_anchor" if previous is None or float(args.baseline_val_map) >= required_baseline else "not_improved"
        return {"status": status, "best": best, "ranking_path": str(ranking_path), "required_baseline": required_baseline}

    epoch = int(best["epoch"])
    checkpoint = args.train_dir / best["checkpoint"]
    score_threshold, nms_iou, sweep_ranking_path, sweep_best = run_optional_sweep(
        args,
        checkpoint,
        run_dir / "sweep" / f"ep{epoch:04d}",
    )
    _pred_dir, zip_path, manifest_path = run_test_inference(
        args,
        checkpoint=checkpoint,
        epoch=epoch,
        score_threshold=score_threshold,
        nms_iou_threshold=nms_iou,
        run_dir=run_dir,
    )

    promoted = None
    if args.auto_promote and not args.dry_run:
        reason = (
            f"{args.train_dir.name} epoch {epoch} val mAP@50:95={best.get('map_50_95'):.4f}; "
            f"threshold={score_threshold}, nms={nms_iou}"
        )
        promoted = promote_submission_candidate(
            candidate_zip=zip_path,
            submissions_dir=args.submissions_dir,
            reason=reason,
            local_map=best.get("map_50_95"),
            leaderboard_baseline=args.leaderboard_baseline,
            force=args.force,
            manifest_path=manifest_path,
            dataset_root=args.dataset_root_test,
            candidate_kind="inference_sweep_best" if sweep_best else "checkpoint_best",
            checkpoint_path=checkpoint,
            epoch=epoch,
            train_dir=args.train_dir,
            val_map_50_95=best.get("map_50_95"),
            val_map_50=best.get("map_50"),
            prediction_objects=best.get("prediction_objects"),
            score_threshold=score_threshold,
            nms_iou_threshold=nms_iou,
            image_max_side=args.image_max_side,
            config_path=args.config_file,
            quality_cache=args.quality_cache,
            split_manifest=ROOT / "outputs" / "splits" / "split_manifest.json",
            git_commit=_git_commit(),
            source_ranking_json=ranking_path,
            source_sweep_ranking_json=sweep_ranking_path,
        )
        promoted_shas = set(state.get("promoted_sha256s", []))
        promoted_shas.add(promoted.zip_sha256)
        state["promoted_sha256s"] = sorted(promoted_shas)
        state["last_promoted_epoch"] = epoch
    elif args.auto_promote and args.dry_run:
        print(f"DRY-RUN would promote {zip_path} with manifest {manifest_path}")

    state["best_epoch"] = epoch
    state["best_local_map"] = best.get("map_50_95")
    state["best_checkpoint_sha256"] = file_sha256(checkpoint) if checkpoint.exists() else None
    save_state(state_path, state)
    return {
        "status": "promoted" if promoted else "candidate_generated",
        "best": best,
        "zip_path": str(zip_path),
        "manifest_path": str(manifest_path),
        "promoted_zip": str(promoted.zip_path) if promoted else None,
        "state_path": str(state_path),
    }


def main() -> int:
    args = parse_args()
    if not args.train_dir.exists():
        print(f"train dir not found: {args.train_dir}", file=sys.stderr)
        return 2
    while True:
        try:
            result = run_once(args)
            print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
        except Exception as exc:  # noqa: BLE001 - persist clear CLI error
            print(f"pipeline failed: {exc}", file=sys.stderr)
            return 1
        if args.once:
            return 0
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    raise SystemExit(main())
