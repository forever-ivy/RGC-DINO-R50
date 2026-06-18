#!/usr/bin/env python
"""Prepare and preflight the Co-DETR + InternImage-L training route.

This command is intentionally lightweight: it exports fold COCO annotations,
checks local external-code/weight paths, and writes LSF job scripts.  It never
submits jobs and never starts training in the interactive shell.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rgc_dino.codetr_integration import check_codetr_tree, format_status  # noqa: E402
from rgc_dino.coco_export import load_split_ids, write_coco_rgb_dataset  # noqa: E402

# Import sibling scripts after adding ROOT to sys.path.  The repository's
# ``scripts`` directory is not a Python package, so import by path-independent
# module name while the current working directory is the repo root.
from write_bsub_codetr_smoke import render_bsub_script as render_smoke_script  # noqa: E402
from write_bsub_codetr_train import render_bsub_script as render_train_script  # noqa: E402

DEFAULT_CODETR_ROOT = Path("external/Co-DETR")
DEFAULT_R50_CONFIG = ROOT / "configs" / "codetr_r50_stage0_mm_config.py"
DEFAULT_INTERNIMAGE_CONFIG = ROOT / "configs" / "codetr_internimage_l_mm_config.py"
DEFAULT_COCO_OUTPUT = ROOT / "outputs" / "codetr_coco" / "fold0"
DEFAULT_SMOKE_JOB = ROOT / "outputs" / "jobs" / "codetr_smoke.lsf"
DEFAULT_R50_TRAIN_JOB = ROOT / "outputs" / "jobs" / "codetr_r50_stage0_train.lsf"
DEFAULT_INTERNIMAGE_TRAIN_JOB = ROOT / "outputs" / "jobs" / "codetr_internimage_l_stage0_train.lsf"
DEFAULT_R50_WORK_DIR = ROOT / "outputs" / "codetr" / "r50_stage0_fold0"
DEFAULT_INTERNIMAGE_WORK_DIR = ROOT / "outputs" / "codetr" / "internimage_l_stage0_fold0"
DEFAULT_INTERNIMAGE_WEIGHTS = Path("/data1/liuxuan/checkpoints/internimage/internimage_l_public_pretrain.pth")
DEFAULT_CODETR_WEIGHTS = Path("/data1/liuxuan/checkpoints/codetr/codetr_internimage_l_public_pretrain.pth")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--stage", choices=("r50_sanity", "internimage_l"), default="r50_sanity")
    parser.add_argument("--codetr-root", type=Path, default=DEFAULT_CODETR_ROOT)
    parser.add_argument("--config", type=Path, help="override the stage default Co-DETR config")
    parser.add_argument("--coco-output", type=Path, default=DEFAULT_COCO_OUTPUT)
    parser.add_argument("--smoke-job", type=Path, default=DEFAULT_SMOKE_JOB)
    parser.add_argument("--train-job", type=Path, help="override the stage default LSF output")
    parser.add_argument("--work-dir", type=Path, help="override the stage default work directory")
    parser.add_argument("--internimage-weights", type=Path, default=DEFAULT_INTERNIMAGE_WEIGHTS)
    parser.add_argument("--codetr-weights", type=Path, default=DEFAULT_CODETR_WEIGHTS)
    parser.add_argument(
        "--require-weights",
        action="store_true",
        help="force required-weight checks; defaults to true only for internimage_l stage",
    )
    parser.add_argument("--no-require-weights", action="store_false", dest="require_weights")
    parser.set_defaults(require_weights=None)
    parser.add_argument("--skip-export", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.config is None:
        args.config = DEFAULT_R50_CONFIG if args.stage == "r50_sanity" else DEFAULT_INTERNIMAGE_CONFIG
    if args.train_job is None:
        args.train_job = DEFAULT_R50_TRAIN_JOB if args.stage == "r50_sanity" else DEFAULT_INTERNIMAGE_TRAIN_JOB
    if args.work_dir is None:
        args.work_dir = DEFAULT_R50_WORK_DIR if args.stage == "r50_sanity" else DEFAULT_INTERNIMAGE_WORK_DIR
    if args.require_weights is None:
        args.require_weights = args.stage == "internimage_l"
    weight_paths = [args.internimage_weights, args.codetr_weights] if args.stage == "internimage_l" else []

    if not args.skip_export:
        train_ids, val_ids = load_split_ids(ROOT / "outputs" / "splits" / "fold_assignments.jsonl", fold=args.fold)
        write_coco_rgb_dataset(
            dataset_root=ROOT / "source" / "训练集",
            labels_dir=ROOT / "source" / "训练集" / "labels",
            output_root=args.coco_output,
            train_ids=train_ids,
            val_ids=val_ids,
            clip_labels=True,
        )
        print(f"wrote: {args.coco_output}")
        print(f"train_ids: {len(train_ids)}")
        print(f"val_ids: {len(val_ids)}")

    args.smoke_job.parent.mkdir(parents=True, exist_ok=True)
    args.train_job.parent.mkdir(parents=True, exist_ok=True)
    args.smoke_job.write_text(
        render_smoke_script(
            job_name="codetr-smoke",
            queue="normal",
            gpu=1,
            codetr_root=args.codetr_root,
            config=args.config,
            coco_output=args.coco_output,
            internimage_weights=args.internimage_weights,
            codetr_weights=args.codetr_weights,
            require_weights=args.stage == "internimage_l" and args.require_weights,
            env_prefix=Path("/data1/liuxuan/envs/codetr"),
        ),
        encoding="utf-8",
    )
    args.train_job.write_text(
        render_train_script(
            job_name="codetr-r50-stage0" if args.stage == "r50_sanity" else "codetr-intl-stage0",
            queue="normal",
            gpu=1 if args.stage == "r50_sanity" else 2,
            fold=args.fold,
            codetr_root=args.codetr_root,
            config=args.config,
            coco_output=args.coco_output,
            work_dir=args.work_dir,
            internimage_weights=args.internimage_weights,
            codetr_weights=args.codetr_weights,
            require_weights=args.stage == "internimage_l" and args.require_weights,
            num_workers=0,
            env_prefix=Path("/data1/liuxuan/envs/codetr"),
        ),
        encoding="utf-8",
    )

    status = check_codetr_tree(
        args.codetr_root,
        weight_paths=weight_paths,
        require_weights=args.stage == "internimage_l" and args.require_weights,
    )
    config_exists = args.config.exists()
    coco_ready = (args.coco_output / "annotations" / "instances_train2017.json").exists() and (
        args.coco_output / "annotations" / "instances_val2017.json"
    ).exists()
    ready = status.ok and config_exists and coco_ready
    summary = {
        "ready_for_manual_bsub": ready,
        "stage": args.stage,
        "codetr_status_ok": status.ok,
        "config_exists": config_exists,
        "coco_ready": coco_ready,
        "smoke_job": str(args.smoke_job),
        "train_job": str(args.train_job),
        "next_manual_commands": [
            f"bsub < {args.smoke_job}",
            f"bsub < {args.train_job}",
        ] if ready else [],
        "blocked_by": [] if ready else _blocked_by(status_ok=status.ok, config_exists=config_exists, coco_ready=coco_ready),
    }
    print(format_status(status))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if ready else 2


def _blocked_by(*, status_ok: bool, config_exists: bool, coco_ready: bool) -> list[str]:
    blocked: list[str] = []
    if not status_ok:
        blocked.append("external Co-DETR tree and/or required public pretrained weights are missing")
    if not config_exists:
        blocked.append("Co-DETR config scaffold is missing")
    if not coco_ready:
        blocked.append("COCO fold export is missing")
    return blocked


if __name__ == "__main__":
    raise SystemExit(main())
