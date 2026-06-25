#!/usr/bin/env python
"""Write LSF scripts for safe Co-DETR InternImage-L continuation jobs.

The generated script runs one GPU job, evaluates the resulting checkpoint under
final-TXT validation, and stops before test inference unless the strict local
metric beats the supplied baseline.  The default baseline is the current
Co-DETR InternImage-L GPU1 load-from epoch6 top100 allocation anchor
(50.353 leaderboard / 0.437940274 strict mAP).  This writer does not submit the job.
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "jobs" / "codetr_internimage_l_continue.lsf"
DEFAULT_CODETR_ROOT = Path("external/Co-DETR")
DEFAULT_CONFIG = ROOT / "configs" / "codetr_internimage_l_mm_config.py"
DEFAULT_BASE_CKPT = ROOT / "outputs" / "codetr" / "internimage_l_epoch20_fresh_ft8_fold0_20260621_epoch20_fresh_ft8_direct" / "best_bbox_mAP_epoch_7.pth"
DEFAULT_WORK_DIR = ROOT / "outputs" / "codetr" / "internimage_l_continue_from_fresh_ep7"
DEFAULT_ENV_PREFIX = Path("/data1/liuxuan/envs/codetr")
DEFAULT_BASELINE = 0.4379615851682616
DEFAULT_LEADERBOARD_BASELINE = 50.353


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--job-name", default="codetr-intl-cont")
    parser.add_argument("--queue", default="normal")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--codetr-root", type=Path, default=DEFAULT_CODETR_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    init = parser.add_mutually_exclusive_group(required=True)
    init.add_argument("--resume-from", type=Path)
    init.add_argument("--load-from", type=Path)
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--lr-steps", default="[8,10]")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--env-prefix", type=Path, default=DEFAULT_ENV_PREFIX)
    parser.add_argument("--baseline-val-map", type=float, default=DEFAULT_BASELINE)
    parser.add_argument("--leaderboard-baseline", type=float, default=DEFAULT_LEADERBOARD_BASELINE)
    parser.add_argument("--post-allocation-sweep", action="store_true", default=True)
    parser.add_argument("--no-post-allocation-sweep", dest="post_allocation_sweep", action="store_false")
    return parser.parse_args()


def render_script(args: argparse.Namespace) -> str:
    init_arg = f"--resume-from {args.resume_from}" if args.resume_from is not None else ""
    cfg_options = [
        f"data.workers_per_gpu={args.num_workers}",
        f"runner.max_epochs={args.max_epochs}",
        f"lr_config.step={args.lr_steps}",
        "checkpoint_config.max_keep_ckpts=4",
        "log_config.interval=50",
    ]
    if args.load_from is not None:
        cfg_options.append(f"load_from={args.load_from}")
    cfg_options_text = " \\\n    ".join(cfg_options)
    allocation_block = ""
    if args.post_allocation_sweep:
        allocation_block = f'''
PYTHONPATH=src python scripts/sweep_codetr_top100_allocation.py \\
  --results-pkl "$VAL_OUT/results.pkl" \\
  --coco-ann outputs/codetr_coco/fold0/annotations/instances_val2017.json \\
  --labels source/训练集/labels \\
  --hard-val-sample-ids-file outputs/codetr/phase35_hard_val/hard_val_sample_ids.txt \\
  --initial-thresholds-json outputs/codetr/internimage_l_epoch20_fresh_ft8_fold0_20260621_epoch20_fresh_ft8_direct/diagnostics/best_bbox_mAP_epoch_7_class_threshold_resweep_20260622/class_score_thresholds.json \\
  --mode single-class-weight \\
  --target-classes 0 1 2 4 5 7 8 11 \\
  --weights 0.85 0.90 0.95 1.0 1.05 1.10 1.20 1.30 \\
  --max-detections 100 \\
  --output-dir "$VAL_OUT/top100_allocation_sweep" \\
  --no-write-best-predictions
'''
    return f'''#!/usr/bin/env bash
#BSUB -J {args.job_name}
#BSUB -q {args.queue}
#BSUB -n {max(2, args.gpu * 2)}
#BSUB -gpu "num={args.gpu}:mode=exclusive_process"
#BSUB -R "rusage[mem=32000]"
#BSUB -o /data1/liuxuan/logs/{args.job_name}.%J.out
#BSUB -e /data1/liuxuan/logs/{args.job_name}.%J.err

set -e
cd {ROOT}
. /data1/miniconda3/etc/profile.d/conda.sh
conda activate {args.env_prefix}

mkdir -p /data1/liuxuan/logs {args.work_dir} outputs/codetr_coco/fold0

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

python scripts/check_codetr_integration.py \
  --codetr-root {args.codetr_root} \
  --internimage-weights /data1/liuxuan/checkpoints/internimage/internimage_l_public_pretrain.pth \
  --codetr-weights /data1/liuxuan/checkpoints/codetr/codetr_internimage_l_public_pretrain.pth \
  --require-weights

if [ -f outputs/codetr_coco/fold0/annotations/instances_train2017.json ] && [ -f outputs/codetr_coco/fold0/annotations/instances_val2017.json ]; then
  echo "reuse existing COCO export: outputs/codetr_coco/fold0"
else
  python scripts/export_codetr_coco.py --fold 0 --output-root outputs/codetr_coco/fold0 --clip-labels
fi
PYTHONPATH={args.codetr_root}:${{PYTHONPATH:-}} python scripts/check_codetr_environment.py \
  --codetr-root {args.codetr_root} \
  --config {args.config}

PYTHONPATH={args.codetr_root}:${{PYTHONPATH:-}} python {args.codetr_root}/tools/train.py \
  {args.config} \
  --work-dir {args.work_dir} \
  {init_arg} \
  --cfg-options \
    {cfg_options_text}

BEST_CKPT=$(ls -t {args.work_dir}/best_bbox_mAP_epoch_*.pth 2>/dev/null | head -1)
if [ -z "$BEST_CKPT" ]; then
  BEST_CKPT=$(ls -t {args.work_dir}/epoch_*.pth 2>/dev/null | head -1)
fi
VAL_OUT="{args.work_dir}/final_txt_val_eval"
mkdir -p "$VAL_OUT"

PYTHONPATH={args.codetr_root}:${{PYTHONPATH:-}} python {args.codetr_root}/tools/test.py \
  {args.config} \
  "$BEST_CKPT" \
  --out "$VAL_OUT/results.pkl" \
  --eval bbox \
  --work-dir "$VAL_OUT" \
  --cfg-options data.test_dataloader.samples_per_gpu=1 data.test_dataloader.workers_per_gpu=0

PYTHONPATH=src python scripts/sweep_codetr_submission_params.py \
  --results-pkl "$VAL_OUT/results.pkl" \
  --coco-ann outputs/codetr_coco/fold0/annotations/instances_val2017.json \
  --labels source/训练集/labels \
  --sample-ids-file outputs/splits/fold0_val_ids.txt \
  --output-dir "$VAL_OUT/submission_contract_sweep" \
  --thresholds 0 0.001 0.003 0.005 0.01 0.02 0.03 0.05 \
  --max-detections 80 100 \
  --top-k 10
{allocation_block}
python - <<'PY'
import json, sys
from pathlib import Path
baseline = {args.baseline_val_map}
ranking = Path("{args.work_dir}/final_txt_val_eval/submission_contract_sweep/submission_param_ranking.json")
rows = json.loads(ranking.read_text())
best = rows[0]
print("best_submission_contract=", json.dumps(best, sort_keys=True))
if float(best["map_50_95"]) <= baseline:
    print(f"No test inference/promotion: strict mAP {{best['map_50_95']:.9f}} <= baseline {{baseline:.9f}}")
    sys.exit(0)
print(f"Candidate beats baseline {{baseline:.9f}}. Review outputs before test inference/promotion.")
PY
'''


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_script(args), encoding="utf-8")
    print(f"wrote: {args.output}")
    print("submit manually with:")
    print(f"  bsub < {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
