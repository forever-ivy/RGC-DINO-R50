#!/usr/bin/env python
"""Write an LSF script for Co-DETR stage-0/stage-1 training.

The generated job performs preflight checks and COCO export, then runs the
external Co-DETR training entrypoint.  This script does not submit the job.
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "jobs" / "codetr_r50_stage0_train.lsf"
DEFAULT_CODETR_ROOT = Path("external/Co-DETR")
DEFAULT_CONFIG = ROOT / "configs" / "codetr_r50_stage0_mm_config.py"
DEFAULT_COCO_OUTPUT = ROOT / "outputs" / "codetr_coco" / "fold0"
DEFAULT_WORK_DIR = ROOT / "outputs" / "codetr" / "r50_stage0_fold0"
DEFAULT_INTERNIMAGE_WEIGHTS = Path("/data1/liuxuan/checkpoints/internimage/internimage_l_public_pretrain.pth")
DEFAULT_CODETR_WEIGHTS = Path("/data1/liuxuan/checkpoints/codetr/codetr_internimage_l_public_pretrain.pth")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--job-name", default="codetr-intl-stage0")
    parser.add_argument("--queue", default="normal")
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--codetr-root", type=Path, default=DEFAULT_CODETR_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--coco-output", type=Path, default=DEFAULT_COCO_OUTPUT)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--internimage-weights", type=Path, default=DEFAULT_INTERNIMAGE_WEIGHTS)
    parser.add_argument("--codetr-weights", type=Path, default=DEFAULT_CODETR_WEIGHTS)
    parser.add_argument("--require-weights", action="store_true", default=False)
    parser.add_argument("--no-require-weights", action="store_false", dest="require_weights")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--env-prefix", type=Path, default=Path("/data1/liuxuan/envs/codetr"))
    return parser.parse_args()


def render_bsub_script(
    *,
    job_name: str,
    queue: str,
    gpu: int,
    fold: int,
    codetr_root: Path,
    config: Path,
    coco_output: Path,
    work_dir: Path,
    internimage_weights: Path,
    codetr_weights: Path,
    require_weights: bool,
    num_workers: int,
    env_prefix: Path,
) -> str:
    require_flag = " --require-weights" if require_weights else ""
    cfg_options = f"--cfg-options data.workers_per_gpu={num_workers}"
    if gpu <= 1:
        train_cmd = (
            f"PYTHONPATH={codetr_root}:$PYTHONPATH python {codetr_root}/tools/train.py "
            f"{config} --work-dir {work_dir} {cfg_options}"
        )
    else:
        train_cmd = (
            f"PYTHONPATH={codetr_root}:$PYTHONPATH bash {codetr_root}/tools/dist_train.sh "
            f"{config} {gpu} {work_dir} {cfg_options}"
        )
    return f"""#!/bin/sh
#BSUB -J {job_name}
#BSUB -q {queue}
#BSUB -n {max(2, gpu * 2)}
#BSUB -gpu "num={gpu}:mode=exclusive_process"
#BSUB -R "rusage[mem=32000]"
#BSUB -o /data1/liuxuan/logs/{job_name}.%J.out
#BSUB -e /data1/liuxuan/logs/{job_name}.%J.err

set -eu
cd {ROOT}
. /data1/miniconda3/etc/profile.d/conda.sh
conda activate {env_prefix}

mkdir -p /data1/liuxuan/logs {work_dir}

python scripts/check_codetr_integration.py \
  --codetr-root {codetr_root} \
  --internimage-weights {internimage_weights} \
  --codetr-weights {codetr_weights}{require_flag}

python scripts/export_codetr_coco.py \
  --fold {fold} \
  --output-root {coco_output} \
  --clip-labels

python scripts/check_codetr_environment.py \
  --codetr-root {codetr_root} \
  --config {config}

# Keep all heavy work in this batch job, not the interactive shell.
# Keep CPU pressure low so Claude Code and cc-switch.tmux remain usable.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
# Stage-0 config is RGB COCO sanity first; RGC tri-modal Co-DETR adapter comes next.
{train_cmd}
"""


def main() -> int:
    args = parse_args()
    script = render_bsub_script(
        job_name=args.job_name,
        queue=args.queue,
        gpu=args.gpu,
        fold=args.fold,
        codetr_root=args.codetr_root,
        config=args.config,
        coco_output=args.coco_output,
        work_dir=args.work_dir,
        internimage_weights=args.internimage_weights,
        codetr_weights=args.codetr_weights,
        require_weights=args.require_weights,
        num_workers=args.num_workers,
        env_prefix=args.env_prefix,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(script, encoding="utf-8")
    print(f"wrote: {args.output}")
    print("submit manually with:")
    print(f"  bsub < {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
