#!/usr/bin/env python
"""Write an LSF script for Co-DETR integration smoke checks.

This script only generates a job file.  It does not submit the job and it does
not start heavy training from the interactive session.
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "jobs" / "codetr_smoke.lsf"
DEFAULT_CODETR_ROOT = Path("external/Co-DETR")
DEFAULT_CONFIG = ROOT / "configs" / "codetr_internimage_l_mm_config.py"
DEFAULT_COCO_OUTPUT = ROOT / "outputs" / "codetr_coco" / "fold0"
DEFAULT_INTERNIMAGE_WEIGHTS = Path("/data1/liuxuan/checkpoints/internimage/internimage_l_public_pretrain.pth")
DEFAULT_CODETR_WEIGHTS = Path("/data1/liuxuan/checkpoints/codetr/codetr_internimage_l_public_pretrain.pth")
DEFAULT_ENV_PREFIX = Path("/data1/liuxuan/envs/codetr")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--job-name", default="codetr-smoke")
    parser.add_argument("--queue", default="normal")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--codetr-root", type=Path, default=DEFAULT_CODETR_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--coco-output", type=Path, default=DEFAULT_COCO_OUTPUT)
    parser.add_argument("--internimage-weights", type=Path, default=DEFAULT_INTERNIMAGE_WEIGHTS)
    parser.add_argument("--codetr-weights", type=Path, default=DEFAULT_CODETR_WEIGHTS)
    parser.add_argument("--env-prefix", type=Path, default=DEFAULT_ENV_PREFIX)
    parser.add_argument(
        "--require-weights",
        action="store_true",
        help="make the smoke job fail if the supplied public pretrained weights are missing",
    )
    return parser.parse_args()


def render_bsub_script(
    *,
    job_name: str,
    queue: str,
    gpu: int,
    codetr_root: Path,
    config: Path,
    coco_output: Path,
    internimage_weights: Path,
    codetr_weights: Path,
    require_weights: bool,
    env_prefix: Path,
) -> str:
    require_flag = " --require-weights" if require_weights else ""
    return f"""#!/bin/sh
#BSUB -J {job_name}
#BSUB -q {queue}
#BSUB -n 2
#BSUB -gpu "num={gpu}:mode=exclusive_process"
#BSUB -R "rusage[mem=12000]"
#BSUB -o /data1/liuxuan/logs/{job_name}.%J.out
#BSUB -e /data1/liuxuan/logs/{job_name}.%J.err

set -eu
cd {ROOT}
. /data1/miniconda3/etc/profile.d/conda.sh
conda activate {env_prefix}

mkdir -p /data1/liuxuan/logs {ROOT / 'outputs' / 'codetr'}

python scripts/check_codetr_batch_smoke.py \
  --codetr-root {codetr_root} \
  --config {config} \
  --coco-root {coco_output} \
  --internimage-weights {internimage_weights} \
  --codetr-weights {codetr_weights} \
  --require-cuda{require_flag}

echo "Co-DETR batch smoke completed. This job checks config/model/DCNv3/load_from; it does not train."
"""


def main() -> int:
    args = parse_args()
    script = render_bsub_script(
        job_name=args.job_name,
        queue=args.queue,
        gpu=args.gpu,
        codetr_root=args.codetr_root,
        config=args.config,
        coco_output=args.coco_output,
        internimage_weights=args.internimage_weights,
        codetr_weights=args.codetr_weights,
        require_weights=args.require_weights,
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
