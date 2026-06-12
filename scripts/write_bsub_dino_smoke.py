#!/usr/bin/env python
"""Write an LSF script for DINO integration smoke checks without submitting it."""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "jobs" / "dino_smoke.lsf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--job-name", default="dino-smoke")
    parser.add_argument("--queue", default="normal")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--dino-root", type=Path, default=Path("external/IDEA-Research-DINO"))
    return parser.parse_args()


def render_bsub_script(*, job_name: str, queue: str, gpu: int, dino_root: Path) -> str:
    return f"""#!/bin/sh
#BSUB -J {job_name}
#BSUB -q {queue}
#BSUB -n 2
#BSUB -gpu "num={gpu}:mode=exclusive_process"
#BSUB -R "rusage[mem=8000]"
#BSUB -o /data1/liuxuan/logs/{job_name}.%J.out
#BSUB -e /data1/liuxuan/logs/{job_name}.%J.err

set -eu
cd {ROOT}
. /data1/liuxuan/activate-py310.sh

python scripts/check_dino_integration.py --dino-root {dino_root}
echo "DINO integration smoke check completed. Compile CUDA ops manually only after approval."
"""


def main() -> int:
    args = parse_args()
    script = render_bsub_script(
        job_name=args.job_name,
        queue=args.queue,
        gpu=args.gpu,
        dino_root=args.dino_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(script, encoding="utf-8")
    print(f"wrote: {args.output}")
    print("submit manually with:")
    print(f"  bsub < {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
