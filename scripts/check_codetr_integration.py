#!/usr/bin/env python
"""Check whether a local Co-DETR tree is present and usable."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rgc_dino.codetr_integration import DEFAULT_CODETR_ROOT, check_codetr_tree, format_status  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codetr-root", type=Path, default=ROOT / DEFAULT_CODETR_ROOT)
    parser.add_argument(
        "--internimage-weights",
        type=Path,
        help="optional public InternImage-L pretrained checkpoint path to check",
    )
    parser.add_argument(
        "--codetr-weights",
        type=Path,
        help="optional public Co-DETR/Co-DINO pretrained checkpoint path to check",
    )
    parser.add_argument(
        "--require-weights",
        action="store_true",
        help="fail if any supplied pretrained-weight path is missing",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    weights = [path for path in (args.internimage_weights, args.codetr_weights) if path is not None]
    status = check_codetr_tree(
        args.codetr_root,
        weight_paths=weights,
        require_weights=args.require_weights,
    )
    print(format_status(status))
    return 0 if status.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
