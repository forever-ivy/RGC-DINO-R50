#!/usr/bin/env python
"""Check whether the external IDEA-Research DINO tree is present and usable."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rgc_dino.dino_integration import DEFAULT_DINO_ROOT, check_dino_tree, format_status  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dino-root", type=Path, default=ROOT / DEFAULT_DINO_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    status = check_dino_tree(args.dino_root)
    print(format_status(status))
    return 0 if status.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
