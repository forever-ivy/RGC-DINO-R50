#!/usr/bin/env python
"""Print a lightweight runtime environment summary."""

from __future__ import annotations

import importlib
import platform
import sys


def main() -> int:
    print(f"python: {sys.version.split()[0]}")
    print(f"platform: {platform.platform()}")

    torch = importlib.import_module("torch")
    print(f"torch: {torch.__version__}")
    print(f"cuda build: {torch.version.cuda}")
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"cuda device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"cuda device 0: {torch.cuda.get_device_name(0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

