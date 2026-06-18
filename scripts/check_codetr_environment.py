#!/usr/bin/env python
"""Check Python dependencies needed before running external Co-DETR training."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CODETR_ROOT = ROOT / "external" / "Co-DETR"
DEFAULT_CONFIG = ROOT / "configs" / "codetr_r50_stage0_mm_config.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codetr-root", type=Path, default=DEFAULT_CODETR_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.codetr_root.exists():
        sys.path.insert(0, str(args.codetr_root))

    checks = []
    ok = True
    for module_name in ("torch", "mmcv", "mmdet"):
        try:
            module = importlib.import_module(module_name)
            checks.append(
                {
                    "module": module_name,
                    "ok": True,
                    "version": str(getattr(module, "__version__", "unknown")),
                }
            )
        except Exception as exc:  # pragma: no cover - exercised in real env
            ok = False
            checks.append({"module": module_name, "ok": False, "error": repr(exc)})

    config_ok = False
    config_error = None
    if ok:
        try:
            from mmcv import Config

            cfg = Config.fromfile(str(args.config))
            config_ok = bool(getattr(cfg, "model", None))
        except Exception as exc:  # pragma: no cover - depends on external env
            config_error = repr(exc)
            ok = False
    else:
        config_error = "skipped because dependency imports failed"

    result = {
        "ok": ok,
        "checks": checks,
        "codetr_root": str(args.codetr_root),
        "config": str(args.config),
        "config_ok": config_ok,
        "config_error": config_error,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
