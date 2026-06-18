#!/usr/bin/env python
"""Fail fast on suspicious pretrained-weight load reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rgc_dino.weight_audit import audit_load_report  # noqa: E402


DEFAULT_FATAL_PATTERNS = (
    "patch_embed",
    "backbone.patch",
    "backbone.stages",
    "backbone.levels",
    "backbone.layers",
    "stages.",
    "levels.",
    "layers.",
)
DEFAULT_ALLOWED_PATTERNS = (
    "class_embed",
    "label_enc",
    "enc_out_class_embed",
    "bbox_head.fc_cls",
    "roi_head.bbox_head",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="JSON load report with missing_keys/skipped_keys arrays")
    parser.add_argument("--fatal-pattern", action="append", dest="fatal_patterns")
    parser.add_argument("--allow-pattern", action="append", dest="allowed_patterns")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.report.read_text(encoding="utf-8"))
    fatal_patterns = tuple(args.fatal_patterns) if args.fatal_patterns is not None else DEFAULT_FATAL_PATTERNS
    allowed_patterns = tuple(args.allowed_patterns) if args.allowed_patterns is not None else DEFAULT_ALLOWED_PATTERNS
    audit = audit_load_report(
        missing_keys=payload.get("missing_keys", ()),
        skipped_keys=payload.get("skipped_keys", payload.get("skipped_shape_mismatch_keys", ())),
        fatal_missing_patterns=fatal_patterns,
        fatal_skipped_patterns=fatal_patterns,
        allowed_missing_patterns=allowed_patterns,
    )
    print(
        json.dumps(
            {
                "ok": audit.ok,
                "fatal_missing_count": len(audit.fatal_missing_keys),
                "fatal_skipped_count": len(audit.fatal_skipped_keys),
                "messages": audit.messages,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if audit.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
