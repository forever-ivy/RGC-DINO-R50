#!/usr/bin/env python
"""Promote a deliberate test-set prediction ZIP into outputs/submissions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rgc_dino.submission_promotion import promote_submission_candidate  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_zip", type=Path)
    parser.add_argument("--submissions-dir", type=Path, default=ROOT / "outputs" / "submissions")
    parser.add_argument("--reason", required=True)
    parser.add_argument("--local-map", type=float)
    parser.add_argument("--leaderboard-baseline", type=float)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = promote_submission_candidate(
        candidate_zip=args.candidate_zip,
        submissions_dir=args.submissions_dir,
        reason=args.reason,
        local_map=args.local_map,
        leaderboard_baseline=args.leaderboard_baseline,
        force=args.force,
    )
    print(
        json.dumps(
            {
                "promoted_zip": str(result.zip_path),
                "metadata_path": str(result.metadata_path),
                "zip_sha256": result.zip_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
