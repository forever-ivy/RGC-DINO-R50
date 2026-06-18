#!/usr/bin/env python
"""Install a user-provided Co-DETR archive into external/Co-DETR.

Use this when the server cannot clone GitHub directly.  The archive should be a
public Co-DETR source tree downloaded elsewhere and uploaded under /data1/liuxuan/.
This script extracts code only; it does not compile CUDA ops, download weights,
or start training.
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = ROOT / "external" / "Co-DETR"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help=".zip/.tar/.tar.gz public Co-DETR source archive")
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--force", action="store_true", help="replace an existing target directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.archive.exists():
        raise FileNotFoundError(args.archive)
    if args.target.exists():
        if not args.force:
            raise FileExistsError(f"target already exists; pass --force to replace: {args.target}")
        shutil.rmtree(args.target)

    with tempfile.TemporaryDirectory(dir=str(ROOT / "external" if (ROOT / "external").exists() else ROOT)) as tmp:
        tmp_path = Path(tmp)
        _extract(args.archive, tmp_path)
        source = _find_codetr_root(tmp_path)
        args.target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, args.target)

    print(f"installed: {args.target}")
    print("next:")
    print("  python scripts/prepare_codetr_training.py --no-require-weights")
    return 0


def _extract(archive: Path, output: Path) -> None:
    suffixes = ''.join(archive.suffixes).lower()
    if archive.suffix.lower() == '.zip':
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(output)
    elif suffixes.endswith('.tar.gz') or suffixes.endswith('.tgz') or archive.suffix.lower() == '.tar':
        with tarfile.open(archive) as handle:
            handle.extractall(output)
    else:
        raise ValueError(f"unsupported archive type: {archive}")


def _find_codetr_root(root: Path) -> Path:
    candidates = [root] + [path for path in root.rglob('*') if path.is_dir()]
    for candidate in candidates:
        if (
            (candidate / 'tools' / 'train.py').exists()
            and (candidate / 'tools' / 'test.py').exists()
            and (candidate / 'mmdet').exists()
        ):
            return candidate
    raise FileNotFoundError("archive does not contain a recognizable Co-DETR source tree")


if __name__ == "__main__":
    raise SystemExit(main())
