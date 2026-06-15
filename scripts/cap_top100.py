#!/usr/bin/env python3
"""将预测目录中每个文件按 score（最后一列）降序截断到 top-100。"""
import argparse
from pathlib import Path


def cap_file(path: Path, top_k: int) -> int:
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    if len(lines) <= top_k:
        return len(lines)
    # 末列为 score
    lines.sort(key=lambda ln: float(ln.split()[-1]), reverse=True)
    kept = lines[:top_k]
    path.write_text("\n".join(kept) + "\n")
    return len(kept)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True, type=Path)
    ap.add_argument("--top-k", type=int, default=100)
    args = ap.parse_args()

    files = sorted(args.pred_dir.glob("*.txt"))
    capped = 0
    for f in files:
        before = len([ln for ln in f.read_text().splitlines() if ln.strip()])
        after = cap_file(f, args.top_k)
        if before > after:
            capped += 1
    print(f"处理 {len(files)} 个文件，截断 {capped} 个到 top-{args.top_k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
