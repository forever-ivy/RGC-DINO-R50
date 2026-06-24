#!/usr/bin/env python
"""Build a deterministic hard-validation subset from existing validation labels.

The subset is a diagnostic gate for postprocess sweeps. It uses only local
training labels and optional cached validation predictions; it does not touch the
test set or create pseudo-labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rgc_dino.constants import CLASS_NAMES, NUM_CLASSES  # noqa: E402
from rgc_dino.labels import DetectionLabel, load_label_dir  # noqa: E402
from rgc_dino.sample_ids import load_sample_ids_file, restrict_mapping_to_sample_ids  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=ROOT / "source" / "训练集" / "labels")
    parser.add_argument("--sample-ids-file", type=Path, help="validation sample IDs to choose from")
    parser.add_argument("--prediction-cache", type=Path, help="optional prediction cache from cache_codetr_predictions.py")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "codetr" / "phase35_hard_val")
    parser.add_argument("--output-ids", type=Path, help="optional output path for selected sample IDs")
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--fraction", type=float, default=0.25, help="maximum fraction of available samples to select")
    parser.add_argument("--small-area-threshold", type=float, default=0.01)
    parser.add_argument("--rare-class-count", type=int, default=4, help="number of least frequent non-empty classes treated as rare")
    parser.add_argument("--clip-labels", action="store_true", default=True)
    parser.add_argument("--no-clip-labels", dest="clip_labels", action="store_false")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    if not 0.0 < args.fraction <= 1.0:
        raise ValueError("--fraction must be in (0, 1]")

    labels = load_label_dir(args.labels, clip=args.clip_labels)
    if args.sample_ids_file is not None:
        sample_ids = load_sample_ids_file(args.sample_ids_file)
        labels = restrict_mapping_to_sample_ids(labels, sample_ids)
    else:
        sample_ids = sorted(labels)
    sample_ids = [sample_id for sample_id in sample_ids if sample_id in labels]
    if not sample_ids:
        print("no labeled samples available for hard-val selection", file=sys.stderr)
        return 2

    prediction_counts = _load_prediction_counts(args.prediction_cache) if args.prediction_cache is not None else {}
    rare_classes = _rare_classes(labels, sample_ids, rare_class_count=args.rare_class_count)
    rows = [
        _score_sample(
            sample_id,
            labels.get(sample_id, []),
            prediction_count=prediction_counts.get(sample_id, 0),
            rare_classes=rare_classes,
            small_area_threshold=args.small_area_threshold,
        )
        for sample_id in sample_ids
    ]
    selected_count = min(args.max_samples, max(1, int(round(len(rows) * args.fraction))))
    rows.sort(key=lambda row: (-row["hard_score"], row["sample_id"]))
    selected_rows = rows[:selected_count]
    selected_ids = [row["sample_id"] for row in selected_rows]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ids_path = args.output_ids or (args.output_dir / "hard_val_sample_ids.txt")
    ids_path.parent.mkdir(parents=True, exist_ok=True)
    ids_path.write_text("\n".join(selected_ids) + "\n", encoding="utf-8")

    manifest = {
        "sample_ids_file": str(ids_path),
        "labels": str(args.labels),
        "source_sample_ids_file": str(args.sample_ids_file) if args.sample_ids_file is not None else None,
        "prediction_cache": str(args.prediction_cache) if args.prediction_cache is not None else None,
        "available_samples": len(rows),
        "selected_samples": len(selected_ids),
        "fraction": args.fraction,
        "max_samples": args.max_samples,
        "small_area_threshold": args.small_area_threshold,
        "rare_classes": {str(class_id): CLASS_NAMES[class_id] for class_id in rare_classes},
        "selected": selected_rows,
        "score_distribution": _score_distribution(rows),
    }
    (args.output_dir / "hard_val_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(args.output_dir / "hard_val_selection_report.md", manifest)
    print(json.dumps({"sample_ids_file": str(ids_path), "selected_samples": len(selected_ids)}, sort_keys=True))
    return 0


def _rare_classes(labels: dict[str, list[DetectionLabel]], sample_ids: list[str], *, rare_class_count: int) -> set[int]:
    counts = {class_id: 0 for class_id in range(NUM_CLASSES)}
    for sample_id in sample_ids:
        for label in labels.get(sample_id, []):
            counts[label.class_id] += 1
    nonzero = [(count, class_id) for class_id, count in counts.items() if count > 0]
    nonzero.sort(key=lambda item: (item[0], item[1]))
    return {class_id for _count, class_id in nonzero[: max(0, rare_class_count)]}


def _score_sample(
    sample_id: str,
    labels: list[DetectionLabel],
    *,
    prediction_count: int,
    rare_classes: set[int],
    small_area_threshold: float,
) -> dict[str, Any]:
    object_count = len(labels)
    small_object_count = sum(1 for label in labels if label.norm_w * label.norm_h <= small_area_threshold)
    rare_object_count = sum(1 for label in labels if label.class_id in rare_classes)
    person_count = sum(1 for label in labels if label.class_id == 0)
    crowded_person = person_count >= 5
    class_ids = sorted({label.class_id for label in labels})
    # Keep the formula simple and auditable. Prediction pressure is capped so a
    # noisy cache cannot dominate actual label-based hard cases.
    hard_score = (
        object_count
        + 2.0 * small_object_count
        + 2.0 * rare_object_count
        + (3.0 if crowded_person else 0.0)
        + min(prediction_count / 25.0, 4.0)
    )
    return {
        "sample_id": sample_id,
        "hard_score": hard_score,
        "object_count": object_count,
        "small_object_count": small_object_count,
        "rare_object_count": rare_object_count,
        "person_count": person_count,
        "crowded_person": crowded_person,
        "prediction_count": prediction_count,
        "class_ids": class_ids,
    }


def _load_prediction_counts(path: Path | None) -> dict[str, int]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("predictions", payload)
    counts: dict[str, int] = {}
    if isinstance(rows, dict):
        for sample_id, records in rows.items():
            counts[str(sample_id)] = len(records)
        return counts
    for row in rows:
        sample_id = str(row["sample_id"])
        counts[sample_id] = counts.get(sample_id, 0) + 1
    return counts


def _score_distribution(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    scores = [float(row["hard_score"]) for row in rows]
    return {"min": min(scores), "mean": sum(scores) / len(scores), "max": max(scores)}


def _write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Hard validation subset",
        "",
        f"- available samples: `{manifest['available_samples']}`",
        f"- selected samples: `{manifest['selected_samples']}`",
        f"- sample IDs file: `{manifest['sample_ids_file']}`",
        f"- rare classes: `{manifest['rare_classes']}`",
        "",
        "## Top selected samples",
        "",
        "| sample_id | score | objects | small | rare | person | pred_count | classes |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in manifest["selected"][:50]:
        lines.append(
            f"| {row['sample_id']} | {row['hard_score']:.3f} | {row['object_count']} | {row['small_object_count']} | "
            f"{row['rare_object_count']} | {row['person_count']} | {row['prediction_count']} | {row['class_ids']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
