#!/usr/bin/env python
"""Convert Co-DETR MMDetection outputs to competition TXT/ZIP submission files."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rgc_dino.constants import MAX_PREDICTIONS_PER_IMAGE, NUM_CLASSES  # noqa: E402
from rgc_dino.dataset import discover_aligned_samples  # noqa: E402
from rgc_dino.labels import DetectionLabel  # noqa: E402
from rgc_dino.submission import validate_submission_dir, write_submission_files, zip_submission_dir  # noqa: E402
from rgc_dino.submission_manifest import build_submission_manifest, write_submission_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "source" / "AIC2026_PHASE_1_1000")
    parser.add_argument("--coco-ann", type=Path, required=True)
    parser.add_argument("--results-pkl", type=Path, help="MMDetection results.pkl from tools/test.py")
    parser.add_argument("--bbox-json", type=Path, help="COCO bbox JSON from format_results")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--zip-path", type=Path)
    parser.add_argument("--manifest-path", type=Path, help="optional manifest JSON path; defaults to <zip>.manifest.json")
    parser.add_argument("--checkpoint-path", type=Path, help="checkpoint used to produce --results-pkl/--bbox-json")
    parser.add_argument("--config-path", type=Path, help="config used to produce --results-pkl/--bbox-json")
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "outputs" / "splits" / "split_manifest.json")
    parser.add_argument("--git-commit", default=None)
    parser.add_argument("--calibrator-version", default="none")
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--max-detections", type=int, default=MAX_PREDICTIONS_PER_IMAGE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (args.results_pkl is None) == (args.bbox_json is None):
        print("provide exactly one of --results-pkl or --bbox-json", file=sys.stderr)
        return 2

    images, id_to_image = _load_coco_images(args.coco_ann)
    samples = discover_aligned_samples(args.dataset_root)
    image_ids = [sample.sample_id for sample in samples]
    if len(images) != len(image_ids):
        print(f"warning: coco images {len(images)} != aligned samples {len(image_ids)}", file=sys.stderr)

    if args.bbox_json is not None:
        predictions = _predictions_from_bbox_json(
            args.bbox_json,
            id_to_image=id_to_image,
            score_threshold=args.score_threshold,
        )
    else:
        predictions = _predictions_from_results_pkl(
            args.results_pkl,
            images=images,
            score_threshold=args.score_threshold,
        )

    write_submission_files(
        image_ids,
        predictions,
        args.output_dir,
        max_predictions_per_image=args.max_detections,
    )
    errors = validate_submission_dir(image_ids, args.output_dir, max_predictions_per_image=args.max_detections)
    if errors:
        print("submission validation failed:", file=sys.stderr)
        for error in errors[:20]:
            print(f"  {error}", file=sys.stderr)
        return 1
    summary: dict[str, Any] = {
        "files": len(image_ids),
        "prediction_objects": sum(len(records) for records in predictions.values()),
        "output_dir": str(args.output_dir),
        "score_threshold": args.score_threshold,
        "max_detections": args.max_detections,
    }
    if args.zip_path is not None:
        zip_submission_dir(args.output_dir, args.zip_path)
        summary["zip_path"] = str(args.zip_path)
        if args.checkpoint_path is not None and args.config_path is not None and args.split_manifest.exists():
            git_commit = args.git_commit or _git_commit()
            manifest = build_submission_manifest(
                zip_path=args.zip_path,
                checkpoint_path=args.checkpoint_path,
                git_commit=git_commit,
                split_manifest_path=args.split_manifest,
                calibrator_version=args.calibrator_version,
                config_path=args.config_path,
            )
            manifest_path = args.manifest_path or args.zip_path.with_suffix(".manifest.json")
            write_submission_manifest(manifest_path, manifest)
            summary["manifest_path"] = str(manifest_path)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def _load_coco_images(coco_ann: Path) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    payload = json.loads(coco_ann.read_text(encoding="utf-8"))
    images = sorted(payload["images"], key=lambda item: int(item["id"]))
    id_to_image = {int(item["id"]): item for item in images}
    return images, id_to_image


def _predictions_from_bbox_json(
    bbox_json: Path,
    *,
    id_to_image: dict[int, dict[str, Any]],
    score_threshold: float,
) -> dict[str, list[DetectionLabel]]:
    records_by_id: dict[str, list[DetectionLabel]] = {}
    records = json.loads(bbox_json.read_text(encoding="utf-8"))
    for row in records:
        image = id_to_image.get(int(row["image_id"]))
        if image is None:
            continue
        sample_id = Path(str(image["file_name"])).stem
        label = _coco_bbox_to_label(
            class_id=int(row["category_id"]),
            bbox=row["bbox"],
            score=float(row["score"]),
            width=float(image["width"]),
            height=float(image["height"]),
            score_threshold=score_threshold,
        )
        if label is not None:
            records_by_id.setdefault(sample_id, []).append(label)
    return records_by_id


def _predictions_from_results_pkl(
    results_pkl: Path,
    *,
    images: list[dict[str, Any]],
    score_threshold: float,
) -> dict[str, list[DetectionLabel]]:
    with results_pkl.open("rb") as handle:
        results = pickle.load(handle)
    records_by_id: dict[str, list[DetectionLabel]] = {}
    for image, result in zip(images, results):
        sample_id = Path(str(image["file_name"])).stem
        width = float(image["width"])
        height = float(image["height"])
        image_records: list[DetectionLabel] = []
        if isinstance(result, tuple):
            result = result[0]
        for class_id, class_boxes in enumerate(result):
            if class_id >= NUM_CLASSES:
                continue
            for box in class_boxes:
                x1, y1, x2, y2, score = [float(value) for value in box[:5]]
                label = _xyxy_to_label(
                    class_id=class_id,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    score=score,
                    width=width,
                    height=height,
                    score_threshold=score_threshold,
                )
                if label is not None:
                    image_records.append(label)
        if image_records:
            records_by_id[sample_id] = image_records
    return records_by_id


def _coco_bbox_to_label(
    *,
    class_id: int,
    bbox: list[float],
    score: float,
    width: float,
    height: float,
    score_threshold: float,
) -> DetectionLabel | None:
    x, y, box_w, box_h = [float(value) for value in bbox]
    return _xyxy_to_label(
        class_id=class_id,
        x1=x,
        y1=y,
        x2=x + box_w,
        y2=y + box_h,
        score=score,
        width=width,
        height=height,
        score_threshold=score_threshold,
    )


def _xyxy_to_label(
    *,
    class_id: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    score: float,
    width: float,
    height: float,
    score_threshold: float,
) -> DetectionLabel | None:
    if not 0 <= class_id < NUM_CLASSES or score < score_threshold:
        return None
    x1 = _clip(x1, 0.0, width)
    x2 = _clip(x2, 0.0, width)
    y1 = _clip(y1, 0.0, height)
    y2 = _clip(y2, 0.0, height)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w <= 0.0 or box_h <= 0.0:
        return None
    return DetectionLabel(
        class_id=class_id,
        norm_center_x=_clip((x1 + box_w / 2.0) / width, 0.0, 1.0),
        norm_center_y=_clip((y1 + box_h / 2.0) / height, 0.0, 1.0),
        norm_w=_clip(box_w / width, 1e-12, 1.0),
        norm_h=_clip(box_h / height, 1e-12, 1.0),
        confidence=_clip(score, 0.0, 1.0),
    )


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _git_commit() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
