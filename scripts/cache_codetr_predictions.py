#!/usr/bin/env python
"""Cache Co-DETR validation/test predictions as JSON rows for postprocess sweeps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from codetr_results_to_submission import (  # noqa: E402
    _load_coco_images,
    _predictions_from_bbox_json,
    _predictions_from_results_pkl,
)
from rgc_dino.labels import DetectionLabel  # noqa: E402
from rgc_dino.postprocess import summarize_predictions  # noqa: E402
from rgc_dino.sample_ids import load_sample_ids_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--results-pkl", type=Path, help="MMDetection results.pkl from tools/test.py")
    source.add_argument("--bbox-json", type=Path, help="COCO bbox JSON from format_results")
    parser.add_argument("--coco-ann", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-threshold", type=float, default=0.0)
    parser.add_argument("--sample-ids-file", type=Path, help="optional newline-delimited sample IDs to keep")
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--config-path", type=Path)
    parser.add_argument("--image-max-side", type=int)
    parser.add_argument("--nms-iou-threshold", type=float)
    parser.add_argument("--metadata", action="append", default=[], help="extra key=value metadata entries")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    images, id_to_image = _load_coco_images(args.coco_ann)
    image_ids = [Path(str(image["file_name"])).stem for image in images]
    if args.sample_ids_file is not None:
        keep_ids = set(load_sample_ids_file(args.sample_ids_file))
        image_ids = [sample_id for sample_id in image_ids if sample_id in keep_ids]
    else:
        keep_ids = set(image_ids)

    if args.bbox_json is not None:
        predictions = _predictions_from_bbox_json(
            args.bbox_json,
            id_to_image=id_to_image,
            score_threshold=args.candidate_threshold,
        )
    else:
        predictions = _predictions_from_results_pkl(
            args.results_pkl,
            images=images,
            score_threshold=args.candidate_threshold,
        )
    predictions = {sample_id: records for sample_id, records in predictions.items() if sample_id in keep_ids}

    rows = []
    for sample_id in image_ids:
        for record in predictions.get(sample_id, []):
            rows.append(_record_to_row(sample_id, record))

    metadata = {
        "results_pkl": str(args.results_pkl) if args.results_pkl is not None else None,
        "bbox_json": str(args.bbox_json) if args.bbox_json is not None else None,
        "coco_ann": str(args.coco_ann),
        "candidate_threshold": args.candidate_threshold,
        "checkpoint_path": str(args.checkpoint_path) if args.checkpoint_path is not None else None,
        "config_path": str(args.config_path) if args.config_path is not None else None,
        "image_max_side": args.image_max_side,
        "nms_iou_threshold": args.nms_iou_threshold,
        "sample_ids_file": str(args.sample_ids_file) if args.sample_ids_file is not None else None,
    }
    metadata.update(_parse_metadata(args.metadata))
    payload = {
        "metadata": metadata,
        "summary": summarize_predictions(predictions, image_ids=image_ids),
        "image_ids": image_ids,
        "predictions": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "predictions": len(rows), "images": len(image_ids)}, sort_keys=True))
    return 0


def _record_to_row(sample_id: str, record: DetectionLabel) -> dict[str, float | int | str]:
    return {
        "sample_id": sample_id,
        "class_id": record.class_id,
        "norm_center_x": record.norm_center_x,
        "norm_center_y": record.norm_center_y,
        "norm_w": record.norm_w,
        "norm_h": record.norm_h,
        "confidence": record.confidence if record.confidence is not None else 0.0,
    }


def _parse_metadata(items: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"metadata must be key=value, got: {item}")
        key, value = item.split("=", maxsplit=1)
        parsed[key] = value
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
