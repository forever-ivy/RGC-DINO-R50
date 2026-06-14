#!/usr/bin/env python
"""Run RGC-DINO inference and write competition TXT predictions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
DINO_ROOT = ROOT / "external" / "IDEA-Research-DINO"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(DINO_ROOT))
os.environ.setdefault("TORCH_HOME", "/data1/liuxuan/cache/torch")

from main import build_model_main, get_args_parser  # noqa: E402
from util.slconfig import SLConfig  # noqa: E402

from rgc_dino.dino_batch import collate_rgc_dino_batch  # noqa: E402
from rgc_dino.dino_dataset import MultimodalDinoInferenceDataset  # noqa: E402
from rgc_dino.dino_inference import ClasswiseScoreCalibrator, dino_result_to_detection_labels  # noqa: E402
from rgc_dino.dino_training import load_checkpoint_into_model  # noqa: E402
from rgc_dino.models.rgc_dino_adapter import RgcDinoModel  # noqa: E402
from rgc_dino.sample_ids import load_sample_ids_file  # noqa: E402
from rgc_dino.submission import validate_submission_dir, write_submission_files, zip_submission_dir  # noqa: E402
from rgc_dino.submission_manifest import build_submission_manifest, write_submission_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", type=Path, default=ROOT / "configs" / "dino_a0_rgb_4scale.py")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "source" / "AIC2026_PHASE_1_1000")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-mode", choices=("rgc", "rgb"), default="rgc")
    parser.add_argument("--checkpoint-scope", choices=("auto", "rgc", "dino"), default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--zip-path", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-max-side", type=int, default=640)
    parser.add_argument("--side-base-channels", type=int, default=32)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--nms-iou-threshold", type=float, help="optional classwise NMS IoU threshold")
    parser.add_argument("--score-calibrator", type=Path, help="optional JSON classwise score calibrator")
    parser.add_argument("--manifest-path", type=Path, help="optional submission manifest JSON path")
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "outputs" / "splits" / "split_manifest.json")
    parser.add_argument("--sample-ids-file", type=Path, help="optional newline-delimited sample IDs to predict")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not DINO_ROOT.exists():
        print(f"external DINO tree not found: {DINO_ROOT}", file=sys.stderr)
        return 2

    device = torch.device(args.device)
    official_args = _build_official_args(args)
    model, _criterion, postprocessors = build_model_main(official_args)
    if args.model_mode == "rgc":
        detector = RgcDinoModel(model, side_base_channels=args.side_base_channels)
        report = _load_checkpoint(detector, args.checkpoint, scope=args.checkpoint_scope)
    else:
        detector = model
        report = _load_base_checkpoint(detector, args.checkpoint)
    print(json.dumps(report, sort_keys=True))
    detector.to(device)
    detector.eval()
    score_calibrator = (
        ClasswiseScoreCalibrator.from_path(args.score_calibrator)
        if args.score_calibrator is not None
        else None
    )

    sample_ids = load_sample_ids_file(args.sample_ids_file) if args.sample_ids_file is not None else None
    dataset = MultimodalDinoInferenceDataset.from_paths(
        dataset_root=args.dataset_root,
        sample_ids=sample_ids,
        image_max_side=args.image_max_side,
    )
    if args.limit is not None:
        dataset = MultimodalDinoInferenceDataset(dataset.samples[: args.limit], image_max_side=args.image_max_side)
    image_ids = [sample.sample_id for sample in dataset.samples]
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_rgc_dino_batch,
        num_workers=args.num_workers,
    )

    predictions = {}
    with torch.no_grad():
        for samples, targets in loader:
            samples = samples.to(device)
            target_sizes = torch.stack([target["orig_size"] for target in targets]).to(device)
            with torch.cuda.amp.autocast(enabled=args.amp):
                outputs = detector(samples.rgb if args.model_mode == "rgb" else samples)
            results = postprocessors["bbox"](outputs, target_sizes)
            for target, result in zip(targets, results):
                orig_height, orig_width = [int(value) for value in target["orig_size"].tolist()]
                predictions[str(target["sample_id"])] = dino_result_to_detection_labels(
                    result,
                    orig_height=orig_height,
                    orig_width=orig_width,
                    score_threshold=args.score_threshold,
                    max_detections=args.max_detections,
                    nms_iou_threshold=args.nms_iou_threshold,
                    score_calibrator=score_calibrator,
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

    summary = {
        "files": len(image_ids),
        "prediction_objects": sum(len(records) for records in predictions.values()),
        "output_dir": str(args.output_dir),
    }
    if args.zip_path is not None:
        zip_submission_dir(args.output_dir, args.zip_path)
        summary["zip_path"] = str(args.zip_path)
        manifest_path = args.manifest_path or args.zip_path.with_suffix(".manifest.json")
        if args.split_manifest.exists():
            manifest = build_submission_manifest(
                zip_path=args.zip_path,
                checkpoint_path=args.checkpoint,
                git_commit=_git_commit(),
                split_manifest_path=args.split_manifest,
                calibrator_version=_calibrator_version(args.score_calibrator),
                config_path=args.config_file,
            )
            write_submission_manifest(manifest_path, manifest)
            summary["manifest_path"] = str(manifest_path)
    print(json.dumps(summary, sort_keys=True))
    return 0


def _build_official_args(args: argparse.Namespace) -> argparse.Namespace:
    parser = get_args_parser()
    official_args = parser.parse_args(
        [
            "--config_file",
            str(args.config_file),
            "--dataset_file",
            "coco",
            "--coco_path",
            str(ROOT / "outputs" / "dino_a0_rgb" / "fold0_coco"),
            "--output_dir",
            str(args.output_dir),
            "--device",
            args.device,
            "--num_workers",
            str(args.num_workers),
        ]
        + (["--amp"] if args.amp else [])
    )
    cfg = SLConfig.fromfile(str(args.config_file))
    for key, value in cfg._cfg_dict.to_dict().items():
        if hasattr(official_args, key):
            raise ValueError(f"config key collides with official CLI arg: {key}")
        setattr(official_args, key, value)
    official_args.distributed = False
    official_args.rank = 0
    official_args.local_rank = None
    official_args.gpu = 0
    official_args.use_ema = False
    official_args.debug = False
    official_args.save_results = False
    official_args.save_log = False
    return official_args


def _load_checkpoint(model: RgcDinoModel, checkpoint_path: Path, *, scope: str) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("checkpoint payload must be a dictionary")

    model_state = payload.get("model", payload)
    if scope == "auto":
        if isinstance(model_state, dict) and _looks_like_rgc_state(model_state):
            scope = "rgc"
        elif "dino_model" in payload:
            scope = "dino_model_payload"
        else:
            scope = "dino"

    if scope == "rgc":
        report = load_checkpoint_into_model(
            model,
            checkpoint_path,
            skip_mismatched_shapes=True,
            weights_only=False,
        )
        return {
            "checkpoint": str(report.checkpoint_path),
            "loaded_scope": scope,
            "missing_keys": len(report.missing_keys),
            "skipped_shape_mismatch_keys": len(report.skipped_keys),
            "unexpected_keys": len(report.unexpected_keys),
        }
    elif scope == "dino_model_payload":
        incompatible = model.dino_model.load_state_dict(payload["dino_model"], strict=False)
    else:
        incompatible = model.dino_model.load_state_dict(model_state, strict=False)

    return {
        "checkpoint": str(checkpoint_path),
        "loaded_scope": scope,
        "missing_keys": len(incompatible.missing_keys),
        "unexpected_keys": len(incompatible.unexpected_keys),
    }


def _load_base_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("checkpoint payload must be a dictionary")
    state = payload.get("model", payload.get("dino_model", payload))
    incompatible = model.load_state_dict(state, strict=False)
    return {
        "checkpoint": str(checkpoint_path),
        "loaded_scope": "dino",
        "missing_keys": len(incompatible.missing_keys),
        "unexpected_keys": len(incompatible.unexpected_keys),
    }


def _looks_like_rgc_state(state: dict[str, Any]) -> bool:
    return any(key.startswith(("dino_model.", "feature_fusion.")) for key in state)


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


def _calibrator_version(path: Path | None) -> str:
    if path is None:
        return "none"
    from rgc_dino.submission_manifest import file_sha256

    return f"{path.name}:{file_sha256(path)}"


if __name__ == "__main__":
    raise SystemExit(main())
