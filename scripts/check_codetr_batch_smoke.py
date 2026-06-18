#!/usr/bin/env python
"""Run a no-training batch smoke check for the Co-DETR + InternImage route."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rgc_dino.codetr_integration import check_codetr_tree  # noqa: E402

DEFAULT_CODETR_ROOT = ROOT / "external" / "Co-DETR"
DEFAULT_CONFIG = ROOT / "configs" / "codetr_internimage_l_mm_config.py"
DEFAULT_COCO_ROOT = ROOT / "outputs" / "codetr_coco" / "fold0"
DEFAULT_INTERNIMAGE_WEIGHTS = Path("/data1/liuxuan/checkpoints/internimage/internimage_l_public_pretrain.pth")
DEFAULT_CODETR_WEIGHTS = Path("/data1/liuxuan/checkpoints/codetr/codetr_internimage_l_public_pretrain.pth")
DEFAULT_LOG_FILE = ROOT / "outputs" / "codetr" / "codetr_batch_smoke_model.log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codetr-root", type=Path, default=DEFAULT_CODETR_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--coco-root", type=Path, default=DEFAULT_COCO_ROOT)
    parser.add_argument("--internimage-weights", type=Path, default=DEFAULT_INTERNIMAGE_WEIGHTS)
    parser.add_argument("--codetr-weights", type=Path, default=DEFAULT_CODETR_WEIGHTS)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE)
    parser.add_argument("--require-weights", action="store_true")
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="fail if CUDA is unavailable or the tiny DCNv3 CUDA forward cannot run",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    status = check_codetr_tree(
        args.codetr_root,
        weight_paths=[args.internimage_weights, args.codetr_weights],
        require_weights=args.require_weights,
    )
    if not status.ok:
        print(json.dumps({"ok": False, "messages": status.messages}, ensure_ascii=False, indent=2))
        return 2

    coco_ready = _check_coco_export(args.coco_root)
    if not coco_ready["ok"]:
        print(json.dumps({"ok": False, **coco_ready}, ensure_ascii=False, indent=2, sort_keys=True))
        return 2

    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    with args.log_file.open("w", encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log):
        summary = _run_model_smoke(args)

    result = {
        "ok": True,
        "codetr_root": str(args.codetr_root),
        "config": str(args.config),
        "coco_root": str(args.coco_root),
        "model_log": str(args.log_file),
        **coco_ready,
        **summary,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _check_coco_export(coco_root: Path) -> dict[str, Any]:
    train_ann = coco_root / "annotations" / "instances_train2017.json"
    val_ann = coco_root / "annotations" / "instances_val2017.json"
    train_dir = coco_root / "train2017"
    val_dir = coco_root / "val2017"
    paths = (train_ann, val_ann, train_dir, val_dir)
    missing = [str(path) for path in paths if not path.exists()]
    return {
        "ok": not missing,
        "coco_ready": not missing,
        "missing_coco_paths": missing,
        "train_annotation_bytes": train_ann.stat().st_size if train_ann.exists() else 0,
        "val_annotation_bytes": val_ann.stat().st_size if val_ann.exists() else 0,
    }


def _run_model_smoke(args: argparse.Namespace) -> dict[str, Any]:
    codetr_root = args.codetr_root.resolve()
    if str(codetr_root) not in sys.path:
        sys.path.insert(0, str(codetr_root))

    import importlib

    import torch
    from mmcv import Config
    from mmcv.runner import load_checkpoint

    importlib.import_module("projects")

    from mmdet.models import BACKBONES, build_detector
    from ops_dcnv3.functions import dcnv3_func
    from ops_dcnv3.modules.dcnv3 import DCNv3

    internimage_registered = "InternImage" in BACKBONES.module_dict
    dcnv3_extension_loaded = dcnv3_func.DCNv3 is not None
    if not internimage_registered:
        raise RuntimeError("InternImage is not registered in mmdet BACKBONES")
    if not dcnv3_extension_loaded:
        raise RuntimeError("DCNv3 extension did not load; dcnv3_func.DCNv3 is None")

    tiny_cuda = _run_tiny_dcnv3_cuda(torch=torch, DCNv3=DCNv3, require_cuda=args.require_cuda)

    cfg = Config.fromfile(str(args.config))
    if not getattr(cfg, "model", None):
        raise RuntimeError(f"config has no model: {args.config}")
    model = build_detector(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    model.init_weights()
    checkpoint = load_checkpoint(model, cfg.load_from, map_location="cpu", strict=False)

    return {
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "cuda_available": bool(torch.cuda.is_available()),
        "internimage_registered": internimage_registered,
        "dcnv3_extension_loaded": dcnv3_extension_loaded,
        "dcnv3_extension_file": str(getattr(dcnv3_func.DCNv3, "__file__", "")),
        "tiny_dcnv3_cuda": tiny_cuda,
        "model_class": model.__class__.__name__,
        "backbone_class": model.backbone.__class__.__name__,
        "backbone_core_op": getattr(model.backbone, "core_op", None),
        "params": int(sum(parameter.numel() for parameter in model.parameters())),
        "load_from": str(cfg.load_from),
        "checkpoint_top_keys": list(checkpoint.keys())[:8] if isinstance(checkpoint, dict) else [],
    }


def _run_tiny_dcnv3_cuda(*, torch: Any, DCNv3: Any, require_cuda: bool) -> dict[str, Any]:
    if not torch.cuda.is_available():
        if require_cuda:
            raise RuntimeError("CUDA is unavailable for required DCNv3 CUDA smoke")
        return {"ok": False, "skipped": True, "reason": "cuda unavailable"}

    device = torch.device("cuda:0")
    module = DCNv3(channels=32, group=4).to(device).eval()
    x = torch.randn(1, 8, 8, 32, device=device)
    with torch.no_grad():
        y = module(x)
    torch.cuda.synchronize(device)
    if tuple(y.shape) != tuple(x.shape):
        raise RuntimeError(f"unexpected tiny DCNv3 output shape: {tuple(y.shape)} vs {tuple(x.shape)}")
    return {
        "ok": True,
        "device_name": torch.cuda.get_device_name(0),
        "input_shape": list(x.shape),
        "output_shape": list(y.shape),
    }


if __name__ == "__main__":
    raise SystemExit(main())
