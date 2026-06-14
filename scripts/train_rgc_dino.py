#!/usr/bin/env python
"""Train or smoke-check the project RGC-DINO wrapper on aligned tri-modal data."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
DINO_ROOT = ROOT / "external" / "IDEA-Research-DINO"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(DINO_ROOT))
os.environ.setdefault("TORCH_HOME", "/data1/liuxuan/cache/torch")

from engine import train_one_epoch  # noqa: E402
from main import build_model_main, get_args_parser  # noqa: E402
from util.get_param_dicts import get_param_dict  # noqa: E402
from util.slconfig import SLConfig  # noqa: E402

from rgc_dino.dino_batch import collate_rgc_dino_batch  # noqa: E402
from rgc_dino.dino_dataset import MultimodalDinoDataset  # noqa: E402
from rgc_dino.dino_training import load_checkpoint_into_model, load_training_state, move_targets_to_device  # noqa: E402
from rgc_dino.models.rgc_dino_adapter import RgcDinoModel  # noqa: E402
from rgc_dino.quality_features import load_quality_feature_cache  # noqa: E402
from rgc_dino.training_splits import select_train_val_ids  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", type=Path, default=ROOT / "configs" / "dino_a0_rgb_4scale.py")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "source" / "训练集")
    parser.add_argument("--labels", type=Path, default=ROOT / "source" / "训练集" / "labels")
    parser.add_argument("--assignments", type=Path, default=ROOT / "outputs" / "splits" / "fold_assignments.jsonl")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--train-all", action="store_true", help="train on every labeled aligned sample; no val split")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr-drop", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-max-side", type=int, default=640)
    parser.add_argument(
        "--train-image-max-sides",
        type=int,
        nargs="+",
        help="optional per-sample train-time multi-scale longest-side choices; validation uses --image-max-side",
    )
    parser.add_argument("--random-horizontal-flip-prob", type=float, default=0.5)
    parser.add_argument("--quality-cache", type=Path)
    parser.add_argument("--side-base-channels", type=int, default=32)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-val", type=int)
    parser.add_argument("--val-batches", type=int, default=0)
    parser.add_argument("--log-gates-batches", type=int, default=0)
    parser.add_argument("--init-dino-checkpoint", type=Path)
    parser.add_argument("--resume", type=Path, help="resume a prior RGC-DINO training checkpoint")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--debug", action="store_true", help="use official engine debug break behavior")
    parser.add_argument("--smoke-only", action="store_true", help="run one forward/loss batch and exit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not DINO_ROOT.exists():
        print(f"external DINO tree not found: {DINO_ROOT}", file=sys.stderr)
        return 2
    if args.resume is not None and args.init_dino_checkpoint is not None:
        print("--resume and --init-dino-checkpoint cannot be used together", file=sys.stderr)
        return 2
    if args.train_all and args.val_batches > 0:
        print("--train-all requires --val-batches 0 because no validation split is built", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    official_args = _build_official_args(args)
    _set_seed(args.seed)

    device = torch.device(args.device)
    model, criterion, _postprocessors = build_model_main(official_args)
    wrapped_model = RgcDinoModel(
        model,
        side_base_channels=args.side_base_channels,
    )
    if args.init_dino_checkpoint is not None:
        report = load_checkpoint_into_model(
            wrapped_model.dino_model,
            args.init_dino_checkpoint,
            skip_mismatched_shapes=True,
        )
        print(
            json.dumps(
                {
                    "loaded_init_dino_checkpoint": str(report.checkpoint_path),
                    "missing_keys": len(report.missing_keys),
                    "skipped_shape_mismatch_keys": len(report.skipped_keys),
                    "unexpected_keys": len(report.unexpected_keys),
                },
                sort_keys=True,
            )
        )
    wrapped_model.to(device)
    criterion.to(device)

    train_loader, val_loader = _build_loaders(args)
    if args.log_gates_batches > 0:
        print(
            json.dumps(
                {
                    "fusion_gates_initial": _summarize_fusion_gates(
                        wrapped_model,
                        train_loader,
                        device=device,
                        amp=args.amp,
                        max_batches=args.log_gates_batches,
                    )
                },
                sort_keys=True,
            )
        )
    if args.smoke_only:
        smoke = _run_smoke_batch(
            wrapped_model,
            criterion,
            train_loader,
            device=device,
            use_dn=official_args.use_dn,
            amp=args.amp,
        )
        print(json.dumps(smoke, indent=2, sort_keys=True))
        return 0

    param_dicts = get_param_dict(official_args, wrapped_model)
    _validate_param_group_coverage(wrapped_model, param_dicts)
    print(
        json.dumps(
            {
                "optimizer_param_groups": _optimizer_param_group_diagnostics(
                    wrapped_model,
                    param_dicts,
                    default_lr=float(official_args.lr),
                ),
            },
            sort_keys=True,
        )
    )
    optimizer = torch.optim.AdamW(param_dicts, lr=official_args.lr, weight_decay=official_args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, official_args.lr_drop)

    history = _load_history(args.output_dir)
    if args.resume is not None:
        report = load_training_state(
            wrapped_model,
            args.resume,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )
        official_args.start_epoch = report.start_epoch
        history = [item for item in history if int(item.get("epoch", -1)) < report.start_epoch]
        print(
            json.dumps(
                {
                    "resumed_checkpoint": str(report.checkpoint_path),
                    "start_epoch": report.start_epoch,
                    "missing_keys": len(report.missing_keys),
                    "unexpected_keys": len(report.unexpected_keys),
                    "optimizer_loaded": report.optimizer_loaded,
                    "lr_scheduler_loaded": report.lr_scheduler_loaded,
                },
                sort_keys=True,
            )
        )
    for epoch in range(int(official_args.start_epoch), int(args.epochs)):
        train_stats = train_one_epoch(
            wrapped_model,
            criterion,
            train_loader,
            optimizer,
            device,
            epoch,
            max_norm=official_args.clip_max_norm,
            wo_class_error=False,
            lr_scheduler=lr_scheduler,
            args=official_args,
            logger=None,
            ema_m=None,
        )
        if not official_args.onecyclelr:
            lr_scheduler.step()

        val_stats: dict[str, float] = {}
        if args.val_batches > 0:
            val_stats = _evaluate_loss_only(
                wrapped_model,
                criterion,
                val_loader,
                device=device,
                use_dn=official_args.use_dn,
                amp=args.amp,
                max_batches=args.val_batches,
            )

        row = {"epoch": epoch, "train": train_stats, "val": val_stats}
        if args.log_gates_batches > 0:
            row["fusion_gates"] = _summarize_fusion_gates(
                wrapped_model,
                train_loader,
                device=device,
                amp=args.amp,
                max_batches=args.log_gates_batches,
            )
        history.append(row)
        _save_checkpoint(args.output_dir, wrapped_model, optimizer, lr_scheduler, epoch, args, official_args)
        (args.output_dir / "rgc_train_log.jsonl").write_text(
            "\n".join(json.dumps(item, sort_keys=True) for item in history) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(row, sort_keys=True))

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
            "--seed",
            str(args.seed),
            "--num_workers",
            str(args.num_workers),
        ]
        + (["--amp"] if args.amp else [])
        + (["--debug"] if args.debug else [])
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
    official_args.start_epoch = 0
    official_args.batch_size = args.batch_size
    official_args.epochs = args.epochs
    official_args.lr_drop = _resolve_lr_drop(
        config_lr_drop=int(official_args.lr_drop),
        epochs=int(args.epochs),
        explicit_lr_drop=args.lr_drop,
    )
    official_args.debug = args.debug
    official_args.use_ema = False
    official_args.save_results = False
    official_args.save_log = False
    return official_args


def _resolve_lr_drop(*, config_lr_drop: int, epochs: int, explicit_lr_drop: int | None) -> int:
    if explicit_lr_drop is not None:
        if explicit_lr_drop <= 0:
            raise ValueError("lr_drop must be positive")
        return int(explicit_lr_drop)
    if config_lr_drop <= 0:
        raise ValueError("config lr_drop must be positive")
    if epochs > 1 and config_lr_drop <= 1:
        return max(1, min(epochs, round(0.9 * epochs)))
    return int(config_lr_drop)


def _validate_param_group_coverage(model: torch.nn.Module, param_dicts: list[dict[str, Any]]) -> None:
    trainable_names_by_id = {
        id(param): name
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    grouped_ids: list[int] = []
    for group in param_dicts:
        grouped_ids.extend(id(param) for param in group.get("params", []) if param.requires_grad)

    grouped_counts = Counter(grouped_ids)
    missing = sorted(name for param_id, name in trainable_names_by_id.items() if param_id not in grouped_counts)
    duplicated = sorted(trainable_names_by_id[param_id] for param_id, count in grouped_counts.items() if count > 1)
    unexpected = sorted(param_id for param_id in grouped_counts if param_id not in trainable_names_by_id)
    if missing or duplicated or unexpected:
        raise ValueError(
            "optimizer parameter groups do not exactly cover trainable parameters: "
            f"missing={missing[:10]}, duplicated={duplicated[:10]}, unexpected_count={len(unexpected)}"
        )


def _optimizer_param_group_diagnostics(
    model: torch.nn.Module,
    param_dicts: list[dict[str, Any]],
    *,
    default_lr: float,
) -> list[dict[str, Any]]:
    names_by_id = {id(param): name for name, param in model.named_parameters()}
    diagnostics: list[dict[str, Any]] = []
    for group_index, group in enumerate(param_dicts):
        params = [param for param in group.get("params", []) if param.requires_grad]
        names = [names_by_id.get(id(param), "<unnamed>") for param in params]
        fusion_names = [name for name in names if name.startswith("feature_fusion.")]
        diagnostics.append(
            {
                "group": group_index,
                "lr": float(group.get("lr", default_lr)),
                "param_count": int(sum(param.numel() for param in params)),
                "tensor_count": len(params),
                "feature_fusion_tensor_count": len(fusion_names),
                "feature_fusion_sample": fusion_names[:5],
            }
        )
    return diagnostics


def _build_loaders(args: argparse.Namespace):
    train_ids, val_ids = select_train_val_ids(
        dataset_root=args.dataset_root,
        labels_dir=args.labels,
        assignments_path=args.assignments,
        fold=args.fold,
        train_all=args.train_all,
    )
    if args.limit_train is not None:
        train_ids = train_ids[: args.limit_train]
    if args.limit_val is not None:
        val_ids = val_ids[: args.limit_val]
    quality_cache_path = getattr(args, "quality_cache", None)
    quality_cache = load_quality_feature_cache(quality_cache_path) if quality_cache_path is not None else None

    train_dataset = MultimodalDinoDataset.from_paths(
        dataset_root=args.dataset_root,
        labels_dir=args.labels,
        sample_ids=train_ids,
        image_max_side=args.image_max_side,
        image_max_sides=getattr(args, "train_image_max_sides", None),
        random_horizontal_flip_prob=args.random_horizontal_flip_prob,
        quality_cache=quality_cache,
    )
    val_dataset = None
    if val_ids:
        val_dataset = MultimodalDinoDataset.from_paths(
            dataset_root=args.dataset_root,
            labels_dir=args.labels,
            sample_ids=val_ids,
            image_max_side=args.image_max_side,
            quality_cache=quality_cache,
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_rgc_dino_batch,
        num_workers=args.num_workers,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_rgc_dino_batch,
            num_workers=args.num_workers,
        )
    return train_loader, val_loader


def _run_smoke_batch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: DataLoader,
    *,
    device: torch.device,
    use_dn: bool,
    amp: bool,
) -> dict[str, Any]:
    model.train()
    criterion.train()
    samples, targets = next(iter(data_loader))
    samples = samples.to(device)
    targets = move_targets_to_device(targets, device)
    with torch.cuda.amp.autocast(enabled=amp):
        outputs = model(samples, targets if use_dn else None)
        loss_dict = criterion(outputs, targets)
        losses = _weighted_loss(loss_dict, criterion.weight_dict)
    return {
        "loss": float(losses.detach().cpu()),
        "pred_logits_shape": list(outputs["pred_logits"].shape),
        "pred_boxes_shape": list(outputs["pred_boxes"].shape),
        "target_objects": int(sum(len(target["labels"]) for target in targets)),
    }


@torch.no_grad()
def _evaluate_loss_only(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: DataLoader,
    *,
    device: torch.device,
    use_dn: bool,
    amp: bool,
    max_batches: int,
) -> dict[str, float]:
    model.eval()
    criterion.eval()
    total_loss = 0.0
    count = 0
    for count, (samples, targets) in enumerate(data_loader, start=1):
        samples = samples.to(device)
        targets = move_targets_to_device(targets, device)
        with torch.cuda.amp.autocast(enabled=amp):
            outputs = model(samples, targets if use_dn else None)
            loss_dict = criterion(outputs, targets)
            losses = _weighted_loss(loss_dict, criterion.weight_dict)
        total_loss += float(losses.detach().cpu())
        if count >= max_batches:
            break
    return {"loss": total_loss / max(count, 1), "batches": float(count)}


@torch.no_grad()
def _summarize_fusion_gates(
    model: RgcDinoModel,
    data_loader: DataLoader,
    *,
    device: torch.device,
    amp: bool,
    max_batches: int,
) -> dict[str, list[dict[str, float]]]:
    was_training = model.training
    model.eval()
    collected: dict[str, list[dict[str, float]]] = {}
    counts: dict[str, list[int]] = {}
    try:
        for batch_index, (samples, _targets) in enumerate(data_loader, start=1):
            samples = samples.to(device)
            with torch.cuda.amp.autocast(enabled=amp):
                gates = model.compute_fusion_gates(samples)
            _accumulate_gate_stats(gates, collected, counts)
            if batch_index >= max_batches:
                break
    finally:
        if was_training:
            model.train()
    return _finalize_gate_stats(collected, counts)


def _accumulate_gate_stats(
    gates: Mapping[str, list[torch.Tensor]],
    collected: dict[str, list[dict[str, float]]],
    counts: dict[str, list[int]],
) -> None:
    for modality, level_gates in gates.items():
        if modality not in collected:
            collected[modality] = [
                {"sum": 0.0, "min": float("inf"), "max": float("-inf")}
                for _ in level_gates
            ]
            counts[modality] = [0 for _ in level_gates]
        for level, gate in enumerate(level_gates):
            detached = gate.detach().float().cpu()
            collected[modality][level]["sum"] += float(detached.sum())
            collected[modality][level]["min"] = min(collected[modality][level]["min"], float(detached.min()))
            collected[modality][level]["max"] = max(collected[modality][level]["max"], float(detached.max()))
            counts[modality][level] += int(detached.numel())


def _finalize_gate_stats(
    collected: dict[str, list[dict[str, float]]],
    counts: dict[str, list[int]],
) -> dict[str, list[dict[str, float]]]:
    summary: dict[str, list[dict[str, float]]] = {}
    for modality, levels in collected.items():
        summary[modality] = []
        for level, stats in enumerate(levels):
            count = max(counts[modality][level], 1)
            summary[modality].append(
                {
                    "mean": stats["sum"] / count,
                    "min": stats["min"] if count > 0 else 0.0,
                    "max": stats["max"] if count > 0 else 0.0,
                }
            )
    return summary


def _weighted_loss(loss_dict: Mapping[str, torch.Tensor], weight_dict: Mapping[str, float]) -> torch.Tensor:
    return sum(loss_dict[key] * weight_dict[key] for key in loss_dict.keys() if key in weight_dict)


def _save_checkpoint(
    output_dir: Path,
    model: RgcDinoModel,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    cli_args: argparse.Namespace,
    official_args: argparse.Namespace,
) -> None:
    payload = {
        "model": model.state_dict(),
        "dino_model": model.dino_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "epoch": epoch,
        "cli_args": vars(cli_args),
        "official_args": vars(official_args),
    }
    torch.save(payload, output_dir / "checkpoint.pth")
    torch.save(payload, output_dir / f"checkpoint{epoch:04d}.pth")


def _load_history(output_dir: Path) -> list[dict[str, Any]]:
    log_path = output_dir / "rgc_train_log.jsonl"
    if not log_path.exists():
        return []
    return [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    raise SystemExit(main())
