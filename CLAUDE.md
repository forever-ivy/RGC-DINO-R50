# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RGC-DINO-R50 is a competition project for multimodal urban scene object detection. It implements a single-model, three-modality detection pipeline using DINO-R50 as the detection core with reliability-gated residual cross-modal fusion to process RGB, Infrared, and Depth inputs.

**Competition constraints:**
- 12 object classes (IDs 0-11)
- Metric: mAP@50:95
- No external training data
- No online API calls
- No simple voting/averaging ensemble
- Output format: TXT files with `[class_id, norm_center_x, norm_center_y, norm_w, norm_h, confidence]`

**Dataset structure:**
- Training: `source/训练集/` - 2000 aligned three-modality samples with labels
- Test/submission: `source/AIC2026_PHASE_1_1000/` - 1000 unlabeled three-modality images

Each modality directory contains spatially aligned images by filename stem:
- `visible/` - RGB (3-channel uint8)
- `infrared/` - pseudo-RGB thermal grayscale (3-channel uint8)
- `depth/` - depth maps (1-channel uint16, millimeter, valid range 300-20000mm)

## Environment Setup

**CRITICAL:** This project runs on Yanshan University 3090 server. All work must stay under `/data1/liuxuan/`. Never use `sudo`, `apt install`, or `apt upgrade`. Do not run heavy training or long GPU jobs directly in interactive sessions without explicit user confirmation.

Activate the project environment:

```bash
source /data1/liuxuan/activate-py310.sh
python --version  # Should be 3.10.20
python scripts/check_environment.py
```

Expected environment:
- Python 3.10.20
- PyTorch 2.12.0+cu126
- CUDA build 12.6

## Competition Automation & Auto-submit

Use the project automation for rank checks and submissions. Do not print or commit sensitive files:

- `outputs/cookies.json` for leaderboard checks.
- `outputs/aicomp_auth.json` for localStorage submission auth; keep it private.

The expected background session is `competition_monitor`:

```bash
tmux attach -t competition_monitor
tail -f outputs/monitor/monitor.log
cat outputs/monitor/monitor_state.json
```

Restart it if needed:

```bash
source /data1/liuxuan/activate-py310.sh
tmux new-session -d -s competition_monitor \
  "bash -lc 'cd /data1/liuxuan/projects/RGC-DINO-R50 && source /data1/liuxuan/activate-py310.sh && PYTHONUNBUFFERED=1 python scripts/monitor_competition.py --cookies outputs/cookies.json --local-storage outputs/aicomp_auth.json --auto-submit --ignore-existing --predictions-dir outputs/submissions --output-dir outputs/monitor --check-interval 3600 > outputs/monitor/monitor.log 2>&1'"
```

Auto-submit watches direct `*.zip` files in `outputs/submissions/`. Existing ZIPs are ignored at startup through `--ignore-existing`; only ZIPs newer than `outputs/monitor/monitor_state.json` field `last_submission` are candidates.

Submit only complete test-set prediction ZIPs. Never submit validation, OoF, debug, empty, or test ZIPs. Candidate filenames should include model/epoch/threshold or local metric context. Submit only with a clear reason: better local validation, a deliberate ensemble/threshold change, or explicit user instruction. After a real submission, wait for the platform leaderboard refresh, usually about 1 hour, before judging the result.

Dry-run the browser flow before relying on changed auth or page behavior:

```bash
python scripts/submit_prediction.py outputs/submissions/<candidate>.zip \
  --local-storage outputs/aicomp_auth.json \
  --dry-run \
  --wait 30
```

Manual rank checks:

```bash
python scripts/check_leaderboard.py --cookies outputs/cookies.json
python scripts/check_leaderboard.py --cookies outputs/cookies.json --quiet
```

## Common Development Commands

### Lightweight verification (safe to run anytime)
```bash
# Syntax checks
python -m py_compile src/rgc_dino/*.py scripts/*.py

# Unit tests (CPU-safe)
PYTHONPATH=src python -m unittest discover -s tests

# Environment check
python scripts/check_environment.py
```

### Data inspection and validation
```bash
# Check three-modality alignment and generate manifest
python scripts/inspect_dataset.py \
  --root source/训练集 \
  --labels source/训练集/labels \
  --manifest outputs/manifests/train_2000.jsonl

# Validate label format and check for out-of-bounds boxes
python scripts/inspect_labels.py --labels source/训练集/labels --max-errors 10
```

### Complete v0 baseline workflow
The v0 baseline establishes the engineering loop without real DINO training:

```bash
# 1. Inspect dataset and generate manifest
python scripts/inspect_dataset.py \
  --root source/训练集 \
  --labels source/训练集/labels \
  --manifest outputs/manifests/train_2000.jsonl

# 2. Check labels (4 slightly out-of-bounds boxes in sample data - warns only)
python scripts/inspect_labels.py --labels source/训练集/labels --max-errors 10

# 3. Generate frozen split manifest (use --clip-labels to handle OOB boxes)
python scripts/make_splits.py \
  --labels source/训练集/labels \
  --folds 3 \
  --output-dir outputs/splits \
  --clip-labels

# 4. Run v0 no-detection baseline inference
python scripts/infer_baseline.py \
  --dataset-root source/AIC2026_PHASE_1_1000 \
  --output-dir outputs/baseline_predictions

# 5. Validate and package submission
python scripts/make_submission.py \
  --dataset-root source/AIC2026_PHASE_1_1000 \
  --submission-dir outputs/baseline_predictions \
  --zip-path outputs/baseline_predictions.zip

# 6. Evaluate predictions against labels (use --clip-labels with sample data)
python scripts/evaluate_predictions.py \
  --labels source/训练集/labels \
  --predictions outputs/baseline_predictions \
  --clip-labels

# 7. Generate LSF training script (does not submit automatically)
python scripts/write_bsub_train.py --output outputs/jobs/train_rgc_dino_r50.lsf
```

### DINO integration workflow
```bash
# Check DINO directory structure
python scripts/check_dino_integration.py --dino-root external/IDEA-Research-DINO

# Generate DINO smoke-check LSF script (does not submit)
python scripts/write_bsub_dino_smoke.py --output outputs/jobs/dino_smoke.lsf
```

## Architecture

**Core modules** (`src/rgc_dino/`):
- `constants.py` - Project-wide constants (num classes, box format, validation thresholds)
- `dataset.py` - Three-modality dataset loading with spatial alignment verification
- `dino_dataset.py` - DINO-format tri-modal dataset; multi-scale + flip augmentation, 2-channel depth (`[normalized, valid_mask]`)
- `labels.py` - Label parsing, validation, and clipping for YOLO-format annotations
- `splits.py` / `training_splits.py` - Grouped stratified K-fold splitting to prevent same-sequence leakage
- `submission.py` - Submission format validation and ZIP packaging
- `metrics.py` - CPU-based mAP@50:95 evaluator using COCO API
- `quality_features.py` - Per-image RGB/IR/depth quality stats feeding the reliability gate
- `dino_training.py` - Checkpoint load/restore helpers (`load_checkpoint_into_model`, `load_training_state`)
- `dino_integration.py` - DINO official repository integration utilities
- `models/` - `rgc_dino_adapter.py` (wraps official DINO, injects fusion), `rgc_fusion.py` (reliability-gated residual fusion, zero-init residual outputs), `side_encoder.py` (lightweight IR/depth encoders)

**Data flow:**
1. Three-modality images loaded from aligned `visible/`, `infrared/`, `depth/` directories
2. Labels in YOLO format: `[class_id, norm_center_x, norm_center_y, norm_w, norm_h]`
3. Model produces predictions with confidence: `[class_id, norm_x, norm_y, norm_w, norm_h, confidence]`
4. Each test image requires exactly one corresponding TXT file (empty file if no detections)
5. Evaluation uses per-IoU-threshold AP averaged from 0.5 to 0.95

**Split strategy:**
Uses grouped stratified K-fold where samples with the same filename stem prefix are kept together to prevent train/val leakage. The split manifest is frozen in `outputs/splits/split_manifest.json` to ensure reproducibility.

**External integration:**
DINO official code expected at `external/IDEA-Research-DINO/`. Do not compile CUDA ops, download weights, or start training in interactive sessions - generate LSF scripts instead.

## Training Workflow

**Do not start training directly.** Prepare commands or LSF `bsub` scripts first:

```bash
# For interactive testing (with explicit user approval only)
CUDA_VISIBLE_DEVICES=0 python scripts/train_baseline.py --config configs/default.yaml

# For cluster submission
python scripts/write_bsub_train.py --output outputs/jobs/train.lsf
# Then submit manually: bsub < outputs/jobs/train.lsf
```

Training outputs go to `/data1/liuxuan/logs/rgc-dino-r50` as configured in `configs/default.yaml`.

### DINO COCO pretrained weights (required for real detection)

The detector transformer/heads must be initialized from official DINO-4scale R50
COCO pretrained weights. Without them the transformer trains from scratch on 2000
images and fails to converge (cardinality explosion, mAP stuck ~0.17).

- File: `checkpoint0011_4scale.pth` (~200MB, 12ep/49.0 AP) from IDEA-Research/DINO.
- Place at: `/data1/liuxuan/checkpoints/dino/checkpoint0011_4scale.pth`.
- `scripts/train_rgc_dino.py --pretrain-dino-weights <path>` loads it into
  `dino_model`, auto-skipping class-dependent params whose shapes mismatch
  (91-class COCO vs 12-class project: `class_embed`, `label_enc`,
  `enc_out_class_embed`). Mutually exclusive with `--init-dino-checkpoint` and
  `--resume`.
- `scripts/write_bsub_train.py` defaults `--pretrain-dino-weights` to that path
  and **hard-fails if the file is missing** (no silent fallback to a self-trained
  checkpoint, which would otherwise mask the missing-pretrain bug).

Train-time augmentation in the RGC dataloader uses multi-scale longest-side
jitter (`--train-image-max-sides`, default 480/560/640/720) and horizontal flip
(`--random-horizontal-flip-prob`, default 0.5). The `data_aug_scales` key in the
DINO config only affects the official COCO transform path, which RGC does not use.

## Configuration

Main config: `configs/default.yaml`

Key settings:
- `paths.dataset_root` - Training data location
- `paths.output_dir` - Where checkpoints and logs are written
- `model.detector` - DINO variant (dino_r50_4scale)
- `model.fusion` - Fusion strategy (reliability_gated_residual_cross_modal_fusion)
- `training.folds` - K-fold count (3)
- `training.split_strategy` - grouped_stratified_kfold
- `data.num_classes` - 12

## File Paths and Data Organization

**Repository structure:**
- `configs/` - YAML configuration files
- `scripts/` - Standalone execution scripts for data prep, training, evaluation
- `src/rgc_dino/` - Reusable Python modules
- `tests/` - Unit tests (CPU-safe)
- `source/` - Training and test datasets (tracked but large files gitignored)
- `outputs/` - Generated manifests, splits, predictions, jobs (gitignored)
- `external/` - External code integrations like DINO (gitignored)

**Large file locations:**
- Datasets: under `/data1/liuxuan/` (either in-repo `source/` or separate `/data1/liuxuan/datasets/`)
- Training outputs: `/data1/liuxuan/logs/rgc-dino-r50`
- Model caches: stay under `/data1/liuxuan/`, never in `/home/`

## Label Format Notes

**Training labels** (5 fields): `class_id norm_cx norm_cy norm_w norm_h`
**Prediction labels** (6 fields): `class_id norm_cx norm_cy norm_w norm_h confidence`

Sample training data has 4 slightly out-of-bounds boxes. Use `--clip-labels` flag with:
- `scripts/make_splits.py`
- `scripts/evaluate_predictions.py`

This clips boxes to `[0.0, 1.0]` range while preserving their validity.

## Git Workflow

Git and `gh` CLI are available in the project environment after running `source /data1/liuxuan/activate-py310.sh`.

Do not commit:
- Datasets (already in `.gitignore`)
- Checkpoints (`*.pt`, `*.pth`)
- Large outputs (`outputs/`, `runs/`, `checkpoints/`)
- Credentials or `.env` files
- External code clones (`external/`)

## Testing

Run tests before committing:
```bash
PYTHONPATH=src python -m unittest discover -s tests
```

Tests are CPU-safe and designed for quick validation without GPU or heavy computation.

## Development Boundaries

This repository has completed v0 engineering loop with lightweight infrastructure. Real training, model architecture, data augmentation, and competition strategies should follow the detailed plan in `doc/RGC-DINO-R50 三模态单模型冲榜工程方案.pdf`.

**Key principle:** Prefer generating LSF scripts over direct execution for any GPU-intensive work. Keep interactive sessions lightweight and verifiable.

## Acknowledgement

If this project contributes to a paper or result using this HPC resource, include:

```text
本论文的数值计算得到了燕山大学超算中心的计算支持和帮助
```
