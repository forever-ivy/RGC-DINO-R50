# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RGC-DINO-R50 is a competition repository for multimodal urban scene object detection. The repository name remains `RGC-DINO-R50`, but the confirmed leaderboard mainline is now **Co-DETR + InternImage-L single-model detection** with strict final-TXT validation, class-wise thresholds, legal top-100 allocation, hard validation, and promotion metadata.

Current anchor:
- Model/recipe: `Co-DETR InternImage-L GPU1 load-from epoch6 + top100 allocation person0865/light10625/uav0825/boat003`
- Leaderboard: `50.353`
- Strict final-TXT mAP: `0.4379615851682616`
- Hard-val: `0.29545499238138817`
- Anchor checkpoint: `outputs/codetr/internimage_l_fresh_ep7_loadfrom_gpu1_20260624/best_bbox_mAP_epoch_6.pth`

Legacy `RGC-DINO/DINO-R50` code remains in the repo as fallback, comparison infrastructure, and a substrate for later IR/Depth reliability-gated fusion migration. Do not treat old RGC-DINO PDFs or archived plans as current strategy; those documents have been removed or consolidated.

**Competition constraints:**
- 12 object classes (IDs 0-11)
- Metric: mAP@50:95
- No external training data
- No online API calls in the training/inference/submission pipeline
- No test-set pseudo-label training
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

**CRITICAL:** This project runs on the Yanshan University 3090 server. All work must stay under `/data1/liuxuan/`. Never use `sudo`, `apt install`, or `apt upgrade`. Do not run heavy training or long GPU jobs directly in interactive sessions without explicit user confirmation.

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

## Network Access Notes

This server has partial outbound network access. Some public sites may be reachable while others may fail from the server itself.

- For literature review, paper search, method comparison, and general web research, Claude may still use built-in web search/research tools.
- This does **not** relax the competition rule against online API calls in the project pipeline; training, inference, and submission artifacts must remain offline/local.
- When code, PDFs, or model weights must be downloaded onto the server, prefer reachable mirrors, GitHub releases, author pages, or have the user download locally and upload under `/data1/liuxuan/`.
- If a resource host is unreachable from the server, treat that as a deployment/download constraint rather than a reason to skip literature search.

## Current Strategy Docs

Use these as the source of truth:

- `README.md` - current repo entrypoint
- `docs/README.md` - documentation navigation and current status
- `docs/FINAL_ROADMAP.md` - Co-DETR + InternImage-L mainline and submission gates
- `docs/2x3090_FEASIBLE_ROADMAP.md` - hardware feasibility on 2×RTX 3090
- `docs/CODETR_INTERNIMAGE_L_TRAINING_HANDOFF.md` - Co-DETR external dependency and training handoff
- `docs/OPPONENT_URBAN_VISUAL_RECOGNITION_LESSONS.md` - opponent-public-method lessons; learn ideas only, never reuse weights/submissions
- `doc/official/` - official competition statement and process PDFs

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

Submit only complete test-set prediction ZIPs. Never submit validation, OoF, debug, empty, partial, or smoke-test ZIPs. Candidate filenames should include model/epoch/threshold or local metric context. Submit only with a clear reason: better local validation, a deliberate postprocess change, or explicit user instruction. After a real submission, wait for the platform leaderboard refresh, usually about 1 hour, before judging the result.

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
python scripts/check_environment.py
python -m py_compile src/rgc_dino/*.py scripts/*.py
PYTHONPATH=src python -m unittest discover -s tests
```

### Data inspection and validation
```bash
python scripts/inspect_dataset.py \
  --root source/训练集 \
  --labels source/训练集/labels \
  --manifest outputs/manifests/train_2000.jsonl

python scripts/inspect_labels.py --labels source/训练集/labels --max-errors 10
```

Sample training data has 4 slightly out-of-bounds boxes. Use `--clip-labels` with `scripts/make_splits.py` and `scripts/evaluate_predictions.py` when needed.

## Current Co-DETR Mainline Workflow

### External integration checks

```bash
source /data1/liuxuan/activate-py310.sh
python scripts/check_codetr_environment.py
python scripts/check_codetr_integration.py --codetr-root external/Co-DETR
python scripts/prepare_codetr_training.py --no-require-weights
```

Expected external resources:
- Co-DETR source: `external/Co-DETR/`
- InternImage source/reference: `external/InternImage-master/`
- Legacy DINO source: `external/IDEA-Research-DINO/`
- InternImage-L public pretrain: `/data1/liuxuan/checkpoints/internimage/internimage_l_public_pretrain.pth`
- Co-DETR/Co-DINO public pretrain: `/data1/liuxuan/checkpoints/codetr/codetr_internimage_l_public_pretrain.pth`

### Training job generation

**Do not start training directly.** Prepare commands or LSF `bsub` scripts first:

```bash
python scripts/write_bsub_codetr_train.py --output outputs/jobs/codetr_train.lsf
python scripts/write_bsub_codetr_continue.py --output outputs/jobs/codetr_continue.lsf
# Submit manually only after resource/user confirmation:
# bsub < outputs/jobs/codetr_continue.lsf
```

Continuation jobs must stop before test inference unless strict final-TXT validation beats the current anchor. The current Co-DETR gate is strict mAP `> 0.4379615851682616`, hard-val not degraded, and leaderboard baseline `>= 50.353` in promotion metadata.

### Inference, postprocess, and promotion

Primary scripts:
- `scripts/cache_codetr_predictions.py` - cache validation raw predictions
- `scripts/sweep_codetr_class_thresholds.py` - class threshold search
- `scripts/sweep_codetr_top100_allocation.py` - legal top-100 allocation search
- `scripts/sweep_codetr_submission_params.py` - strict final-TXT sweep
- `scripts/codetr_results_to_submission.py` - convert Co-DETR results to competition TXT/ZIP with manifest
- `scripts/run_codetr_test_and_promote.sh` - test inference + packaging + guarded promotion
- `scripts/promote_submission_candidate.py` - explicit candidate promotion into `outputs/submissions/`

Submission candidates must include checkpoint/config/git/split/postprocess provenance and promotion reason. The monitor also rejects stale or weak candidates based on local mAP and hard-val metadata.

## Legacy RGC-DINO / DINO-R50 Code Path

Legacy code is still useful for comparisons and future IR/Depth fusion migration:

- `scripts/train_rgc_dino.py`
- `scripts/infer_rgc_dino.py`
- `scripts/write_bsub_train.py`
- `configs/default.yaml`
- `configs/dino_r50_4scale.yaml`
- `src/rgc_dino/dino_dataset.py`
- `src/rgc_dino/models/rgc_dino_adapter.py`
- `src/rgc_dino/models/rgc_fusion.py`
- `src/rgc_dino/models/side_encoder.py`

The DINO R50 COCO pretrained checkpoint is still expected at `/data1/liuxuan/checkpoints/dino/checkpoint0011_4scale.pth` for legacy experiments. Do not let this legacy path supersede the current Co-DETR + InternImage-L mainline.

## Architecture Notes

**Shared/core modules** (`src/rgc_dino/`):
- `constants.py` - project-wide constants
- `dataset.py` - three-modality dataset loading and alignment checks
- `labels.py` - label parsing, validation, and clipping
- `splits.py` / `training_splits.py` - grouped stratified K-fold splitting
- `submission.py` - submission format validation and ZIP packaging
- `metrics.py` - CPU mAP@50:95 evaluator using COCO API
- `submission_manifest.py` / `submission_promotion.py` - candidate provenance and guarded promotion
- `quality_features.py` and `rdt.py` - RGB/IR/depth quality/RDT features for ablations and future gating
- `dino_integration.py` / `dino_training.py` - legacy DINO helpers
- `models/` - legacy reliability-gated residual fusion and side encoders

**Competition data flow:**
1. Load aligned `visible/`, `infrared/`, `depth/` images by stem.
2. Train/evaluate on labels in YOLO format: `[class_id, norm_center_x, norm_center_y, norm_w, norm_h]`.
3. Convert predictions to competition format: `[class_id, norm_x, norm_y, norm_w, norm_h, confidence]`.
4. Ensure each test image has exactly one TXT file, empty if no detections.
5. Evaluate with per-IoU AP averaged from 0.50 to 0.95.

## Configuration

Current Co-DETR configs:
- `configs/codetr_internimage_l_stage0.yaml`
- `configs/codetr_internimage_l_mm_config.py`
- `configs/codetr_internimage_l_aic2026_test.py`
- `configs/codetr_internimage_l_eval_s*.py`
- `configs/codetr_r50_stage0_mm_config.py`
- `configs/codetr_r50_tiny_mm_config.py`

Legacy configs:
- `configs/default.yaml`
- `configs/dino_r50_4scale.yaml`
- `configs/dino_a0_rgb_4scale.py`
- `configs/swin_l_stage1.yaml`

## File Paths and Data Organization

- Datasets: under `/data1/liuxuan/` (in-repo `source/` or `/data1/liuxuan/datasets/`)
- Training outputs/logs: `/data1/liuxuan/logs/rgc-dino-r50` or ignored `outputs/`
- Checkpoints/model caches: under `/data1/liuxuan/`, never under `/home/`
- Third-party source trees: `external/`, ignored by git except small project-owned docs

Do not commit:
- Datasets
- Checkpoints (`*.pt`, `*.pth`)
- Large outputs (`outputs/`, `runs/`, `checkpoints/`)
- Credentials or `.env` files
- External code clones

## Testing

Run CPU-safe tests before committing:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

For code changes touching scripts, also run syntax checks:

```bash
python -m py_compile src/rgc_dino/*.py scripts/*.py
```

## Development Boundaries

- Prefer generating LSF scripts over direct execution for GPU-intensive work.
- Keep interactive sessions lightweight and verifiable.
- Do not submit or promote candidates without strict local evidence and clean provenance.
- Do not resurrect old failed directions: low-threshold blind tuning, naive TTA averaging, weak fold fusion, test pseudo-label training, or simple averaging/voting ensembles.

## Acknowledgement

If this project contributes to a paper or result using this HPC resource, include:

```text
本论文的数值计算得到了燕山大学超算中心的计算支持和帮助
```
