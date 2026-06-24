---
name: training-debugger
description: This skill should be used when the user asks to debug training, mentions "loss不下降", "OOM", "显存爆炸", "shape mismatch", "resume失败", "checkpoint加载失败", "train/val异常", "训练报错", or wants help preparing, checking, or troubleshooting RGC-DINO / DINO training runs on this repository.
version: 1.0.0
---

# Training Debugger

Use this skill for training preparation, training failure triage, and safe debugging of the RGC-DINO training stack.

## Core operating rules

- Treat `CLAUDE.md` as the operational authority for this repo.
- Keep all heavy work under `/data1/liuxuan/`.
- Do not start long GPU jobs in an interactive session without explicit user approval.
- Prefer smoke checks, config inspection, checkpoint inspection, and LSF/bsub script generation before proposing a full run.
- Prefer generating or reviewing `bsub` scripts over launching full training directly.

## Read these files first

- `CLAUDE.md`
- `configs/default.yaml`
- `scripts/train_rgc_dino.py`
- `scripts/write_bsub_train.py`
- `scripts/check_dino_integration.py`
- `scripts/write_bsub_dino_smoke.py`
- `src/rgc_dino/dino_training.py`
- `src/rgc_dino/training_splits.py`

## Key repo facts to remember

- Main trainer: `scripts/train_rgc_dino.py`
- Preferred cluster launch path: `scripts/write_bsub_train.py`
- External DINO root: `external/IDEA-Research-DINO`
- Official DINO pretrained weights expected at `/data1/liuxuan/checkpoints/dino/checkpoint0011_4scale.pth`
- Training data: `source/训练集`
- Split manifest area: `outputs/splits/`
- Project output root from config: `/data1/liuxuan/logs/rgc-dino-r50`

## Debugging workflow

### 1. Classify the failure

Classify the problem before suggesting fixes:

- environment/import problem
- external DINO integration problem
- missing pretrained checkpoint / wrong init path
- resume / checkpoint scope mismatch
- dataset / split / label issue
- batch / memory issue
- loss or metric behavior issue
- output / logging / checkpoint writing issue

### 2. Check the safest upstream causes first

Check these before proposing model changes:

- environment activation assumptions from `CLAUDE.md`
- DINO repo exists and has expected structure
- pretrained checkpoint exists
- split files and assignments exist
- quality cache path assumptions match the job
- `--resume`, `--init-dino-checkpoint`, and `--pretrain-dino-weights` are not being combined illegally

### 3. Prefer low-cost validation steps

Prefer these in order:

1. inspect config and CLI flags
2. inspect logs and checkpoint metadata
3. run import/syntax/smoke-safe checks
4. use `--smoke-only` or small `--limit-train` / `--val-batches` style checks when appropriate
5. only then propose a real training job

### 4. For loss or convergence issues

Check repo-specific causes before offering generic advice:

- wrong checkpoint initialization path or scope
- class-dependent weight mismatch handling
- fold/split quality
- image max side / multiscale choices
- horizontal flip settings
- quality cache / gate setup
- mismatch between documented plan and actual script arguments

Avoid jumping straight to vague advice like “lower LR” unless the repo evidence supports it.

## Safe outputs to provide

Good outcomes for this skill:

- a concrete diagnosis tied to repo files
- a short preflight checklist
- a minimal smoke-check command
- a corrected training command
- a corrected `bsub` generation command
- a ranked list of likely root causes with evidence

## Avoid

- Do not recommend direct heavy retraining by default.
- Do not ignore the repo’s cluster-first workflow.
- Do not assume MMDetection conventions; this repo is a custom DINO-based pipeline.
- Do not suggest changes that violate the offline/local competition workflow.

## Success criterion

The result should help the user move from “training is broken or risky” to “the next safe diagnostic or launch step is clear and grounded in this repo.”
