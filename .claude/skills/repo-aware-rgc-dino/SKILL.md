---
name: repo-aware-rgc-dino
description: This skill should be used when the user asks where something lives in this repository, how the RGC-DINO pipeline is organized, which script owns training/inference/submission behavior, how RGB/IR/depth flow through the code, or how to navigate this project quickly without re-deriving its structure.
version: 1.0.0
---

# Repo Aware RGC-DINO

Use this skill for fast, accurate navigation of the RGC-DINO-R50 repository.

## Main purpose

Avoid rediscovering the same repo structure every time. Map the user’s intent to the right docs, scripts, modules, and outputs quickly.

## Start with these anchors

- `CLAUDE.md`
- `README.md`
- `docs/README.md`
- `configs/default.yaml`

## Core repo map

### Training and launch

- `scripts/train_rgc_dino.py` — main training entrypoint
- `scripts/write_bsub_train.py` — preferred cluster job generation
- `scripts/check_dino_integration.py` — external DINO readiness
- `scripts/write_bsub_dino_smoke.py` — DINO smoke-check job generation

### Inference and evaluation

- `scripts/infer_rgc_dino.py` — main inference path
- `scripts/evaluate_predictions.py` — local metric evaluation
- `scripts/select_best_checkpoint.py` — checkpoint ranking
- `scripts/sweep_inference_params.py` — inference threshold/NMS sweeps

### Submission and leaderboard workflow

- `scripts/make_submission.py`
- `scripts/promote_submission_candidate.py`
- `scripts/submit_prediction.py`
- `scripts/monitor_competition.py`
- `scripts/check_leaderboard.py`
- `src/rgc_dino/submission_manifest.py`
- `src/rgc_dino/submission_promotion.py`

### Model and data internals

- `src/rgc_dino/dino_dataset.py` — DINO-format tri-modal dataset
- `src/rgc_dino/quality_features.py` — quality feature computation
- `src/rgc_dino/models/rgc_dino_adapter.py` — DINO wrapper and fusion insertion point
- `src/rgc_dino/models/rgc_fusion.py` — reliability-gated residual fusion
- `src/rgc_dino/models/side_encoder.py` — IR/depth side encoders

### Strategy and historical evidence

- `docs/FINAL_ROADMAP.md`
- `docs/2x3090_FEASIBLE_ROADMAP.md`
- `docs/archive/lessons_learned/TTA_FAILURE_ANALYSIS.md`
- `docs/archive/lessons_learned/2FOLD_FAILURE_LESSONS.md`

## Navigation rules

- Prefer current root docs and `CLAUDE.md` for operational truth.
- Treat archived docs as historical evidence, not the first operational source.
- Distinguish training scripts, inference scripts, and submission scripts clearly.
- Distinguish repo-specific DINO adaptation from generic DINO or MMDetection assumptions.

## Common intent-to-file mapping

- “How do I launch training?” → `scripts/write_bsub_train.py`, `scripts/train_rgc_dino.py`
- “Where is fusion implemented?” → `src/rgc_dino/models/rgc_fusion.py`, `src/rgc_dino/models/rgc_dino_adapter.py`
- “How are modalities loaded?” → `src/rgc_dino/dino_dataset.py`
- “How do we score checkpoints?” → `scripts/select_best_checkpoint.py`, `scripts/evaluate_predictions.py`
- “How do we submit safely?” → `scripts/promote_submission_candidate.py`, `scripts/monitor_competition.py`
- “What already failed?” → lessons-learned docs in `docs/archive/lessons_learned/`

## Avoid

- Do not default to generic framework guidance without first checking repo-specific files.
- Do not assume this repo is notebook-first or MMDetection-native.
- Do not ignore the competition automation and submission-discipline scripts.

## Success criterion

The result should quickly point the user to the right files and explain the pipeline shape without unnecessary re-exploration.
