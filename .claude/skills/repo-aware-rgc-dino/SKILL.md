---
name: repo-aware-rgc-dino
description: This skill should be used when the user asks where something lives in this repository, how the RGC-DINO / Co-DETR pipeline is organized, which script owns training/inference/submission behavior, how RGB/IR/depth flow through the code, or how to navigate this project quickly without re-deriving its structure.
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
- `docs/FINAL_ROADMAP.md`
- `configs/codetr_internimage_l_stage0.yaml`

## Current direction

- The repository name remains `RGC-DINO-R50`, but the confirmed mainline is Co-DETR + InternImage-L.
- Current anchor is leaderboard 50.353 / strict final-TXT mAP `0.4379615851682616` / hard-val `0.29545499238138817`.
- Legacy RGC-DINO/DINO-R50 code remains for fallback, comparison, and future IR/Depth fusion migration.
- Old direction docs and archived plans have been removed; use current docs, not memory of old PDFs.

## Core repo map

### Current Co-DETR training and launch

- `scripts/check_codetr_environment.py` — Co-DETR environment check
- `scripts/check_codetr_integration.py` — external Co-DETR tree and weight path readiness
- `scripts/prepare_codetr_training.py` — lightweight COCO export + preflight + LSF generation
- `scripts/write_bsub_codetr_smoke.py` — smoke LSF generation
- `scripts/write_bsub_codetr_train.py` — Co-DETR train LSF generation
- `scripts/write_bsub_codetr_continue.py` — guarded continuation LSF generation
- `scripts/continue_codetr_internimage_l_stage1.sh` — continuation job body with validation/promotion gates

### Current Co-DETR inference, postprocess, and promotion

- `scripts/cache_codetr_predictions.py` — validation raw prediction cache
- `scripts/sweep_codetr_class_thresholds.py` — class-wise threshold search
- `scripts/sweep_codetr_top100_allocation.py` — legal top100 allocation search
- `scripts/sweep_codetr_submission_params.py` — strict final-TXT postprocess sweep
- `scripts/codetr_results_to_submission.py` — convert Co-DETR results to competition TXT/ZIP with manifest
- `scripts/run_codetr_test_and_promote.sh` — test inference and guarded promotion
- `scripts/promote_submission_candidate.py` — explicit candidate promotion into `outputs/submissions/`

### Submission and leaderboard workflow

- `scripts/make_submission.py`
- `scripts/submit_prediction.py`
- `scripts/monitor_competition.py`
- `scripts/check_leaderboard.py`
- `src/rgc_dino/submission_manifest.py`
- `src/rgc_dino/submission_promotion.py`

### Legacy RGC-DINO / DINO-R50 path

- `scripts/train_rgc_dino.py` — legacy RGC-DINO training entrypoint
- `scripts/infer_rgc_dino.py` — legacy RGC-DINO inference path
- `scripts/write_bsub_train.py` — legacy training LSF writer
- `scripts/check_dino_integration.py` — external IDEA DINO readiness
- `scripts/write_bsub_dino_smoke.py` — legacy DINO smoke-check LSF generation
- `src/rgc_dino/dino_dataset.py` — DINO-format tri-modal dataset
- `src/rgc_dino/quality_features.py` — quality feature computation
- `src/rgc_dino/models/rgc_dino_adapter.py` — DINO wrapper and fusion insertion point
- `src/rgc_dino/models/rgc_fusion.py` — reliability-gated residual fusion
- `src/rgc_dino/models/side_encoder.py` — IR/depth side encoders

### Strategy and official docs

- `docs/README.md`
- `docs/FINAL_ROADMAP.md`
- `docs/2x3090_FEASIBLE_ROADMAP.md`
- `docs/CODETR_INTERNIMAGE_L_TRAINING_HANDOFF.md`
- `docs/OPPONENT_URBAN_VISUAL_RECOGNITION_LESSONS.md`
- `doc/official/`

## Navigation rules

- Prefer current root docs, `CLAUDE.md`, and `docs/FINAL_ROADMAP.md` for operational truth.
- Treat legacy RGC-DINO scripts as real code but not the default leaderboard route.
- Distinguish training scripts, inference scripts, postprocess sweep scripts, and submission/promotion scripts clearly.
- Distinguish native Co-DETR/MMDetection evaluation from strict final-TXT competition evaluation.
- When asked about failed directions, use the summarized lessons in `docs/README.md` and `docs/FINAL_ROADMAP.md`.

## Common intent-to-file mapping

- “How do I continue current training?” → `scripts/write_bsub_codetr_continue.py`, `docs/CODETR_INTERNIMAGE_L_TRAINING_HANDOFF.md`
- “Where is Co-DETR postprocess?” → `scripts/sweep_codetr_class_thresholds.py`, `scripts/sweep_codetr_top100_allocation.py`, `scripts/codetr_results_to_submission.py`
- “How do we submit safely?” → `scripts/promote_submission_candidate.py`, `scripts/monitor_competition.py`, `src/rgc_dino/submission_promotion.py`
- “Where is fusion implemented?” → legacy/future path in `src/rgc_dino/models/rgc_fusion.py`, `src/rgc_dino/models/rgc_dino_adapter.py`
- “How are modalities loaded?” → `src/rgc_dino/dataset.py`, `src/rgc_dino/dino_dataset.py`
- “What already failed?” → summarized in `docs/README.md` and `docs/FINAL_ROADMAP.md`

## Avoid

- Do not default to generic framework guidance without first checking repo-specific files.
- Do not assume this repo is notebook-first.
- Do not treat legacy DINO/RGC docs or removed old PDFs as current strategy.
- Do not ignore the competition automation and submission-discipline scripts.

## Success criterion

The result should quickly point the user to the right files and explain the current Co-DETR mainline plus legacy fallback shape without unnecessary re-exploration.
