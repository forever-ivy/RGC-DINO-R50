---
name: experiment-tracker
description: This skill should be used when the user asks to compare experiments, rank checkpoints, sweep thresholds, inspect validation results, track ablations, summarize training outcomes, decide which checkpoint is best, or organize experiment evidence for this repository.
version: 1.0.0
---

# Experiment Tracker

Use this skill for experiment bookkeeping, checkpoint comparison, inference-parameter sweeps, and evidence-based model selection in this repo.

## Core goal

Turn scattered logs, checkpoints, inference outputs, and local metrics into a compact decision about what changed, what improved, and what should be tried next.

## Read these files first

- `CLAUDE.md`
- `docs/README.md`
- `docs/FINAL_ROADMAP.md`
- `scripts/cache_codetr_predictions.py`
- `scripts/sweep_codetr_class_thresholds.py`
- `scripts/sweep_codetr_top100_allocation.py`
- `scripts/sweep_codetr_submission_params.py`
- `scripts/codetr_results_to_submission.py`
- `scripts/promote_submission_candidate.py`

For legacy RGC-DINO/DINO experiments, additionally read:

- `scripts/select_best_checkpoint.py`
- `scripts/sweep_inference_params.py`
- `scripts/infer_rgc_dino.py`
- `scripts/evaluate_predictions.py`

## Important repo evidence

- Current mainline is Co-DETR + InternImage-L.
- Current anchor is 50.353 / strict final-TXT mAP `0.4379615851682616` / hard-val `0.29545499238138817`.
- Checkpoint ranking and strict final-TXT evaluation are first-class workflows.
- Class-wise thresholds and legal top100 allocation are critical; native COCO eval alone is insufficient.
- Historical failure lessons: low-threshold blind tuning, naive TTA averaging, and weak fold fusion reduced leaderboard score.
- Local validation evidence should be summarized before any submission suggestion.

## Standard workflow

### 1. Identify the experiment unit

Figure out what is being compared:

- epochs within one run
- checkpoints within Co-DETR continuation
- threshold/NMS/top100 allocation settings
- image-side/resize settings
- TTA/tile variants
- fusion/RDT/future IR-depth variants
- backbone/config variants

### 2. Collect the canonical evidence

Prefer these sources:

- run output dirs under `outputs/codetr/`
- strict final-TXT eval reports
- hard-val reports
- class threshold / allocation JSON files
- submission manifests and promotion JSON sidecars
- monitor/leaderboard history when comparing submitted candidates
- local evaluation metrics from `evaluate_predictions.py` when working on legacy TXT predictions

### 3. Normalize the comparison

Always state:

- dataset split or fold
- checkpoint path(s)
- inference settings
- postprocess settings: thresholds, NMS, image side, max detections/top100 allocation
- whether the metric is native COCO, strict final-TXT validation, hard-val, or leaderboard
- whether comparison is apples-to-apples

### 4. Recommend the next action

End with one of these:

- keep current best checkpoint/recipe
- sweep a narrower threshold/NMS/allocation range
- stop pursuing a failed direction
- promote a candidate for submission review
- gather missing evidence before deciding

## Strong repo-specific heuristics

- Treat documented failure lessons as priors, not noise.
- Do not assume TTA helps; check whether the implementation is actually supported in the path being used.
- Do not assume multi-fold or multi-model fusion helps; weak components can drag a stronger model down.
- Distinguish local validation, hard-val, test prediction, OOF, and leaderboard behavior.
- Prefer compact experiment summaries over vague “looks better” judgments.

## Suggested summary format

When useful, summarize experiments as:

- experiment name / run directory
- checkpoint(s)
- strict local metric / hard-val / leaderboard if available
- inference and postprocess settings
- notable differences
- decision: keep / reject / sweep / submit-review

## Avoid

- Do not recommend submission purely from intuition.
- Do not blur validation, test, OOF, and leaderboard evidence.
- Do not present incomparable runs as a ranked list without caveats.
