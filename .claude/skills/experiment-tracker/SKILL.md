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

- `scripts/select_best_checkpoint.py`
- `scripts/sweep_inference_params.py`
- `scripts/infer_rgc_dino.py`
- `scripts/evaluate_predictions.py`
- `scripts/validate_3fold_fusion.py`
- `scripts/validate_tta_on_fold0.py`
- `docs/README.md`
- `docs/archive/lessons_learned/TTA_FAILURE_ANALYSIS.md`
- `docs/archive/lessons_learned/2FOLD_FAILURE_LESSONS.md`

## Important repo evidence

- Checkpoint ranking is a first-class workflow here.
- Inference threshold / NMS sweeps are already supported.
- This repo has documented failure cases where “more fusion” or naïve TTA reduced score.
- Local validation evidence should be summarized before any submission suggestion.

## Standard workflow

### 1. Identify the experiment unit

Figure out what is being compared:

- epochs within one run
- folds
- threshold/NMS settings
- TTA variants
- fusion variants
- backbone/config variants

### 2. Collect the canonical evidence

Prefer these sources:

- training logs under `logs/` or run output dirs
- checkpoint files and ranking outputs
- inference sweep outputs such as ranking JSON
- local evaluation metrics from `evaluate_predictions.py`
- docs that record prior known failures

### 3. Normalize the comparison

Always state:

- dataset split or fold
- checkpoint path(s)
- inference settings
- whether the metric is local validation or leaderboard
- whether comparison is apples-to-apples

### 4. Recommend the next action

End with one of these:

- keep current best checkpoint
- sweep a narrower threshold/NMS range
- stop pursuing a failed direction
- promote a candidate for submission review
- gather missing evidence before deciding

## Strong repo-specific heuristics

- Treat documented failure lessons as priors, not noise.
- Do not assume TTA helps; check whether the implementation is actually supported in the path being used.
- Do not assume multi-fold or multi-model fusion helps; weak components can drag a stronger model down.
- Distinguish local validation from leaderboard behavior.
- Prefer compact experiment summaries over vague “looks better” judgments.

## Suggested summary format

When useful, summarize experiments as:

- experiment name / run directory
- checkpoint(s)
- local metric
- inference settings
- notable differences
- decision: keep / reject / sweep / submit-review

## Avoid

- Do not recommend submission purely from intuition.
- Do not blur validation, test, OOF, and leaderboard evidence.
- Do not present incomparable runs as a ranked list without caveats.

## Success criterion

The result should leave the user with a clean, evidence-backed answer to “what is best now?” and “what is the most justified next experiment?”
