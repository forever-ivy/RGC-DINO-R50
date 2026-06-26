---
name: paper-review-reproduce
description: This skill should be used when the user asks to analyze a paper, evaluate a GitHub method, judge whether a new architecture fits RGB-IR-Depth detection, compare a paper idea to the current repo, or plan how to reproduce/adapt a method under this project's constraints.
version: 1.0.0
---

# Paper Review Reproduce

Use this skill to evaluate whether a paper, project, or architecture idea is worth adapting into this repository.

## Main job

Translate “this paper/method looks interesting” into a repo-grounded answer:

- what the method really changes
- whether it fits RGB + IR + Depth detection here
- what part of this repo it would touch
- what it costs to reproduce
- whether it conflicts with current evidence or competition rules

## Read these files first

- `CLAUDE.md`
- `README.md`
- `docs/README.md`
- `docs/FINAL_ROADMAP.md`
- `docs/2x3090_FEASIBLE_ROADMAP.md`
- `docs/OPPONENT_URBAN_VISUAL_RECOGNITION_LESSONS.md`
- `configs/codetr_internimage_l_stage0.yaml`
- `configs/codetr_internimage_l_mm_config.py`
- `scripts/write_bsub_codetr_continue.py`
- `scripts/codetr_results_to_submission.py`
- `scripts/sweep_codetr_class_thresholds.py`
- `scripts/sweep_codetr_top100_allocation.py`

For legacy RGC-DINO fusion details, additionally read:

- `scripts/train_rgc_dino.py`
- `scripts/infer_rgc_dino.py`
- `src/rgc_dino/models/rgc_dino_adapter.py`
- `src/rgc_dino/models/rgc_fusion.py`
- `src/rgc_dino/models/side_encoder.py`
- `src/rgc_dino/dino_dataset.py`

## Repo-specific architecture facts to preserve

- Current leaderboard mainline is Co-DETR + InternImage-L, not legacy DINO-R50.
- Current best is 50.353 with strict final-TXT mAP `0.4379615851682616` and hard-val `0.29545499238138817`.
- Current validated high-score path is RGB-first Co-DETR with class-wise thresholds and legal top100 allocation.
- IR and depth are future auxiliary/reliability-gated fusion directions on top of the Co-DETR mainline, not a reason to revert to old RGC-DINO as the primary route.
- The dataset is small and hardware is constrained, so methods requiring massive multimodal pretraining or broad external data are poor fits.
- Prior local evidence already rejects low-threshold blind tuning, naive TTA averaging, weak fold/model fusion, and test-set pseudo-label training.

## Review workflow

### 1. Extract the real innovation

State whether the paper mainly changes:

- backbone
- detector/head/query interaction
- multimodal fusion location
- loss/training recipe
- data augmentation / pseudo-labeling
- post-processing / calibration / NMS / allocation

### 2. Map it to this repo

Answer:

- which current module or script is closest
- whether it extends Co-DETR/InternImage or only the legacy RGC-DINO fallback
- what would be replaced vs extended
- whether this is a light adaptation or a major rewrite

### 3. Check constraints

Judge the method against:

- no external training data
- no online API calls in the pipeline
- no test pseudo-label training
- no simple voting/averaging ensemble
- 2×RTX 3090 hardware limits
- current codebase maturity and reproduction cost

### 4. Compare against local evidence

Use the repo’s current anchor and prior failures as evidence, not just general ML intuition.

### 5. End with a recommendation

Choose one:

- good fit, worth adapting now
- interesting but lower priority than current roadmap
- only worth partial borrowing
- poor fit for this repo right now

## Good output format

When useful, organize the answer as:

- method summary
- what it changes relative to current Co-DETR mainline
- expected benefit
- reproduction cost
- integration points
- risks / rule conflicts
- final recommendation
