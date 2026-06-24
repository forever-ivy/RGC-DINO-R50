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
- `docs/archive/lessons_learned/TTA_FAILURE_ANALYSIS.md`
- `docs/archive/lessons_learned/2FOLD_FAILURE_LESSONS.md`
- `scripts/train_rgc_dino.py`
- `scripts/infer_rgc_dino.py`
- `src/rgc_dino/models/rgc_dino_adapter.py`
- `src/rgc_dino/models/rgc_fusion.py`
- `src/rgc_dino/models/side_encoder.py`
- `src/rgc_dino/dino_dataset.py`

## Repo-specific architecture facts to preserve

- this repo is DINO-centered, not a generic MMDetection project
- RGB is the primary modality path through the official DINO backbone
- IR and depth are auxiliary branches injected through reliability-gated residual fusion
- depth handling and quality-feature gating are already repo-specific design choices
- the dataset is small and hardware is constrained, so methods that require massive multimodal pretraining may be poor fits

## Review workflow

### 1. Extract the real innovation

State whether the paper mainly changes:

- backbone
- multimodal fusion location
- query interaction
- loss/training recipe
- data augmentation / pseudo-labeling
- post-processing

### 2. Map it to this repo

Answer:

- which current module is closest
- what would be replaced vs extended
- whether this is a light adaptation or a major rewrite

### 3. Check constraints

Judge the method against:

- no external training data
- offline/local pipeline
- hardware limits
- current codebase maturity
- reproducibility cost

### 4. Compare against local evidence

Use the repo’s prior failures and roadmap as evidence, not just general ML intuition.

### 5. End with a recommendation

Choose one:

- good fit, worth adapting now
- interesting but lower priority than current roadmap
- only worth partial borrowing
- poor fit for this repo right now

## Good output format

When useful, organize the answer as:

- method summary
- what it changes relative to current repo
- expected benefit
- reproduction cost
- integration points
- risks / rule conflicts
- final recommendation

## Avoid

- Do not give generic paper praise without mapping to repo files.
- Do not recommend architectures that contradict competition rules without calling that out.
- Do not ignore the dataset size and hardware realities.
- Do not confuse “interesting research” with “good immediate engineering choice.”

## Success criterion

The result should let the user quickly decide whether a paper or GitHub method deserves engineering time in this specific project.
