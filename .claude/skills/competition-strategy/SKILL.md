---
name: competition-strategy
description: This skill should be used when the user asks about submission strategy, leaderboard discipline, whether a ZIP should be submitted, competition constraints, ensemble/fusion choices, TTA strategy, candidate promotion, or safe next steps for improving leaderboard performance in this repository.
version: 1.0.0
---

# Competition Strategy

Use this skill for repo-specific competition discipline: legal strategy boundaries, submission safety, candidate promotion, and avoiding repeated leaderboard mistakes.

## Non-negotiable constraints

Read `CLAUDE.md` first and preserve these rules:

- no external training data
- no online API calls in the project pipeline
- no simple voting/averaging ensemble
- submit only complete test-set prediction ZIPs
- do not expose or print sensitive auth/cookie files

## Read these files first

- `CLAUDE.md`
- `scripts/promote_submission_candidate.py`
- `scripts/monitor_competition.py`
- `scripts/check_leaderboard.py`
- `scripts/submit_prediction.py`
- `src/rgc_dino/submission_promotion.py`
- `src/rgc_dino/submission_manifest.py`
- `docs/README.md`
- `docs/archive/lessons_learned/TTA_FAILURE_ANALYSIS.md`
- `docs/archive/lessons_learned/2FOLD_FAILURE_LESSONS.md`

## Core strategic lessons already established in this repo

- weak-model fusion can reduce score
- validation and leaderboard are related but not identical
- naïve TTA averaging can fail badly
- simple averaging/voting ensemble is not an acceptable default path here
- a strong single model is usually a higher-priority milestone than fancy fusion

## Submission decision workflow

### 1. Check whether the artifact is even eligible

Reject or warn on artifacts that look like:

- validation output
- OOF output
- debug output
- empty-submission checks
- partial test predictions

### 2. Require a reason

A candidate should only move forward with a concrete reason such as:

- better local validation
- deliberate threshold/NMS change backed by evidence
- strong new checkpoint
- explicit user instruction

### 3. Check provenance

Prefer candidates with traceable provenance:

- checkpoint path
- config path
- local metric context
- git commit / manifest / checksum where available

### 4. Respect timing discipline

- do not spam submissions
- wait for leaderboard refresh before judging results
- compare against the current baseline, not memory alone

## Preferred repo mechanisms

- Use promotion logic rather than manually tossing ZIPs into submission flow.
- Use monitor-state and leaderboard-state files when explaining current status.
- Use submission manifests and promotion metadata when available.

## Heuristics for advice

- If evidence is weak, recommend more local validation before submission.
- If a direction has already failed locally and historically, state that clearly.
- If the candidate is valid but not clearly better, frame it as optional rather than mandatory.
- If the user asks for aggressive score chasing, keep advice within the repo’s competition constraints.

## Avoid

- Do not recommend invalid ZIP types.
- Do not recommend simple averaging ensembles.
- Do not ignore the repo’s documented TTA/fusion failure lessons.
- Do not use generic Kaggle advice that conflicts with this competition’s rules.

## Success criterion

The result should help the user choose a submission action that is legal, disciplined, and evidence-backed, while avoiding known bad patterns in this repo.
