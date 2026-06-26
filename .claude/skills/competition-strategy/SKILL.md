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
- no test-set pseudo-label training
- no simple voting/averaging ensemble
- submit only complete test-set prediction ZIPs
- do not expose or print sensitive auth/cookie files

## Read these files first

- `CLAUDE.md`
- `README.md`
- `docs/README.md`
- `docs/FINAL_ROADMAP.md`
- `scripts/promote_submission_candidate.py`
- `scripts/monitor_competition.py`
- `scripts/check_leaderboard.py`
- `scripts/submit_prediction.py`
- `src/rgc_dino/submission_promotion.py`
- `src/rgc_dino/submission_manifest.py`

## Core strategic lessons already established in this repo

- Current mainline is Co-DETR + InternImage-L, not legacy RGC-DINO/DINO-R50.
- Current anchor is 50.353 / strict final-TXT mAP `0.4379615851682616` / hard-val `0.29545499238138817`.
- Weak-model/fold fusion can reduce score; do not assume more fusion helps.
- Naive TTA score/box averaging failed badly; only validated NMS/WBF-like single-model merging is acceptable.
- Low-threshold blind tuning increases false positives and can drop leaderboard score.
- Validation and leaderboard are related but not identical; require hard-val and prediction distribution evidence.
- A strong single model plus disciplined postprocess is the priority.

## Submission decision workflow

### 1. Check whether the artifact is eligible

Reject or warn on artifacts that look like:

- validation output
- OOF output
- debug output
- empty-submission checks
- partial test predictions
- stale candidates without promotion metadata

### 2. Require a reason

A candidate should only move forward with a concrete reason such as:

- strict final-TXT mAP beats the current anchor
- deliberate threshold/NMS/top100 allocation change backed by evidence
- strong new checkpoint with hard-val and distribution checks
- explicit user instruction

### 3. Check provenance

Prefer candidates with traceable provenance:

- checkpoint path and hash
- config path
- local strict final-TXT metric
- hard-val metric when applicable
- class thresholds / NMS / image side / top100 allocation settings
- git commit / manifest / checksum

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
- If a candidate is valid but not clearly better, frame it as optional rather than mandatory.
- If the user asks for aggressive score chasing, keep advice within the repo’s competition constraints.

## Avoid

- Do not recommend invalid ZIP types.
- Do not recommend simple averaging ensembles.
- Do not ignore the repo’s documented low-threshold/TTA/fusion failure lessons summarized in `docs/README.md` and `docs/FINAL_ROADMAP.md`.
- Do not use generic Kaggle advice that conflicts with this competition’s rules.

## Success criterion

The result should help the user choose a submission action that is legal, disciplined, and evidence-backed, while avoiding known bad patterns in this repo.
