# Repository Guidelines

## Environment & HPC Rules

This project runs on the Yanshan University 3090 server. Do not run heavy training, benchmarks, simulations, or long GPU/CPU jobs directly unless the user explicitly confirms it is allowed for the current session.

Prefer generating commands or LSF `bsub` scripts for training jobs. Keep all project files, datasets, environments, caches, and outputs under `/data1/liuxuan`; do not use `/home` except for minimal configuration. Never use `sudo`, `apt install`, or `apt upgrade`.

Use the project Python environment with:

```sh
source /data1/liuxuan/activate-py310.sh
```

Expected environment:

- Python 3.10.20
- PyTorch 2.12.0+cu126
- CUDA build 12.6

## Current Project Direction

The repository name remains `RGC-DINO-R50`, but the confirmed competition mainline is now **Co-DETR + InternImage-L single-model detection** with strict final-TXT validation, class-wise thresholds, legal top-100 allocation, hard validation, and guarded promotion.

Current anchor:

```text
Co-DETR InternImage-L GPU1 load-from epoch6
+ top100 allocation person0865/light10625/uav0825/boat003
leaderboard 50.353
strict final-TXT mAP 0.4379615851682616
hard-val 0.29545499238138817
```

Legacy `RGC-DINO/DINO-R50` remains as fallback/comparison infrastructure and a substrate for future IR/Depth reliability-gated fusion migration, not as the active leaderboard route.

## Competition Automation & Submission Rules

Use the existing background automation instead of manual browser work whenever possible. Sensitive login state lives in ignored files:

- `outputs/cookies.json`: leaderboard cookie file.
- `outputs/aicomp_auth.json`: localStorage login state for submission automation; keep mode `600`, never print or commit it.

The expected monitor session is `competition_monitor`. It should run with auto-submit enabled and watch only `outputs/submissions/`:

```sh
tmux attach -t competition_monitor
tail -f outputs/monitor/monitor.log
cat outputs/monitor/monitor_state.json
```

If the monitor is stopped, restart it with:

```sh
source /data1/liuxuan/activate-py310.sh
tmux new-session -d -s competition_monitor \
  "bash -lc 'cd /data1/liuxuan/projects/RGC-DINO-R50 && source /data1/liuxuan/activate-py310.sh && PYTHONUNBUFFERED=1 python scripts/monitor_competition.py --cookies outputs/cookies.json --local-storage outputs/aicomp_auth.json --auto-submit --ignore-existing --predictions-dir outputs/submissions --output-dir outputs/monitor --check-interval 3600 > outputs/monitor/monitor.log 2>&1'"
```

Auto-submit behavior:

- The monitor checks the leaderboard every hour.
- It auto-submits only direct `*.zip` files under `outputs/submissions/`.
- `--ignore-existing` sets a baseline at monitor startup, so existing ZIPs are not submitted.
- A ZIP is considered new when its modification time is newer than `outputs/monitor/monitor_state.json` field `last_submission`.

AI submission policy:

- Only submit complete test-set prediction ZIPs, never validation/OoF/debug/empty/partial ZIPs.
- Put candidate ZIPs in `outputs/submissions/` via promotion tooling and use informative names with model, epoch, thresholds, or local metric.
- Submit only when there is a clear reason: improved strict local validation, deliberate postprocess change backed by evidence, or explicit user instruction.
- Do not blindly submit many near-duplicate ZIPs. After a real submission, wait for the platform leaderboard refresh, usually about 1 hour, before judging the result.
- Before trusting a changed login or page flow, run a dry run that reaches the ZIP upload input but does not upload:

```sh
python scripts/submit_prediction.py outputs/submissions/<candidate>.zip \
  --local-storage outputs/aicomp_auth.json \
  --dry-run \
  --wait 30
```

Manual rank checks:

```sh
python scripts/check_leaderboard.py --cookies outputs/cookies.json
python scripts/check_leaderboard.py --cookies outputs/cookies.json --quiet
```

Manual submission is allowed only for a deliberate single candidate:

```sh
python scripts/submit_prediction.py outputs/submissions/<candidate>.zip \
  --local-storage outputs/aicomp_auth.json \
  --wait 30
```

## Project Structure

Current repository contents:

- `configs/`: Co-DETR/InternImage, legacy DINO/RGC-DINO, and experiment configuration files.
- `doc/official/`: official competition statement/process PDFs.
- `docs/`: current roadmap, hardware feasibility, Co-DETR handoff, and strategy/lesson summaries.
- `external/`: third-party source trees, ignored by git except project-owned notes.
- `scripts/`: environment checks, data export, LSF generation, training/inference helpers, postprocess sweeps, promotion, and monitor automation.
- `source/训练集/`: extracted training set with aligned `visible/`, `infrared/`, `depth/`, and `labels/`.
- `source/AIC2026_PHASE_1_1000/`: unlabeled three-modality prediction/test image set.
- `src/rgc_dino/`: shared data/metrics/submission utilities plus legacy RGC-DINO/RDT/fusion code.
- `tests/`: lightweight CPU-safe tests.

Large datasets should live under `/data1/liuxuan/` and project outputs should live in `/data1/liuxuan/logs` or ignored project-local `outputs/`.

## Development Commands

Use lightweight checks during development:

```sh
python scripts/check_environment.py
python -m py_compile path/to/file.py
PYTHONPATH=src python -m unittest discover -s tests
```

Do not start full training from Codex/Claude without explicit user approval. For Co-DETR training, prepare an LSF script, for example:

```sh
python scripts/write_bsub_codetr_continue.py --output outputs/jobs/codetr_continue.lsf
# user/resource confirmation first, then manually:
# bsub < outputs/jobs/codetr_continue.lsf
```

## Current Co-DETR Entrypoints

- Integration checks: `scripts/check_codetr_environment.py`, `scripts/check_codetr_integration.py`, `scripts/prepare_codetr_training.py`
- Training job writers: `scripts/write_bsub_codetr_train.py`, `scripts/write_bsub_codetr_continue.py`
- Validation/postprocess: `scripts/cache_codetr_predictions.py`, `scripts/sweep_codetr_class_thresholds.py`, `scripts/sweep_codetr_top100_allocation.py`, `scripts/sweep_codetr_submission_params.py`
- Submission conversion/promotion: `scripts/codetr_results_to_submission.py`, `scripts/run_codetr_test_and_promote.sh`, `scripts/promote_submission_candidate.py`
- Competition monitor: `scripts/monitor_competition.py`

## Coding Style

Write Python code in clear modules with small functions. Prefer configuration files over hard-coded paths. Use absolute `/data1/liuxuan/...` paths for datasets, weights, and outputs when running on the server.

Keep secrets, tokens, passwords, and API keys out of code, logs, README files, and prompts.

## Git & GitHub

Activate the environment before using Git tools:

```sh
source /data1/liuxuan/activate-py310.sh
```

`git` and `gh` are installed in `/data1/liuxuan/envs/py310`. Do not commit datasets, checkpoints, caches, `.env` files, credentials, large archives, or external code clones. Use `.gitignore` for `outputs/`, `runs/`, `checkpoints/`, `*.pt`, and `*.pth`.

## Testing & Verification

Prefer small CPU-safe tests first. GPU checks should be minimal, such as importing `torch` and checking device visibility. Avoid long-running jobs on the login/session node.

For real training verification, use a short smoke test only when explicitly approved, or submit through the cluster-approved workflow.

## Acknowledgement

If this project contributes to a paper or result using this HPC resource, include:

```text
本论文的数值计算得到了燕山大学超算中心的计算支持和帮助
```
