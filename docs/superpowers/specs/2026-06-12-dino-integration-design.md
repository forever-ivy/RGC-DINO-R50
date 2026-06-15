# DINO Official Integration Design

## Context

The project plan defines RGC-DINO-R50 as the only mainline solution. The detector core must reuse IDEA-Research DINO 4-scale ResNet-50 rather than reimplementing a DETR detector from scratch. Current repository state already has a lightweight v0 engineering loop for labels, splits, local metric evaluation, no-detection baseline inference, submission packaging, and bsub script generation. It does not yet contain the official DINO code, CUDA op checks, DINO config mapping, or a model smoke-test path.

This phase is the first implementation sub-project for the full plan. It prepares the official DINO engineering base and verifies that the environment can import and inspect it. It must not start full training or benchmarks from Codex.

## Goals

- Add a stable location and loader strategy for official IDEA-Research DINO code.
- Preserve current `src/rgc_dino` utilities as the project-specific engineering layer.
- Add scripts that check whether DINO is present, importable, and structurally ready for later RGC-DINO work.
- Add configuration stubs for DINO R50 4-scale and future RGC overrides.
- Add bsub script generation for DINO smoke checks and future training commands without submitting jobs.
- Keep all heavy training and long GPU work out of Codex sessions unless explicitly approved later.

## Non-Goals

- Do not run full DINO training.
- Do not compile CUDA ops automatically in Codex unless explicitly approved.
- Do not download pretrained weights into the repository.
- Do not implement RGB-only training, IR/Depth side encoders, RGC fusion, score calibration, or final inference in this phase.
- Do not commit datasets, checkpoints, generated ZIP files, or logs.

## Recommended Approach

Use an external vendor directory under the project tree:

```text
external/IDEA-Research-DINO/
```

The directory will be ignored by git except for a small README/manifest that documents the expected source, commit, and setup commands. This is safer than copying third-party source into `src/`, and it keeps the project-specific layer clean. The project will use scripts to detect the external directory and prepend it to `PYTHONPATH` only for DINO checks or future training jobs.

If a future workflow wants a git submodule instead, the same `external/IDEA-Research-DINO/` path can be reused. For now, the repo should remain usable even when the external code is absent: checks should fail with actionable messages instead of import tracebacks.

## Architecture

### Files And Responsibilities

- `external/README.md` documents external code policy and the expected DINO location.
- `configs/dino_r50_4scale.yaml` records project-level DINO integration settings: external path, official config path, class count, dn labelbook size, pretrained-weight policy, and output roots.
- `src/rgc_dino/dino_integration.py` provides small, testable helpers:
  - resolve the external DINO root;
  - verify required files/directories exist;
  - build a temporary import path;
  - produce a structured status report.
- `scripts/check_dino_integration.py` exposes the status report as a CLI and returns non-zero when required DINO files are missing.
- `scripts/write_bsub_dino_smoke.py` writes, but does not submit, an LSF script for future DINO smoke checks.
- `tests/test_dino_integration.py` covers missing-directory handling, status reporting, and bsub script content without requiring the real DINO repo.

### Data Flow

1. User places or clones IDEA-Research DINO into `external/IDEA-Research-DINO/`.
2. `scripts/check_dino_integration.py` reads `configs/dino_r50_4scale.yaml`.
3. The script checks for expected files such as:
   - `main.py`
   - `models/dino/`
   - `models/dino/ops/`
   - an official DINO 4-scale config path if present in the cloned version.
4. The script prints a structured summary and exits:
   - `0` when all required paths exist;
   - `2` when the external DINO tree is missing or incomplete.
5. `scripts/write_bsub_dino_smoke.py` writes an LSF script that activates `/data1/liuxuan/activate-py310.sh`, changes into the project root, checks DINO integration, and prints the intended next manual commands.

## Error Handling

- Missing external DINO directory must produce a concise message with the expected path and clone command.
- Missing CUDA ops directory must be reported separately from missing repo root.
- Missing official config must not crash the script; it should report which path was expected.
- The bsub writer must create parent directories for the script output but must not call `bsub`.
- Checks must avoid importing CUDA extensions directly in this phase unless the user explicitly runs a later compile/test command.

## Testing Strategy

Use TDD for implementation:

1. Write tests for a missing DINO root returning a failed status with actionable paths.
2. Write tests for a fake minimal DINO tree returning success for required path checks.
3. Write tests for the CLI exit code behavior using temporary directories.
4. Write tests for the bsub smoke script content: environment activation, project root, check script invocation, and no `bsub` submission command.
5. Run:

```sh
source /data1/liuxuan/activate-py310.sh
PYTHONPATH=src python -m unittest discover -s tests
python -m py_compile src/rgc_dino/*.py scripts/*.py
```

No GPU training, benchmarks, or long jobs are part of phase 1 verification.

## Acceptance Criteria

- `configs/dino_r50_4scale.yaml` exists and documents the official DINO integration inputs.
- `scripts/check_dino_integration.py` gives a clean, actionable failure when DINO is absent.
- The same check passes against a fake minimal DINO tree in tests.
- `scripts/write_bsub_dino_smoke.py` writes a smoke-check LSF script without submitting it.
- Unit tests cover the new DINO integration helpers and pass together with existing tests.
- README or docs mention the phase 1 DINO integration workflow and the rule that real training remains a manual bsub action.

## Open Decisions

- The exact official DINO commit is not pinned yet. The first implementation should include a manifest field and documentation placeholder for the commit, but it should not guess a commit hash.
- Whether to clone DINO via `gh repo clone`, plain `git clone`, or manual upload is an operational choice. The code should support the target path regardless of how it is populated.
