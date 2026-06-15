# DINO Official Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build phase 1 of the project plan: a safe, testable integration layer for official IDEA-Research DINO code without running heavy training.

**Architecture:** Keep third-party DINO code outside the project package under `external/IDEA-Research-DINO/`, and keep project-owned glue in `src/rgc_dino`. The integration helper reports whether the external tree is present and structurally valid; CLI scripts expose the checks and generate HPC-safe bsub smoke scripts.

**Tech Stack:** Python 3.10, standard library only for phase 1, `unittest`, YAML config as plain documented project config, LSF `bsub` script generation only.

---

## File Structure

- Create `external/README.md`: external-code policy and exact DINO target path.
- Modify `.gitignore`: ignore `external/IDEA-Research-DINO/` while keeping `external/README.md` trackable.
- Create `configs/dino_r50_4scale.yaml`: project-level DINO integration settings.
- Create `src/rgc_dino/dino_integration.py`: status dataclass and helper functions for path checks.
- Create `scripts/check_dino_integration.py`: CLI check with clear exit codes.
- Create `scripts/write_bsub_dino_smoke.py`: writes an LSF smoke-check script, does not submit.
- Create `tests/test_dino_integration.py`: TDD coverage for missing tree, fake minimal tree, CLI behavior, and bsub content.
- Modify `src/rgc_dino/__init__.py`: export only stable helper names if needed.
- Modify `README.md`: add phase 1 DINO integration commands.

---

### Task 1: External DINO Policy And Ignore Rules

**Files:**
- Create: `external/README.md`
- Modify: `.gitignore`

- [ ] **Step 1: Write external-code policy file**

Create `external/README.md`:

```markdown
# External Code

This directory is for third-party source trees used by the project but not owned by this repository.

Expected DINO location:

```text
external/IDEA-Research-DINO/
```

Recommended source:

```text
https://github.com/IDEA-Research/DINO
```

Do not commit the cloned DINO source tree, pretrained weights, compiled CUDA artifacts, checkpoints, or generated logs. Keep only small project-owned manifests and documentation in git.

The phase 1 integration checks expect the external DINO tree to contain:

- `main.py`
- `models/dino/`
- `models/dino/ops/`
```

- [ ] **Step 2: Update `.gitignore`**

Add this near the data/cache ignore section:

```gitignore
# External third-party source trees
external/IDEA-Research-DINO/
```

- [ ] **Step 3: Verify docs are visible and DINO tree is ignored**

Run:

```sh
test -f external/README.md
git check-ignore external/IDEA-Research-DINO/main.py
```

Expected:

```text
external/IDEA-Research-DINO/main.py
```

- [ ] **Step 4: Commit if git metadata is writable**

```sh
source /data1/liuxuan/activate-py310.sh
git add .gitignore external/README.md
git commit -m "docs: document external dino source policy"
```

Expected if git metadata is writable: commit succeeds. If git fails because `.git` points to a read-only path, record the failure and continue without reverting files.

---

### Task 2: DINO Config Stub

**Files:**
- Create: `configs/dino_r50_4scale.yaml`
- Test: `tests/test_dino_integration.py`

- [ ] **Step 1: Write failing test for config existence and required keys**

Add to new `tests/test_dino_integration.py`:

```python
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DinoIntegrationConfigTest(unittest.TestCase):
    def test_dino_config_exists_with_required_keys(self) -> None:
        config = ROOT / "configs" / "dino_r50_4scale.yaml"

        text = config.read_text(encoding="utf-8")

        self.assertIn("external_dino_root:", text)
        self.assertIn("official_repo:", text)
        self.assertIn("official_config:", text)
        self.assertIn("num_classes: 12", text)
        self.assertIn("dn_labelbook_size:", text)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```sh
source /data1/liuxuan/activate-py310.sh
PYTHONPATH=src python -m unittest tests.test_dino_integration.DinoIntegrationConfigTest.test_dino_config_exists_with_required_keys
```

Expected: `FileNotFoundError` for `configs/dino_r50_4scale.yaml`.

- [ ] **Step 3: Create config**

Create `configs/dino_r50_4scale.yaml`:

```yaml
project:
  name: rgc-dino-r50
  phase: dino_official_integration

dino:
  official_repo: https://github.com/IDEA-Research/DINO
  external_dino_root: external/IDEA-Research-DINO
  official_config: config/DINO/DINO_4scale.py
  required_paths:
    - main.py
    - models/dino
    - models/dino/ops
  detector: dino_r50_4scale
  num_classes: 12
  dn_labelbook_size: 13
  num_queries: 900
  num_feature_levels: 4
  pretrained_weights:
    allow_public_pretrained_weights: true
    store_under: /data1/liuxuan/checkpoints

paths:
  output_dir: /data1/liuxuan/logs/rgc-dino-r50
  local_outputs: outputs/dino_integration
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```sh
source /data1/liuxuan/activate-py310.sh
PYTHONPATH=src python -m unittest tests.test_dino_integration.DinoIntegrationConfigTest.test_dino_config_exists_with_required_keys
```

Expected: `OK`.

- [ ] **Step 5: Commit if git metadata is writable**

```sh
source /data1/liuxuan/activate-py310.sh
git add configs/dino_r50_4scale.yaml tests/test_dino_integration.py
git commit -m "test: add dino integration config contract"
```

---

### Task 3: DINO Integration Status Helper

**Files:**
- Create: `src/rgc_dino/dino_integration.py`
- Modify: `tests/test_dino_integration.py`

- [ ] **Step 1: Add failing tests for missing and complete DINO trees**

Append to `tests/test_dino_integration.py`:

```python
import tempfile

from rgc_dino.dino_integration import check_dino_tree


class DinoIntegrationStatusTest(unittest.TestCase):
    def test_missing_dino_root_reports_actionable_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "external" / "IDEA-Research-DINO"

            status = check_dino_tree(root)

            self.assertFalse(status.ok)
            self.assertEqual(status.root, root)
            self.assertIn("DINO root not found", status.messages[0])
            self.assertIn("git clone https://github.com/IDEA-Research/DINO", status.clone_hint)

    def test_minimal_dino_tree_passes_required_path_checks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "external" / "IDEA-Research-DINO"
            (root / "models" / "dino" / "ops").mkdir(parents=True)
            (root / "config" / "DINO").mkdir(parents=True)
            (root / "main.py").write_text("", encoding="utf-8")
            (root / "config" / "DINO" / "DINO_4scale.py").write_text("", encoding="utf-8")

            status = check_dino_tree(root)

            self.assertTrue(status.ok)
            self.assertEqual(status.missing_paths, ())
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```sh
source /data1/liuxuan/activate-py310.sh
PYTHONPATH=src python -m unittest tests.test_dino_integration.DinoIntegrationStatusTest
```

Expected: import failure for `rgc_dino.dino_integration`.

- [ ] **Step 3: Implement status helper**

Create `src/rgc_dino/dino_integration.py`:

```python
"""Helpers for checking the external IDEA-Research DINO tree."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

DEFAULT_DINO_ROOT = Path("external") / "IDEA-Research-DINO"
DEFAULT_OFFICIAL_CONFIG = Path("config") / "DINO" / "DINO_4scale.py"
DEFAULT_REQUIRED_PATHS: tuple[Path, ...] = (
    Path("main.py"),
    Path("models") / "dino",
    Path("models") / "dino" / "ops",
)
DEFAULT_CLONE_HINT = "git clone https://github.com/IDEA-Research/DINO external/IDEA-Research-DINO"


@dataclass(frozen=True)
class DinoIntegrationStatus:
    root: Path
    ok: bool
    missing_paths: tuple[Path, ...]
    messages: tuple[str, ...]
    clone_hint: str = DEFAULT_CLONE_HINT


def check_dino_tree(
    root: str | Path = DEFAULT_DINO_ROOT,
    *,
    required_paths: Iterable[str | Path] = DEFAULT_REQUIRED_PATHS,
    official_config: str | Path = DEFAULT_OFFICIAL_CONFIG,
) -> DinoIntegrationStatus:
    root_path = Path(root)
    if not root_path.exists():
        return DinoIntegrationStatus(
            root=root_path,
            ok=False,
            missing_paths=(root_path,),
            messages=(f"DINO root not found: {root_path}",),
        )

    required = tuple(Path(path) for path in required_paths) + (Path(official_config),)
    missing = tuple(relative for relative in required if not (root_path / relative).exists())
    messages = tuple(f"missing required DINO path: {root_path / relative}" for relative in missing)
    if not messages:
        messages = (f"DINO integration tree looks complete: {root_path}",)
    return DinoIntegrationStatus(
        root=root_path,
        ok=not missing,
        missing_paths=missing,
        messages=messages,
    )


def format_status(status: DinoIntegrationStatus) -> str:
    lines = [f"dino_root: {status.root}", f"ok: {str(status.ok).lower()}"]
    lines.extend(status.messages)
    if not status.ok:
        lines.append(f"clone_hint: {status.clone_hint}")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```sh
source /data1/liuxuan/activate-py310.sh
PYTHONPATH=src python -m unittest tests.test_dino_integration.DinoIntegrationStatusTest
```

Expected: `OK`.

- [ ] **Step 5: Commit if git metadata is writable**

```sh
source /data1/liuxuan/activate-py310.sh
git add src/rgc_dino/dino_integration.py tests/test_dino_integration.py
git commit -m "feat: add dino integration status checks"
```

---

### Task 4: Check CLI

**Files:**
- Create: `scripts/check_dino_integration.py`
- Modify: `tests/test_dino_integration.py`

- [ ] **Step 1: Add failing CLI tests**

Append to `tests/test_dino_integration.py`:

```python
import subprocess
import sys


class DinoIntegrationCliTest(unittest.TestCase):
    def test_check_cli_returns_2_for_missing_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "missing-dino"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "check_dino_integration.py"),
                    "--dino-root",
                    str(root),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("DINO root not found", result.stdout)
            self.assertIn("clone_hint:", result.stdout)

    def test_check_cli_returns_0_for_fake_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dino"
            (root / "models" / "dino" / "ops").mkdir(parents=True)
            (root / "config" / "DINO").mkdir(parents=True)
            (root / "main.py").write_text("", encoding="utf-8")
            (root / "config" / "DINO" / "DINO_4scale.py").write_text("", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "check_dino_integration.py"),
                    "--dino-root",
                    str(root),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0)
            self.assertIn("ok: true", result.stdout)
```

- [ ] **Step 2: Run CLI tests to verify they fail**

Run:

```sh
source /data1/liuxuan/activate-py310.sh
PYTHONPATH=src python -m unittest tests.test_dino_integration.DinoIntegrationCliTest
```

Expected: subprocess cannot open `scripts/check_dino_integration.py`.

- [ ] **Step 3: Implement CLI**

Create `scripts/check_dino_integration.py`:

```python
#!/usr/bin/env python
"""Check whether the external IDEA-Research DINO tree is present and usable."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rgc_dino.dino_integration import DEFAULT_DINO_ROOT, check_dino_tree, format_status  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dino-root", type=Path, default=ROOT / DEFAULT_DINO_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    status = check_dino_tree(args.dino_root)
    print(format_status(status))
    return 0 if status.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run CLI tests to verify they pass**

Run:

```sh
source /data1/liuxuan/activate-py310.sh
PYTHONPATH=src python -m unittest tests.test_dino_integration.DinoIntegrationCliTest
```

Expected: `OK`.

- [ ] **Step 5: Commit if git metadata is writable**

```sh
source /data1/liuxuan/activate-py310.sh
git add scripts/check_dino_integration.py tests/test_dino_integration.py
git commit -m "feat: add dino integration check cli"
```

---

### Task 5: Bsub DINO Smoke Script Writer

**Files:**
- Create: `scripts/write_bsub_dino_smoke.py`
- Modify: `tests/test_dino_integration.py`

- [ ] **Step 1: Add failing bsub writer test**

Append to `tests/test_dino_integration.py`:

```python
from scripts.write_bsub_dino_smoke import render_bsub_script


class DinoBsubSmokeTest(unittest.TestCase):
    def test_render_bsub_script_checks_dino_without_submitting(self) -> None:
        script = render_bsub_script(
            job_name="dino-smoke",
            queue="normal",
            gpu=1,
            dino_root=Path("external/IDEA-Research-DINO"),
        )

        self.assertIn("#BSUB -J dino-smoke", script)
        self.assertIn(". /data1/liuxuan/activate-py310.sh", script)
        self.assertIn("python scripts/check_dino_integration.py --dino-root external/IDEA-Research-DINO", script)
        self.assertNotIn("bsub <", script)
        self.assertNotIn("python main.py", script)
```

- [ ] **Step 2: Run bsub writer test to verify it fails**

Run:

```sh
source /data1/liuxuan/activate-py310.sh
PYTHONPATH=src python -m unittest tests.test_dino_integration.DinoBsubSmokeTest
```

Expected: import failure for `scripts.write_bsub_dino_smoke`.

- [ ] **Step 3: Implement bsub writer**

Create `scripts/write_bsub_dino_smoke.py`:

```python
#!/usr/bin/env python
"""Write an LSF script for DINO integration smoke checks without submitting it."""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "jobs" / "dino_smoke.lsf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--job-name", default="dino-smoke")
    parser.add_argument("--queue", default="normal")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--dino-root", type=Path, default=Path("external/IDEA-Research-DINO"))
    return parser.parse_args()


def render_bsub_script(*, job_name: str, queue: str, gpu: int, dino_root: Path) -> str:
    return f"""#!/bin/sh
#BSUB -J {job_name}
#BSUB -q {queue}
#BSUB -n 2
#BSUB -gpu "num={gpu}:mode=exclusive_process"
#BSUB -R "rusage[mem=8000]"
#BSUB -o /data1/liuxuan/logs/{job_name}.%J.out
#BSUB -e /data1/liuxuan/logs/{job_name}.%J.err

set -eu
cd {ROOT}
. /data1/liuxuan/activate-py310.sh

python scripts/check_dino_integration.py --dino-root {dino_root}
echo "DINO integration smoke check completed. Compile CUDA ops manually only after approval."
"""


def main() -> int:
    args = parse_args()
    script = render_bsub_script(
        job_name=args.job_name,
        queue=args.queue,
        gpu=args.gpu,
        dino_root=args.dino_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(script, encoding="utf-8")
    print(f"wrote: {args.output}")
    print("submit manually with:")
    print(f"  bsub < {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run bsub writer test to verify it passes**

Run:

```sh
source /data1/liuxuan/activate-py310.sh
PYTHONPATH=src python -m unittest tests.test_dino_integration.DinoBsubSmokeTest
```

Expected: `OK`.

- [ ] **Step 5: Commit if git metadata is writable**

```sh
source /data1/liuxuan/activate-py310.sh
git add scripts/write_bsub_dino_smoke.py tests/test_dino_integration.py
git commit -m "feat: add dino smoke bsub writer"
```

---

### Task 6: README Workflow Update

**Files:**
- Modify: `README.md`
- Test: `tests/test_dino_integration.py`

- [ ] **Step 1: Add failing README contract test**

Append to `tests/test_dino_integration.py`:

```python
class DinoReadmeWorkflowTest(unittest.TestCase):
    def test_readme_documents_dino_integration_workflow(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("DINO 官方工程集成", readme)
        self.assertIn("scripts/check_dino_integration.py", readme)
        self.assertIn("scripts/write_bsub_dino_smoke.py", readme)
        self.assertIn("external/IDEA-Research-DINO", readme)
```

- [ ] **Step 2: Run README test to verify it fails**

Run:

```sh
source /data1/liuxuan/activate-py310.sh
PYTHONPATH=src python -m unittest tests.test_dino_integration.DinoReadmeWorkflowTest
```

Expected: assertion failure for missing DINO integration text.

- [ ] **Step 3: Update README**

Add this section after the v0 baseline workflow:

```markdown
## DINO 官方工程集成

计划书要求检测核心复用 IDEA-Research DINO 4-scale ResNet-50。阶段 1 只做官方工程集成和轻量 smoke check，不在交互会话中编译 CUDA ops、下载权重或启动训练。

预期外部代码位置：

```text
external/IDEA-Research-DINO/
```

检查 DINO 目录结构：

```sh
source /data1/liuxuan/activate-py310.sh
python scripts/check_dino_integration.py --dino-root external/IDEA-Research-DINO
```

生成但不提交 DINO smoke-check LSF 脚本：

```sh
python scripts/write_bsub_dino_smoke.py --output outputs/jobs/dino_smoke.lsf
```
```

- [ ] **Step 4: Run README test to verify it passes**

Run:

```sh
source /data1/liuxuan/activate-py310.sh
PYTHONPATH=src python -m unittest tests.test_dino_integration.DinoReadmeWorkflowTest
```

Expected: `OK`.

- [ ] **Step 5: Commit if git metadata is writable**

```sh
source /data1/liuxuan/activate-py310.sh
git add README.md tests/test_dino_integration.py
git commit -m "docs: add dino integration workflow"
```

---

### Task 7: Full Phase 1 Verification

**Files:**
- No new files

- [ ] **Step 1: Run compile check**

Run:

```sh
source /data1/liuxuan/activate-py310.sh
python -m py_compile src/rgc_dino/*.py scripts/*.py
```

Expected: no output and exit code `0`.

- [ ] **Step 2: Run full test suite**

Run:

```sh
source /data1/liuxuan/activate-py310.sh
PYTHONPATH=src python -m unittest discover -s tests
```

Expected: all tests pass, including existing 21 tests plus the new DINO integration tests.

- [ ] **Step 3: Verify missing-DINO CLI behavior**

Run:

```sh
source /data1/liuxuan/activate-py310.sh
python scripts/check_dino_integration.py --dino-root external/IDEA-Research-DINO
```

Expected if DINO has not been cloned:

```text
dino_root: external/IDEA-Research-DINO
ok: false
DINO root not found: external/IDEA-Research-DINO
clone_hint: git clone https://github.com/IDEA-Research/DINO external/IDEA-Research-DINO
```

Exit code should be `2`.

- [ ] **Step 4: Verify bsub smoke writer**

Run:

```sh
source /data1/liuxuan/activate-py310.sh
python scripts/write_bsub_dino_smoke.py --output outputs/jobs/dino_smoke.lsf
sed -n '1,80p' outputs/jobs/dino_smoke.lsf
```

Expected: script contains environment activation and `python scripts/check_dino_integration.py`, but no `bsub <` command inside the generated script and no training command.

- [ ] **Step 5: Check git status**

Run:

```sh
source /data1/liuxuan/activate-py310.sh
git status --short
```

Expected: only intentional source/config/docs/test changes are shown. Generated outputs under `outputs/` and the external DINO tree are ignored.

- [ ] **Step 6: Commit final phase 1 changes if git metadata is writable**

Run:

```sh
source /data1/liuxuan/activate-py310.sh
git add .gitignore README.md external/README.md configs/dino_r50_4scale.yaml src/rgc_dino/dino_integration.py scripts/check_dino_integration.py scripts/write_bsub_dino_smoke.py tests/test_dino_integration.py
git commit -m "feat: add official dino integration checks"
```

Expected if git metadata is writable: commit succeeds. If git metadata is still read-only, report the exact error and leave files in place.

---

## Self-Review

Spec coverage:

- External DINO location: Task 1.
- DINO R50 4-scale config stub: Task 2.
- Status helper and actionable missing-tree messages: Task 3.
- CLI check with exit codes: Task 4.
- bsub smoke script generation without submission: Task 5.
- README workflow: Task 6.
- CPU-safe verification and no heavy training: Task 7.

Placeholder scan:

- No `TBD` or open-ended implementation steps are used.
- The only unpinned value is the official DINO commit, which the approved spec explicitly leaves as an operational decision; the implementation records the repo URL and target path without inventing a commit hash.

Type/path consistency:

- `check_dino_tree()` returns `DinoIntegrationStatus`.
- `format_status()` prints the same fields expected by CLI tests.
- `render_bsub_script()` in `scripts/write_bsub_dino_smoke.py` matches the import used by the bsub test.
