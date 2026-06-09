# Repository Guidelines

## Environment & HPC Rules
This project runs on the Yanshan University 3090 server. Do not run heavy training, benchmarks, simulations, or long GPU/CPU jobs directly unless the user explicitly confirms it is allowed for the current session.

Prefer generating commands or LSF `bsub` scripts for training jobs. Keep all project files, datasets, environments, caches, and outputs under `/data1/liuxuan`; do not use `/home` except for minimal configuration. Never use `sudo`, `apt install`, or `apt upgrade`.

Use the project Python environment with:

```sh
bash
source /data1/liuxuan/activate-py310.sh
```

Expected environment:

- Python 3.10.20
- PyTorch 2.12.0+cu126
- CUDA build 12.6

## Project Structure
Current repository contents:

- `configs/`: YAML experiment and runtime configuration files.
- `data/`: small metadata and manifest files only; large datasets stay outside the repo.
- `doc/`: competition documents, official PDFs, and project planning notes.
- `scripts/`: lightweight environment, data inspection, launch, and evaluation helpers.
- `source/labels/`: annotation label text files.
- `src/rgc_dino/`: Python utilities and future model/data/evaluation modules.
- `tests/`: lightweight CPU-safe tests.

Use this layout for new code unless the project later defines another one:

```text
.
├── src/              # model, data, training, evaluation code
├── configs/          # YAML/JSON experiment configs
├── scripts/          # launch, preprocessing, evaluation helpers
├── data/             # small metadata only; large datasets stay in /data1/liuxuan/datasets
├── outputs/          # checkpoints, logs, predictions; ignored by git
├── tests/            # lightweight tests
└── README.md
```

Large datasets should live in:

```text
/data1/liuxuan/datasets
```

Project outputs should live in either:

```text
/data1/liuxuan/logs
```

or a project-local ignored `outputs/` directory.

## Development Commands
Use lightweight checks during development:

```sh
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m py_compile path/to/file.py
```

Do not start full training from Codex without explicit user approval. For training, prepare a command or `bsub` script, for example:

```sh
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/default.yaml
```

## Coding Style
Write Python code in clear modules with small functions. Prefer configuration files over hard-coded paths. Use absolute `/data1/liuxuan/...` paths for datasets and outputs when running on the server.

Keep secrets, tokens, passwords, and API keys out of code, logs, README files, and prompts.

## Git & GitHub
Activate the environment before using Git tools:

```sh
source /data1/liuxuan/activate-py310.sh
```

`git` and `gh` are installed in `/data1/liuxuan/envs/py310`. Do not commit datasets, checkpoints, caches, `.env` files, credentials, or large archives. Use `.gitignore` for `outputs/`, `runs/`, `checkpoints/`, `*.pt`, and `*.pth`.

## Testing & Verification
Prefer small CPU-safe tests first. GPU checks should be minimal, such as importing `torch` and checking device visibility. Avoid long-running jobs on the login/session node.

For real training verification, use a short smoke test only when explicitly approved, or submit through the cluster-approved workflow.

## Acknowledgement
If this project contributes to a paper or result using this HPC resource, include:

```text
本论文的数值计算得到了燕山大学超算中心的计算支持和帮助
```
