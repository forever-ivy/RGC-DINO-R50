# RGC-DINO-R50 / Co-DETR InternImage-L

面向城市场景视觉多模态目标检测竞赛工程仓库。项目名称仍沿用 `RGC-DINO-R50`，但当前冲榜方向已经确认：**Co-DETR + InternImage-L 单模型主线**，围绕 strict final-TXT 验证、class-wise threshold、legal top100 allocation、hard validation 和 prediction diagnostics 做候选筛选。

当前线上最佳 anchor：`Co-DETR InternImage-L GPU1 load-from epoch6 + top100 allocation person0865/light10625/uav0825/boat003`，leaderboard **50.353**，strict final-TXT mAP `0.4379615851682616`，hard-val `0.29545499238138817`。

旧 `RGC-DINO/DINO-R50` 三模态路线不再作为主冲榜路线；保留为 fallback、对照和后续 IR/Depth reliability-gated fusion 迁移的技术基底。

## 竞赛约束

- 输入：空间对齐的 `RGB + Infrared + Depth` 三模态图像。
- 输出：每张测试图一个 TXT，字段为 `[class_id, norm_center_x, norm_center_y, norm_w, norm_h, confidence]`。
- 类别：12 类，编号 `0-11`。
- 指标：`mAP@50:95`。
- 禁止：外部训练数据、项目流水线在线 API、测试集伪标签训练、简单投票/平均 ensemble。
- 提交：只提交完整 1000-file test ZIP；validation / OoF / debug / empty / partial ZIP 一律不能提交。

## 当前文档入口

- `docs/README.md`：文档导航与当前状态。
- `docs/FINAL_ROADMAP.md`：当前主线方案，Co-DETR + InternImage-L + top100 allocation 门禁。
- `docs/2x3090_FEASIBLE_ROADMAP.md`：2×RTX 3090 下的执行路线与显存策略。
- `docs/CODETR_INTERNIMAGE_L_TRAINING_HANDOFF.md`：Co-DETR 训练交接和外部依赖检查。
- `docs/OPPONENT_URBAN_VISUAL_RECOGNITION_LESSONS.md`：对手公开方案经验复盘，只学习后处理/验证经验，不使用其权重或产物。
- `doc/official/`：官方赛题与流程 PDF，必须保留。

旧方向 PDF、历史方案、旧自动化文档、一次性训练日志和 agent 旧计划已清理；关键失败经验已合并进当前路线文档。

## 当前仓库结构

```text
.
├── AGENTS.md                         # 通用仓库操作规范
├── CLAUDE.md                         # Claude Code 项目指令
├── README.md                         # 当前入口说明
├── configs/                          # Co-DETR / legacy DINO 配置
│   ├── codetr_internimage_l_*.py
│   ├── codetr_r50_*_mm_config.py
│   ├── codetr_internimage_l_stage0.yaml
│   └── default.yaml                  # legacy RGC-DINO 配置
├── doc/official/                     # 官方比赛文件
├── docs/                             # 当前策略/交接/复盘文档
├── external/                         # 第三方代码树（gitignored）
│   ├── Co-DETR/
│   ├── IDEA-Research-DINO/
│   └── InternImage-master/
├── scripts/                          # 数据、训练脚本生成、推理、后处理、提交自动化
├── source/                           # 训练集与测试/提交图片集
├── src/rgc_dino/                     # 数据、指标、提交、legacy RGC-DINO/RDT 工具
└── tests/                            # CPU-safe 单元测试
```

## 环境

所有工作保持在 `/data1/liuxuan/` 下，不用 `sudo` / `apt install` / `apt upgrade`，不要在交互会话直接启动长 GPU 训练。

```sh
source /data1/liuxuan/activate-py310.sh
python --version
python scripts/check_environment.py
```

预期环境：

- Python 3.10.20
- PyTorch 2.12.0+cu126
- CUDA build 12.6

## 轻量检查

```sh
source /data1/liuxuan/activate-py310.sh
python scripts/check_environment.py
python scripts/inspect_labels.py --labels source/训练集/labels --max-errors 10
python scripts/inspect_dataset.py \
  --root source/训练集 \
  --labels source/训练集/labels \
  --manifest outputs/manifests/train_2000.jsonl
python -m py_compile src/rgc_dino/*.py scripts/*.py
PYTHONPATH=src python -m unittest discover -s tests
```

## 当前 Co-DETR 主线工作流

### 1. 外部依赖与数据准备

```sh
source /data1/liuxuan/activate-py310.sh
python scripts/check_codetr_environment.py
python scripts/check_codetr_integration.py --codetr-root external/Co-DETR
python scripts/prepare_codetr_training.py --no-require-weights
```

常用外部路径：

- `external/Co-DETR/`
- `external/IDEA-Research-DINO/`（legacy DINO / 对照）
- `external/InternImage-master/`
- `/data1/liuxuan/checkpoints/internimage/internimage_l_public_pretrain.pth`
- `/data1/liuxuan/checkpoints/codetr/codetr_internimage_l_public_pretrain.pth`

### 2. 训练与 continuation

交互会话只生成/检查 LSF 脚本，不直接跑长训练：

```sh
python scripts/write_bsub_codetr_train.py --output outputs/jobs/codetr_train.lsf
python scripts/write_bsub_codetr_continue.py --output outputs/jobs/codetr_continue.lsf
# 手动确认资源后再提交：
# bsub < outputs/jobs/codetr_continue.lsf
```

当前 anchor checkpoint/recipe 以 `docs/FINAL_ROADMAP.md` 为准；后续候选必须严格超过 strict mAP `0.4379615851682616`，并以 leaderboard `50.353` 作为 promotion baseline。

### 3. 验证、后处理与提交格式

核心闭环：checkpoint → validation raw prediction/cache → strict final-TXT mAP → class threshold / top100 allocation sweep → hard-val → manifest/promotion metadata。

常用脚本：

```sh
python scripts/cache_codetr_predictions.py --help
python scripts/sweep_codetr_class_thresholds.py --help
python scripts/sweep_codetr_top100_allocation.py --help
python scripts/sweep_codetr_submission_params.py --help
python scripts/codetr_results_to_submission.py --help
```

生成 test 候选并 promotion 时优先用项目封装：

```sh
bash scripts/run_codetr_test_and_promote.sh --help
python scripts/promote_submission_candidate.py --help
```

## 竞赛自动化与提交纪律

敏感文件不要打印、不要提交：

- `outputs/cookies.json`
- `outputs/aicomp_auth.json`

推荐后台会话：

```sh
tmux attach -t competition_monitor
tail -f outputs/monitor/monitor.log
cat outputs/monitor/monitor_state.json
```

重启命令见 `CLAUDE.md` / `AGENTS.md`。自动提交只监控 `outputs/submissions/*.zip`，且候选必须有清晰的 local metric、后处理参数、manifest 和 promotion reason。

## Legacy / fallback 路线

`RGC-DINO/DINO-R50` 代码仍可用于对照、回退和后续 fusion 迁移：

- `scripts/train_rgc_dino.py`
- `scripts/infer_rgc_dino.py`
- `configs/default.yaml`
- `configs/dino_r50_4scale.yaml`
- `src/rgc_dino/models/`

但不要再把 legacy RGC-DINO 文档或旧 PDF 当作当前主线依据。当前主线只看 Co-DETR + InternImage-L 文档和 promotion evidence。

## 开发边界

- 长 GPU 训练：先写 LSF / command，用户确认后再提交。
- 新提交候选：必须通过 strict final-TXT、hard-val/分布检查、promotion metadata。
- 已证明失败的方向：低阈值盲调、朴素 TTA 平均、弱 fold 融合、简单投票/平均 ensemble、测试集伪标签训练。
- 官方规则和比赛流程以 `doc/official/` 为准。
