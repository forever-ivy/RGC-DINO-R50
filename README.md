# RGC-DINO-R50

面向城市场景视觉多模态目标检测的竞赛工程仓库。项目目标是在官方规则约束下，落地 `RGC-DINO-R50` 单模型三模态检测主线：以 DINO-R50 为检测核心，使用可靠性门控残差跨模态融合处理 RGB、Infrared、Depth 三模态输入。

## 项目定位

- 赛题：全球校园人工智能算法精英大赛算法挑战赛，面向城市场景的视觉多模态目标检测。
- 输入：空间对齐的 `RGB + Infrared + Depth` 三模态图像。
- 输出：每张测试图对应一个 TXT 文件，字段为 `[class_id, norm_center_x, norm_center_y, norm_w, norm_h, confidence]`。
- 类别：12 类，编号从 `0` 到 `11`。
- 指标：`mAP@50:95`。
- 约束：不使用外部训练数据，不调用在线 API，不做简单投票/平均集成。

## 当前结构

```text
.
├── AGENTS.md
├── README.md
├── configs/
│   └── default.yaml
├── data/
│   └── README.md
├── doc/
├── scripts/
│   ├── check_environment.py
│   ├── evaluate_predictions.py
│   ├── infer_baseline.py
│   ├── inspect_dataset.py
│   ├── inspect_labels.py
│   ├── make_splits.py
│   └── make_submission.py
├── source/
│   ├── AIC2026_PHASE_1_1000/
│   └── 训练集/
├── src/
│   └── rgc_dino/
└── tests/
```

## 环境

在服务器上使用项目 Python 环境：

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

不启动训练时，可以先运行：

```sh
python scripts/inspect_labels.py --labels source/训练集/labels
python scripts/inspect_dataset.py \
  --root source/训练集 \
  --labels source/训练集/labels \
  --manifest outputs/manifests/train_2000.jsonl
python -m py_compile src/rgc_dino/*.py scripts/*.py
PYTHONPATH=src python -m unittest discover -s tests
```

## v0 Baseline 工程闭环

当前 v0 不包含真实 DINO 训练实现，也不会在交互会话中启动重训练。它先打通可验证的工程闭环：

1. 三模态样本索引：`visible/`、`infrared/`、`depth/` 按文件 stem 对齐。
2. 标签和预测 TXT 校验：训练标签 5 字段，提交预测 6 字段。
3. CPU 版 `mAP@50:95` 评估器：用于验证预测 TXT 格式和基础指标路径。
4. 分组 3-fold split manifest：冻结验证入口，避免后续训练时临时切分。
5. v0 no-detection 推理入口：生成可复现的合法预测目录，后续替换为真实模型推理。
6. LSF 训练脚本生成：只写 `bsub` 脚本，不自动提交训练作业。

推荐顺序：

```sh
source /data1/liuxuan/activate-py310.sh

# 1. 检查三模态对齐，并写出 JSONL manifest
python scripts/inspect_dataset.py \
  --root source/训练集 \
  --labels source/训练集/labels \
  --manifest outputs/manifests/train_2000.jsonl

# 2. 检查官方标签。当前样例标签中有 4 个轻微越界框，默认只告警。
python scripts/inspect_labels.py --labels source/训练集/labels --max-errors 10

# 3. 生成冻结 split manifest。当前样例标签需显式裁剪 4 个轻微越界框。
python scripts/make_splits.py \
  --labels source/训练集/labels \
  --folds 3 \
  --output-dir outputs/splits \
  --clip-labels

# 4. 运行 v0 no-detection baseline 推理，生成合法预测目录。
python scripts/infer_baseline.py \
  --dataset-root source/AIC2026_PHASE_1_1000 \
  --output-dir outputs/baseline_predictions

# 5. 校验并打包提交 ZIP。
python scripts/make_submission.py \
  --dataset-root source/AIC2026_PHASE_1_1000 \
  --submission-dir outputs/baseline_predictions \
  --zip-path outputs/baseline_predictions.zip

# 6. 用标签目录驱动评估预测目录。若使用当前样例标签，可显式启用轻微越界裁剪。
python scripts/evaluate_predictions.py \
  --labels source/训练集/labels \
  --predictions outputs/baseline_predictions \
  --clip-labels

# 7. 生成但不提交 RGC-DINO fold0 短训 LSF 脚本。
python scripts/write_bsub_train.py --output outputs/jobs/train_rgc_dino_fold0_short.lsf
```

当前仓库里的 `source/训练集` 是 2000 张三模态对齐训练样本，包含同名 `labels/`；`source/AIC2026_PHASE_1_1000` 是 1000 张无标签三模态预测/提交图片集。因此训练、split、本地标签检查默认使用 `source/训练集`，生成提交时使用 `source/AIC2026_PHASE_1_1000`。

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

生成当前 RGC-DINO fold0 短训脚本。脚本会先预计算质量特征缓存，再优先从官方 DINO R50 4scale checkpoint 初始化新 RGC-DINO；若 `outputs/checkpoints/checkpoint0011_4scale.pth` 不存在，则退回已有 A0 checkpoint。训练启用 depth valid mask、训练多尺度最长边抖动、水平翻转、gate 日志和 loss-only validation：

```sh
python scripts/write_bsub_train.py --output outputs/jobs/train_rgc_dino_fold0_short.lsf
# 手动确认资源后再提交：
# bsub < outputs/jobs/train_rgc_dino_fold0_short.lsf
```

## 开发边界

当前仓库完成了 v0 工程闭环的轻量基础设施。后续真实训练、模型结构、数据增强、fold 切分和冲榜策略应继续按 `doc/RGC-DINO-R50 三模态单模型冲榜工程方案.pdf` 落地。

不要直接在交互会话中启动完整训练或长时间 GPU 任务。真实训练应准备命令或 LSF `bsub` 脚本后再执行。
