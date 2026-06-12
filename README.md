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
│   └── inspect_labels.py
├── source/
│   └── labels/
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
python scripts/inspect_labels.py --labels source/labels
python -m py_compile src/rgc_dino/*.py scripts/*.py
PYTHONPATH=src python -m unittest discover -s tests
```

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

## 开发边界

当前仓库只完成了初始化骨架和轻量数据/环境检查。后续训练、模型、数据集转换、验证器和提交打包脚本应继续按 `doc/RGC-DINO-R50 三模态单模型冲榜工程方案.pdf` 落地。

不要直接在交互会话中启动完整训练或长时间 GPU 任务。真实训练应准备命令或 LSF `bsub` 脚本后再执行。
