# Co-DETR + InternImage-L 高上限冲榜路线（2×RTX 3090）

> 更新日期：2026-06-18  
> 目标：在竞赛规则内，以最高上限为优先，改用 **Co-DETR + InternImage-L** 作为主冲榜路线；Swin-L 仅保留为强基线/对照线。  
> 硬件：2×RTX 3090（24GB×2）。不怕训练时间长，但必须有显存策略、验证门禁和提交纪律。

---

## 0. 不可突破的边界

本路线必须同时满足仓库与竞赛约束：

- **不使用外部训练数据**。
- **项目训练/推理流水线不调用在线 API**。
- **不使用测试集伪标签训练**；1000 张测试图只用于最终离线推理和提交文件生成。
- **不做简单投票/平均 ensemble**；最终提交优先是一个强单模型，允许合法的单模型 TTA/切图推理，但必须经验证集证明。
- **只提交完整 test-set ZIP**：1000 个根目录 `.txt`，带 checkpoint/config/git/split/hash/metric provenance。
- **不再盲交低阈值、朴素 TTA 平均、弱 fold 融合**；这些在本项目历史中已经证明会掉分。

公开预训练权重（ImageNet/COCO/Objects365 等公开模型权重）可作为离线初始化使用，但必须记录来源、路径和加载报告。

---

## 1. 当前判断

### 已知事实

| 项目 | 状态 |
|---|---|
| R50 baseline | fold0 单模型线上最好约 45.044 |
| 低阈值调优 | 已失败，线上约 43.955 |
| 朴素 TTA 平均 | 灾难性失败，约 34.872 |
| 2-fold 融合 | 弱模型拖累，约 44.263 |
| Swin-L 现有提交 | 不能作为架构上限判断；日志显示疑似权重加载/验证闭环问题 |

### 结论

- **Swin-L 不再作为最终主线**：它是重要对照线，用来确认训练/验证/提交闭环是否正常。
- **主冲榜路线改为 Co-DETR + InternImage-L**：上限更高，但显存和工程风险显著更大。
- **不要从一开始就直接长训最大配置**：先通过最小 smoke、R50 sanity、Swin-L/InternImage-L 单模型验证逐级放大，避免几天训练后才发现权重或数据流错误。

---

## 2. 目标架构

### 2.1 主模型：RGC-Co-DETR-InternImage-L

```text
RGB visible image
  └─ InternImage-L backbone（公开预训练）
       └─ multi-scale feature pyramid
            └─ RGC fusion injection point
                 ├─ IR side encoder
                 ├─ Depth side encoder + valid mask
                 └─ reliability-gated residual fusion
                      └─ Co-DETR detection head / query training scheme
                           └─ 12-class predictions
```

### 2.2 相对当前 RGC-DINO 的主要变化

| 模块 | 当前 RGC-DINO | 新主线 |
|---|---|---|
| 检测器 | IDEA DINO | Co-DETR / Co-DINO-style training |
| Backbone | R50/Swin-L 尝试 | InternImage-L |
| 融合 | projected DINO features 前 RGC fusion | 保留 RGC 思路，迁移到 Co-DETR multi-scale feature path |
| IR/Depth | lightweight side encoder | 先沿用，后续再增强 |
| 训练目标 | DINO set prediction | Co-DETR 多辅助监督/协同 query 训练 |
| 提交形态 | 单模型 TXT | 仍然是单模型 TXT；可加单模型 TTA/切图 |

---

## 3. 2×3090 可行性判断

### 3.1 显存风险排序

| 配置 | 2×3090 可行性 | 建议 |
|---|---:|---|
| Co-DETR + R50 + 640 | 高 | 只做 sanity/smoke |
| Co-DETR + Swin-L + 800 | 中 | 对照线/风险预演 |
| Co-DETR + InternImage-L + 640 | 中 | 主线第一阶段 |
| Co-DETR + InternImage-L + 800 | 中高风险 | 主线核心训练 |
| Co-DETR + InternImage-L + 896 | 高风险 | 后期微调/最终候选 |
| Co-DETR + InternImage-XL | 极高风险 | 不作为当前主线 |
| DINOv2 ViT-g / 巨型 ViT | 不可行 | 放弃 |

### 3.2 必选省显存策略

```yaml
hardware:
  gpus: 2x RTX 3090 24GB
  per_gpu_batch_size: 1
  gradient_accumulation: 16-32
  amp: true
  gradient_checkpointing: true
  optimizer: AdamW
  optional_memory_strategy:
    - DeepSpeed ZeRO-2
    - optimizer CPU offload only if native training cannot fit
```

### 3.3 分辨率策略

```text
不要直接 1024 起步。

Stage A: 640 / 704 证明训练和 mAP 正常
Stage B: 800 / 832 主训练
Stage C: 896 或 tile inference 作为后期提分
```

---

## 4. 分阶段执行计划

## Phase 0：地基修复与门禁（必须先做）

**目标**：防止再次出现“训练了很久但权重没加载/没有 mAP/提交不可解释”的情况。

### 任务

1. **权重加载硬门禁**
   - InternImage-L backbone 公开预训练权重必须存在。
   - Co-DETR/Co-DINO detection checkpoint 如可用，必须离线加载并记录。
   - 大量 backbone 主干 key missing 时直接失败。
   - class head 因类别数不匹配而 skip 是正常的，但必须记录。

2. **本地验证闭环**
   - 每个 checkpoint 生成 validation prediction。
   - 计算 mAP@50:95、AP50、AP75。
   - 记录预测数量、每图 box 分布、score 分布。
   - 没有 local mAP 的 checkpoint 不准提交。

3. **提交门禁**
   - 每个 ZIP 必须有 manifest。
   - 使用 promotion 脚本进入 `outputs/submissions/`。
   - dry-run 成功后才允许正式提交。

### 通过标准

- R50/Swin-L 对照线能复现合理 local/online 对应关系。
- 每次训练日志都有明确 pretrained load report。
- validation mAP 自动落盘，不再出现只有表头没有指标的情况。

---

## Phase 1：Co-DETR-R50 最小可行验证（1-3 天）

**目标**：先确认 Co-DETR 训练、数据、评估、提交格式全链路能跑，而不是直接拿大模型烧时间。

```yaml
model:
  detector: Co-DETR
  backbone: ResNet-50
  modalities: RGB + IR + Depth via RGC fusion adapter
training:
  image_max_side: 640
  epochs: 1-3
  batch_size_per_gpu: 1
  amp: true
  gradient_checkpointing: true
validation:
  every_epoch: true
```

### 通过标准

- loss 正常下降。
- 没有 shape mismatch / NaN / OOM。
- 能输出 12 类 prediction。
- 能跑完整 validation mAP。

### 不提交规则

该阶段产物只用于 smoke/sanity，除非出现意外强结果，否则不提交 leaderboard。

---

## Phase 2：Co-DETR + InternImage-L 低分辨率打通（3-7 天）

**目标**：确认 InternImage-L 与 Co-DETR/RGC 三模态路径能稳定训练。

```yaml
model:
  detector: Co-DETR
  backbone: InternImage-L
  backbone_pretrain: public_offline_pretrained
  fusion: reliability_gated_residual
training:
  image_max_side: [640, 704]
  epochs: 12-18
  per_gpu_batch_size: 1
  grad_accumulation: 16
  lr_backbone: 1e-6 to 5e-6
  lr_detector: 1e-5 to 2e-5
  lr_fusion: 2e-5 to 5e-5
  amp: true
  gradient_checkpointing: true
  clip_grad_norm: 0.1
  ema: true
```

### 建议冻结策略

```text
Epoch 0-3:
  冻结 backbone 前半部分，只训 Co-DETR head + RGC fusion + 高层 backbone
Epoch 4+:
  解冻全 backbone，小 lr 全量训练
```

### 通过标准

- validation mAP 明显超过 R50 smoke。
- 预测分布合理，不能极端稀疏或极端密集。
- 显存峰值可控，训练不会频繁 OOM。

---

## Phase 3：Co-DETR + InternImage-L 主训练（1-3 周）

**目标**：形成真正可冲榜的强单模型。

```yaml
training:
  image_max_side_choices: [704, 768, 800, 832]
  epochs: 50-72
  per_gpu_batch_size: 1
  grad_accumulation: 16-32
  amp: true
  gradient_checkpointing: true
  ema: true
  lr_schedule: warmup + step/cosine decay
  weight_decay: 0.05
augmentation:
  horizontal_flip: 0.5
  multiscale_resize: true
  mild_color_jitter: true
  mild_blur_noise: true
  depth_valid_mask: true
  depth_dropout_or_hole_aug: conservative
```

### 增强原则

- RGB/IR/Depth 的几何增强必须严格同步。
- Copy-Paste/Mosaic/MixUp 只能作为验证候选，不能默认强上。
- 如果增强导致 validation AP 下降，立即回滚。

### 通过标准

- 至少 fold0 + 一个额外 fold 上超过 RGC-DINO/Swin-L 对照线。
- best checkpoint 与 EMA checkpoint 都有 mAP 记录。
- checkpoint provenance 完整。

---

## Phase 4：高分辨率微调与小目标强化（1-2 周）

**目标**：提升 mAP@50:95，特别是高 IoU 和小目标召回。

```yaml
finetune:
  init_from: best_phase3_checkpoint
  image_max_side: [832, 896]
  epochs: 12-24
  lr: 1e-6 to 5e-6
  freeze_low_level_backbone: optional
  ema: true
```

### 小目标策略

1. 高分辨率微调。
2. validation 上测试 tile inference。
3. class-wise threshold / NMS 搜索。
4. top100 截断前做 score calibration。

---

## Phase 5：Cross-validation 与 final train-all（1-3 周）

**目标**：避免单 fold 偶然性，确定最终 recipe。

### 推荐流程

1. 保留现有 fold0 作为历史可比基准。
2. 新建更均衡的 grouped stratified split（推荐 5-fold 或修正 3-fold）。
3. Co-DETR + InternImage-L 至少跑 2 个 fold，理想跑 3 个 fold。
4. 固定最佳 recipe 后，用全部 2000 张训练图训练 final single model。

### 注意

- final train-all 不用 test pseudo-label。
- final model 没有自身 validation，所以必须依赖前面 CV 证明 recipe。
- final 提交必须和 CV recipe 一致，不能临时改大量超参。

---

## Phase 6：合法单模型推理增强（最后 1 周）

**目标**：在不做简单多模型 ensemble 的前提下，提升单模型推理质量。

### 允许优先级

1. **Class-wise threshold/NMS 搜索**（必须基于 validation）。
2. **单模型多尺度 TTA**：例如 800/896/960。
3. **单模型 horizontal flip TTA**。
4. **Tile inference**：full image + overlapping tiles，框还原后 class-wise NMS。

### 禁止

- 多模型简单投票/平均。
- 朴素 box score 平均。
- 弱 fold 融合。
- 没有 validation 证明就提交 TTA。

---

## 5. 时间线

| 阶段 | 内容 | 日历时间 | GPU 风险 | 是否可提交 |
|---|---|---:|---:|---|
| Phase 0 | 权重/评估/提交门禁修复 | 1-3 天 | 低 | 否 |
| Phase 1 | Co-DETR-R50 sanity | 1-3 天 | 低中 | 通常否 |
| Phase 2 | Co-DETR + InternImage-L 低分辨率 | 3-7 天 | 中高 | 视 mAP |
| Phase 3 | 主训练 50-72 epoch | 1-3 周 | 高 | 是 |
| Phase 4 | 高分辨率微调 | 1-2 周 | 高 | 是 |
| Phase 5 | 多 fold + train-all | 1-3 周 | 高 | 是 |
| Phase 6 | 单模型 TTA/切图/后处理 | 3-7 天 | 中 | 是 |

总周期：约 6-10 周。训练时间不短，但比盲目多模型融合更符合最高上限目标。

---

## 6. 预期分数与决策门禁

分数预期只能作为路线假设，不能作为提交依据。

| 里程碑 | 目标 |
|---|---|
| Co-DETR-R50 sanity | 证明链路正确，不追求分数 |
| Co-DETR + InternImage-L 低分辨率 | 超过 R50 baseline local mAP |
| 主训练 best fold | 明显超过 Swin-L/R50 对照线 |
| 多 fold recipe | 不依赖单 fold 偶然性 |
| final train-all + TTA/tile | 冲击最终 leaderboard 高分 |

### 提交门槛

候选必须满足：

```text
完整 test ZIP
+ manifest/hash/config/checkpoint/git commit
+ validation mAP 证据
+ 预测数量/score 分布正常
+ dry-run 成功
+ promotion reason 清晰
```

---

## 7. 风险与回退

| 风险 | 影响 | 缓解 |
|---|---|---|
| InternImage/DCNv3 环境编译失败 | 主线卡住 | 先离线准备依赖；保留 Swin-L/Co-DETR-Swin-L 对照线 |
| OOM | 训练中断 | batch=1、checkpointing、AMP、accumulation、降分辨率 |
| 预训练权重 key 不匹配 | 从头训练失败 | 加载硬门禁；主干大量 missing 直接 fail |
| Co-DETR 改造周期长 | 延迟冲榜 | 先 R50 sanity，再 InternImage-L；不要直接最大配置 |
| validation 与 leaderboard 不一致 | 盲交掉分 | 多 fold、预测分布分析、提交间隔纪律 |
| 后处理引入 FP | mAP 掉分 | validation 搜索；禁止朴素平均和弱融合 |

---

## 8. 立即行动清单

1. 新建 `docs/CoDETR_InternImageL_ROADMAP.md` 或以本文作为最终主线。
2. 准备 Co-DETR 代码集成方案：优先离线引入，不破坏现有 RGC-DINO。
3. 准备 InternImage-L backbone wrapper 与权重加载检查脚本。
4. 写 `configs/codetr_internimage_l_stage0.yaml`：R50/Sanity 与 InternImage-L 参数分开。
5. 写 `scripts/write_bsub_codetr_smoke.py`：只生成 LSF，不直接训练。
6. 先跑 Co-DETR-R50 smoke。
7. 再跑 Co-DETR + InternImage-L 640 低分辨率 12-18 epoch。
8. 只有 local mAP、显存、预测分布都正常后，启动 50-72 epoch 主训练。

---

## 9. 最终结论

本项目主线正式调整为：

```text
Co-DETR + InternImage-L + RGC RGB/IR/Depth reliability-gated fusion
→ 多阶段省显存训练
→ 多 fold 验证
→ train-all final single model
→ 合法单模型 TTA / tile inference / class-wise NMS
→ 严格 promotion 后提交
```

这是比 Swin-L 更高上限的路线，但它不是“直接一键训练”的路线。必须先通过权重加载、Co-DETR sanity、低分辨率 InternImage-L 验证，再进入长训。这样既追求最高分，又避免重复踩 TTA、弱融合、无验证提交的坑。
