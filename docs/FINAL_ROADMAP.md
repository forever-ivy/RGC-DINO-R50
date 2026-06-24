# Co-DETR + InternImage-L 高上限冲榜路线（2×RTX 3090）

> 更新日期：2026-06-22
> 当前线上最佳：**Co-DETR + InternImage-L fresh epoch7 + class thresholds，48.727**（strict final-TXT fold0 val mAP `0.4262677082771047`，hard-val `0.28694971837472955`）。
> 对手参考：公开项目 `urban-visual-recognition` 的 **YOLO11M + RGB-guided-RDT + class-wise threshold** 已达 50.8190；其价值是后处理、验证体系和 RGB 主导三模态引导经验，不是替换当前主线或复用其权重。
> 目标：在竞赛规则内，坚持已线上验证的 **Co-DETR InternImage-L fresh/continuation** 作为当前主冲榜路线；吸收对手经验，继续补齐高分辨率-NMS sweet spot / hard validation / prediction diagnostics，再推进 train-all 和 IR/Depth reliability-gated fusion。
> 硬件：2×RTX 3090（24GB×2）。不怕训练时间长，但必须有显存策略、验证门禁、后处理门禁和提交纪律。

---

## 0. 不可突破的边界

本路线必须同时满足仓库与竞赛约束：

- **不使用外部训练数据**。
- **项目训练/推理流水线不调用在线 API**。
- **不使用测试集伪标签训练**；1000 张测试图只用于最终离线推理和提交文件生成。
- **不做简单投票/平均 ensemble**；最终提交优先是一个强单模型，允许合法的单模型 TTA/切图推理，但必须经验证集证明。
- **只提交完整 test-set ZIP**：1000 个根目录 `.txt`，带 checkpoint/config/git/split/hash/metric provenance。
- **不再盲交低阈值、朴素 TTA 平均、弱 fold 融合**；这些在本项目历史中已经证明会掉分。
- **不使用对手 release 权重、提交 ZIP 或对手训练产物**；可学习公开代码/报告中的思想，但本项目训练、推理、提交必须由本地合法流程生成。

公开预训练权重（ImageNet/COCO/Objects365 等公开模型权重）可作为离线初始化使用，但必须记录来源、路径和加载报告。

---

## 1. 当前判断

### 已知事实

| 项目 | 状态 |
|---|---|
| RGC-DINO-R50 baseline | fold0 单模型线上 45.044；已被 Co-DETR continue 超过，退为 fallback |
| Co-DETR + InternImage-L first 12ep | epoch11 raw/strict local 约 0.325/0.324，线上 42.101；说明短训不足 |
| Co-DETR + InternImage-L continue epoch20 | strict final-TXT local 0.413322，线上 48.335；已被 fresh epoch7/class-threshold 版本超过 |
| Co-DETR + InternImage-L fresh epoch7 raw | strict final-TXT local 0.426216676，hard-val 0.286884750，线上 48.6960；已被 class-threshold 版本超过 |
| Co-DETR + InternImage-L fresh epoch7 + class thresholds | strict final-TXT local 0.426267708，hard-val 0.286949718，线上 **48.727**；当前 anchor |
| 2026-06-22 high-res fine-tune | best strict final-TXT local 0.418843656，低于当前 anchor；不 promotion / 不提交 |
| 低阈值调优 | 已失败，线上约 43.955 |
| 朴素 TTA 平均 | 灾难性失败，约 34.872 |
| 2-fold 融合 | 弱模型拖累，约 44.263 |
| Swin-L 现有提交 | 不能作为架构上限判断；日志显示疑似权重加载/验证闭环问题 |
| 对手 urban-visual-recognition | YOLO11M + RGB-guided-RDT + 1408 推理 + class-wise threshold，公开记录 50.8190；主要启发是 FP suppression / 后处理校准 / RGB 主导三模态引导 |

### 结论

- **当前主线已经确定为 Co-DETR + InternImage-L fresh/continuation**：fresh epoch7 + class thresholds 已线上确认 48.727，高于老 RGC-DINO 45.044、continue epoch20 48.335 和 raw fresh epoch7 48.6960。
- **Swin-L/RGC-DINO/R50 不再作为主冲榜路线**：只保留为历史基线、故障回退或对照，不再默认占用磁盘保存大量 checkpoint。
- **当前 best checkpoint 优先级最高**：`outputs/codetr/internimage_l_epoch20_fresh_ft8_fold0_20260621_epoch20_fresh_ft8_direct/best_bbox_mAP_epoch_7.pth` + class thresholds `[0.05, 0.02, 0.003, 0, 0, 0, 0, 0, 0, 0, 0, 0]` 是当前 anchor；后续候选必须 strict final-TXT mAP > `0.4262677082771047` 且以 48.727 为线上基线。
- **三模态 RGC 融合迁移是下一阶段提分，不是回退旧路线**：先在已验证的 Co-DETR InternImage-L 主线上继续 checkpoint selection、后处理校准、高分辨率 sweet spot、hard validation，再迁移 IR/Depth reliability-gated fusion。
- **对手 50.8190 经验要求我们继续做“后处理与验证升级”**：class-wise threshold 已给出小幅 positive；下一步应继续查清 NMS / image-side sweet spot / prediction distribution，而不是回到旧 epoch20 门槛。

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

## Phase 3.5：对手经验吸收 / 后处理与验证升级（立即插入，1-4 天）

**目标**：不改变当前强单模型主线，先吸收 `urban-visual-recognition` 的成功经验，把当前 Co-DETR InternImage-L checkpoint 的后处理、误检控制和验证可信度补齐。当前优先基于 fresh epoch7 + class-threshold anchor 继续 NMS / image-side / prediction diagnostics；epoch20 best 和 epoch24 仅作历史对照。

### 背景事实

对手公开项目的 50.8190 主要来自：

```text
YOLO11M + RGB-guided-RDT
+ 1408 推理 sweet spot
+ 更保守全局 conf
+ class-wise threshold（person003 等）
+ 提交前严格校验
```

其最可迁移经验不是 YOLO11M 本身，而是：

1. 低阈值取候选后，用**类别级阈值**抑制低置信度 FP。
2. 高分辨率存在 sweet spot；过高分辨率 + 极低 conf 会引入 FP 并掉分。
3. 随机 validation 与平台存在分布差异，必须补 hard validation。
4. RGB 主干保持主导，IR/Depth 只做可靠性引导；不要无脑强融合。

详细复盘见 `docs/OPPONENT_URBAN_VISUAL_RECOGNITION_LESSONS.md`。

### 必做任务

1. **class-wise threshold 支持**
   - 在 Co-DETR/DINO 推理链路中支持 `--class-score-thresholds thresholds.json`。
   - 区分 `candidate_score_threshold` 与最终 per-class threshold。
   - 支持先保留更多候选（例如每图 300），再 class-wise filter + class-wise NMS + top100。

2. **validation raw prediction cache**
   - 优先对当前 fresh epoch7 anchor 输出低阈值 validation raw predictions；epoch20 best / epoch24 仅作历史对照。
   - 缓存字段至少包含：sample_id、class_id、score、box、image side、NMS setting、checkpoint。
   - 后续 threshold sweep 不重复跑 GPU 推理。

3. **class-wise threshold greedy sweep**
   - 目标指标：strict final-TXT `mAP@50:95`。
   - 可选目标：`mAP - penalty * boxes_per_image`，防止通过堆框虚高 validation。
   - 输出：`thresholds.json`、class AP、class box counts、boxes/image、score histogram。

4. **高分辨率 / NMS / threshold 联合 sweep**
   - 候选：`image_max_side = 800 / 832 / 896 / 960`（按显存和速度调整）。
   - 候选：`nms_iou = 0.55 / 0.65 / 0.75`。
   - 候选：`candidate_score_threshold = 0.0005 / 0.001 / 0.0015 / 0.003`。
   - 每个设置都必须记录 prediction count 和 top100 截断影响。

5. **prediction diagnostics**
   - 每类预测数量。
   - 每类 score 分布。
   - 每图框数分布。
   - AP50/AP75/AP90 gap。
   - small/medium/large AP。
   - top100 截断前后被丢弃框的类别和分数。

6. **hard validation**
   - 在现有 grouped split 基础上额外标注 hard-val 子集，不替代原 fold。
   - hard-val 覆盖：低光、夜间/弱光、小目标、遮挡、密集人群、深度无效、红外弱响应、稀有类。
   - 后处理候选必须 normal val 提升且 hard-val 不崩，才进入 test ZIP。

7. **RGB-guided-RDT 低成本 ablation**
   - 不切换主线到 YOLO11M。
   - 可实现可选预处理：`RGB * (0.85 + 0.30 * attention(IR_CLAHE, near_depth))`，仅作为 ablation 或 gate quality feature。
   - 更推荐把 IR CLAHE saliency、depth valid near-depth saliency 加入 RGC gate 的质量统计，而不是直接替换 RGB 输入。

### 通过标准

- 至少一个 postprocess candidate 的 strict final-TXT mAP 超过 `0.4262677082771047`，且 prediction distribution 合理。
- hard-val 不出现明显退化或极端框数膨胀。
- class-wise threshold、image side、NMS、candidate/final box counts 写入 manifest / promotion metadata。
- 若未超过当前 48.727 anchor，不提交，但保留诊断报告指导下一步微调/RGC fusion。

### 禁止

- 不用对手权重或提交包。
- 不通过多次平台提交做单类别盲诊断。
- 不复现“1536 + 极低 conf”式高 FP 策略。
- 不做朴素 TTA 平均或弱模型融合。

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
3. class-wise threshold / NMS 搜索（沿用 Phase 3.5 的 raw prediction cache 与 sweep 工具）。
4. top100 截断前做 score calibration，并记录截断损失。

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
| **Phase 3.5** | **对手经验吸收：class-wise threshold / hard-val / 高分辨率-NMS sweet spot** | **1-4 天** | **低中** | **是，若过门禁** |
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
| Co-DETR + InternImage-L continue epoch20 | 已达成：strict mAP 0.413322，线上 48.335；已被 fresh epoch7 超过 |
| Co-DETR + InternImage-L fresh epoch7 | 已达成：strict mAP 0.426216676，hard-val 0.286884750，线上 48.6960；已被 class-threshold 版本超过 |
| Co-DETR + InternImage-L fresh epoch7 + class thresholds | 当前 anchor：strict mAP 0.426267708，hard-val 0.286949718，线上 48.727 |
| 后续 continuation / longer train | 必须 strict final-TXT mAP > 0.426267708 且 hard-val 不崩才允许提交 |
| class-wise threshold / NMS / high-res postprocess | 优先基于当前 fresh epoch7 + class-threshold anchor 执行；必须 strict final-TXT mAP > 0.426267708，prediction distribution 合理，hard-val 不崩；旧 eval_s768/s832/s896 因 `(1333, side)` 被 width cap 成同一 `1333×750` 输出，修正 s832 smoke strict 0.422748 仍低于 anchor |
| train-all / high-res / TTA / tile | 必须以线上 48.727 为 baseline，先 validation + hard-val 证明再提交；2026-06-22 high-res fine-tune strict 0.418843656 已淘汰 |
| IR/Depth RGC fusion 迁移 | 必须在 Co-DETR InternImage-L 主线上验证，不能回退旧 RGC-DINO 体系盲训 |

### 提交门槛

候选必须满足：

```text
完整 test ZIP
+ manifest/hash/config/checkpoint/git commit
+ strict final-TXT validation mAP 证据
+ Co-DETR InternImage-L 候选必须 strict mAP > 0.4262677082771047（当前 fresh epoch7 + class-threshold anchor）
+ promotion metadata 的 leaderboard_baseline 必须 >= 48.727
+ 预测数量/score 分布正常
+ class-wise threshold / NMS / image side / candidate box count 等后处理参数完整记录（如适用）
+ hard-val 或额外验证分布检查没有明显退化（如适用）
+ dry-run/leaderboard 确认链路正常
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
8. 对当前 fresh epoch7 anchor 继续补 `class-wise threshold sweep`、`prediction diagnostics`、`high-res/NMS sweet spot` 和 `hard-val`；旧 epoch20/epoch24 仅作对照。
9. 只有 local mAP、hard-val、显存、预测分布都正常后，启动 50-72 epoch 主训练、train-all 或其它高风险实验；已完成的 2026-06-22 high-res fine-tune strict 退步，不进入 test ZIP / promotion / 提交。

---

## 9. 最终结论

本项目当前主线正式调整为：

```text
Co-DETR + InternImage-L continuation（当前已验证 fresh epoch7）
→ strict final-TXT sweep 选择 best checkpoint
→ 吸收对手经验：class-wise threshold / 高分辨率-NMS sweet spot / hard validation / prediction diagnostics
→ 以 48.727 / strict mAP 0.4262677082771047 作为新提交门槛
→ 更长训练、高分辨率微调、train-all final single model
→ 合法单模型 TTA / tile inference / class-wise NMS
→ 在 Co-DETR InternImage-L 主线上迁移 RGB/IR/Depth reliability-gated fusion
→ 严格 promotion 后提交
```

当前线上已验证最优是 fresh epoch7 checkpoint `outputs/codetr/internimage_l_epoch20_fresh_ft8_fold0_20260621_epoch20_fresh_ft8_direct/best_bbox_mAP_epoch_7.pth` + class thresholds，分数 48.727。旧 epoch20/48.335、raw fresh/48.6960、RGC-DINO/Swin/R50 路线都不再作为主冲榜 anchor；其 checkpoint 可以按需清理以节省空间，但需保留文档、日志、ranking、提交 ZIP/manifest 等复现实验证据。
