# 2×RTX 3090 可行性路线：Co-DETR + InternImage-L 主线

> 更新日期：2026-06-20
> 本文是 `docs/FINAL_ROADMAP.md` 的硬件可行性版，专门回答：在 **2×RTX 3090（24GB×2）** 上，如何坚持已线上验证的 **Co-DETR + InternImage-L continuation** 主线冲击更高分。当前 best：epoch20，strict mAP 0.413322，leaderboard 48.335。

---

## 1. 结论

**Co-DETR + InternImage-L continuation 已经是当前线上最高分主线。**

它不是轻量路线，但已在 2×3090 上跑通并把线上分从 RGC-DINO baseline 45.044 提升到 48.335。对手 `urban-visual-recognition` 公开记录 50.8190，说明在继续长训之前，应先补齐 class-wise threshold、NMS/image-side sweet spot、hard validation 和 prediction diagnostics。后续继续沿 Co-DETR InternImage-L 做 checkpoint selection、后处理升级、长训、train-all、高分辨率/单模型后处理，以及再迁移 IR/Depth RGC fusion。

在 2×3090 上的判断：

| 方案 | 可行性 | 说明 |
|---|---:|---|
| Co-DETR + R50 | 高 | 用于 smoke/sanity，不作为最终主线 |
| Co-DETR + Swin-L | 中 | 可作为 Co-DETR 对照线 |
| **Co-DETR + InternImage-L** | **中高风险但可冲** | 主冲榜路线，需要强省显存策略 |
| Co-DETR + InternImage-XL | 极高风险 | 不作为当前主线 |
| 巨型 ViT / DINOv2 ViT-g | 不可行 | 显存和工程成本过高 |

所以最终策略是：

```text
保留当前 Co-DETR + InternImage-L continue epoch20 best checkpoint
→ 后续 continuation/longer train 必须 strict mAP > 0.413322 才提交
→ 立即插入 class-wise threshold / NMS-image-side sweet spot / hard-val / prediction diagnostics
→ 再做 train-all + 单模型 TTA/切图/后处理
→ 最后在 Co-DETR InternImage-L 主线上迁移 IR/Depth RGC fusion
```

---

## 2. 硬件现实

### 当前机器

```text
GPU: 2×RTX 3090
显存: 24GB/card
总显存: 48GB，但不能当作单卡 48GB 使用
建议 batch: 1 image/GPU
```

### 为什么 Co-DETR + InternImage-L 吃紧

显存主要来自：

1. InternImage-L backbone 特征图和 DCNv3/大卷积中间激活。
2. Co-DETR 多辅助监督、多 head/query 训练路径。
3. 三模态 RGC side encoder 与 fusion 分支。
4. 高分辨率输入（800/896 以上会显著放大激活显存）。
5. AdamW optimizer state。

---

## 3. 必选省显存配置

```yaml
training_memory_policy:
  per_gpu_batch_size: 1
  gradient_accumulation_steps: 16-32
  amp: true
  gradient_checkpointing: true
  clip_grad_norm: 0.1
  num_workers: 2-4
  persistent_workers: false_or_careful
  optimizer: AdamW
  optional:
    deepspeed_zero2: only_if_native_training_oom
    cpu_optimizer_offload: last_resort_due_to_speed
```

### 分辨率阶梯

```text
Stage 0: 640 只验证能跑
Stage 1: 640/704 低分辨率训练
Stage 2: 704/768/800 主训练
Stage 3: 832/896 微调或只用于推理
Stage 4: tile inference 作为小目标增强
```

不要一开始就 1024 训练；Co-DETR + InternImage-L 在 3090 上这样做风险很高。

---

## 4. 训练阶段

## Stage 0：Co-DETR-R50 sanity

目标：验证框架、数据、12 类 head、RGC 接口、validation mAP 全链路。

```yaml
model:
  detector: Co-DETR
  backbone: ResNet-50
  fusion: RGC reliability-gated residual
input:
  max_side: 640
training:
  epochs: 1-3
  batch_size_per_gpu: 1
  amp: true
  gradient_checkpointing: true
```

通过标准：

- 不 OOM。
- loss 正常下降。
- validation mAP 能落盘。
- prediction TXT 格式正确。

该阶段通常不提交 leaderboard。

---

## Stage 1：Co-DETR + InternImage-L 低分辨率验证

目标：确认 InternImage-L backbone 权重、Co-DETR head、RGC fusion 三者兼容。

```yaml
model:
  detector: Co-DETR
  backbone: InternImage-L
  backbone_pretrain: public_offline_pretrained
  detector_pretrain: public_detection_pretrained_if_compatible
training:
  max_side_choices: [640, 704]
  epochs: 12-18
  batch_size_per_gpu: 1
  grad_accumulation: 16
  lr_backbone: 1e-6_to_5e-6
  lr_detector: 1e-5_to_2e-5
  lr_fusion: 2e-5_to_5e-5
  amp: true
  gradient_checkpointing: true
  ema: true
```

冻结策略：

```text
Epoch 0-3:
  冻结低层 backbone，先训 Co-DETR head + RGC fusion + 高层 backbone
Epoch 4+:
  解冻全 backbone，小学习率全量训练
```

通过标准：

- backbone 主体权重加载成功，不允许大量 `patch/stage/layer` missing。
- local mAP 明显超过 R50 sanity。
- 预测数量和 score 分布正常。

---

## Stage 2：Co-DETR + InternImage-L 主训练

目标：得到可提交强候选。

```yaml
training:
  max_side_choices: [704, 768, 800, 832]
  epochs: 50-72
  batch_size_per_gpu: 1
  grad_accumulation: 16-32
  amp: true
  gradient_checkpointing: true
  ema: true
  weight_decay: 0.05
  lr_schedule: warmup_cosine_or_step
augmentation:
  horizontal_flip: 0.5
  multiscale_resize: true
  mild_color_jitter: true
  mild_blur_noise: true
  depth_valid_mask: true
  conservative_depth_dropout: true
```

注意：

- 所有几何增强必须 RGB/IR/Depth 同步。
- Mosaic/Copy-Paste/MixUp 不能默认强上；必须先在 validation 上证明。
- 每个 epoch 或固定间隔生成 validation mAP。

---

## Stage 2.5：后处理与验证升级（对手经验吸收）

目标：在继续重 GPU 训练前，用现有 epoch20 best / epoch24 checkpoint 榨干后处理潜力，验证 48.335 是否主要受阈值、NMS、分辨率和 FP 控制限制。

该阶段 GPU 成本主要是推理，不是长训；适合在 2×3090 上并行于训练监控进行。

```yaml
postprocess_sweep:
  candidate_score_threshold: [0.0005, 0.001, 0.0015, 0.003]
  image_max_side: [800, 832, 896, 960]
  nms_iou_threshold: [0.55, 0.65, 0.75]
  candidate_max_detections: 300
  final_max_detections: 100
  class_wise_threshold: greedy_on_validation
validation:
  metric: strict_final_txt_map50_95
  required_reports:
    - class_ap
    - class_prediction_counts
    - score_histogram
    - boxes_per_image
    - top100_truncation
    - hard_val_replay
```

通过标准：

- strict final-TXT mAP 超过 0.413322。
- hard-val 不明显退化。
- 每类预测数量和每图框数没有异常膨胀。
- class thresholds、NMS、image side、候选框数量写入 manifest/promotion metadata。

禁止：

- 不用对手权重或提交包。
- 不用 leaderboard 反复做单类别盲诊断。
- 不复现“1536 + 极低 conf”式高 FP 策略。
- 不做朴素 TTA 平均或弱模型融合。

---

## Stage 3：高分辨率微调

目标：提升高 IoU AP 和小目标召回。

```yaml
finetune:
  init_from: best_stage2_checkpoint
  max_side_choices: [832, 896]
  epochs: 12-24
  lr: 1e-6_to_5e-6
  batch_size_per_gpu: 1
  grad_accumulation: 16-32
  amp: true
  gradient_checkpointing: true
  ema: true
```

如果 896 训练显存不稳定，则只在推理阶段使用 896/960 或 tile inference。

---

## Stage 4：多 fold 验证与 train-all

目标：避免单 fold 偶然性。

流程：

1. fold0 作为历史基准对齐 R50 baseline。
2. 修正或新建 balanced grouped split。
3. Co-DETR + InternImage-L 至少验证 2 个 fold，理想 3 个 fold。
4. 固定最佳 recipe 后，用全部 2000 张训练图训练 final single model。

禁止：

- 不使用测试集伪标签训练。
- 不使用外部训练数据。
- 不用多模型投票/平均作为最终路径。

---

## Stage 5：单模型推理增强

允许并推荐验证：

1. class-wise threshold 搜索。
2. class-wise NMS / Soft-NMS。
3. 单模型多尺度 TTA。
4. 单模型 flip TTA。
5. full image + tile inference，再做 class-wise NMS。

禁止：

- 朴素平均 TTA。
- 弱 fold 融合。
- 简单多模型 voting/averaging ensemble。
- 没有 validation 证明的后处理直接提交。

---

## 5. 时间与资源估计

| 阶段 | 预计时间 | 风险 |
|---|---:|---|
| Stage 0 Co-DETR-R50 sanity | 1-3 天 | 低 |
| Stage 1 InternImage-L 低分辨率 | 3-7 天 | 中高 |
| Stage 2 主训练 | 1-3 周 | 高 |
| **Stage 2.5 后处理与验证升级** | **1-4 天** | **低中** |
| Stage 3 高分辨率微调 | 1-2 周 | 高 |
| Stage 4 多 fold / train-all | 1-3 周 | 高 |
| Stage 5 推理增强 | 3-7 天 | 中 |

总周期：6-10 周。训练时间很长，但符合“最高上限优先”的目标。

---

## 6. 提交门禁

任何候选提交前必须具备：

```text
1. 完整 1000 test TXT
2. zip 根目录结构正确
3. checkpoint path + sha256
4. config path
5. git commit
6. split manifest
7. local validation mAP
8. prediction count / score distribution analysis
9. class-wise thresholds / NMS / image side / candidate and final box counts（如适用）
10. hard-val 或额外验证分布检查（如适用）
11. promotion reason
12. dry-run 成功
```

没有以上证据，不进入 `outputs/submissions/` 自动监控目录。

---

## 7. 回退策略

如果 Co-DETR + InternImage-L 卡住：

1. 回退到 Co-DETR + Swin-L，确认 detector-family 是否带来收益。
2. 回退到 RGC-DINO + InternImage-L，确认 backbone 是否带来收益。
3. 回退到正确初始化的 RGC-DINO + Swin-L，作为强基线。

不要回退到已失败路线：低阈值盲调、朴素 TTA 平均、弱 fold 融合、测试集伪标签训练。

---

## 8. 最终判断

**Co-DETR + InternImage-L 是当前 2×3090 条件下最高上限但仍可落地的主线。**

它比 Swin-L 更难、更慢、更吃显存，但如果目标是最高分，它值得作为主方案。执行时必须坚持：

```text
小模型 sanity
→ 大模型低分辨率验证
→ 后处理与验证升级（class-wise threshold / top100 allocation / hard-val / high-res-NMS sweet spot；当前 anchor 为 GPU1 ep6 person0865/light10625/uav0825/boat003 50.353 / strict 0.437961585）
→ 长训主模型
→ 多 fold 证明
→ train-all final
→ 单模型 TTA/切图
→ 严格提交门禁
```
