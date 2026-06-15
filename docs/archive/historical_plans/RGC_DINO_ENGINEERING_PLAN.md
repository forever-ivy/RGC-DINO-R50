# RGC-DINO-R50 三模态单模型冲榜工程方案

> 基于PDF方案文档转换 - 面向城市场景视觉多模态目标检测

## 项目概述

**目标**：在AIC2026多模态目标检测竞赛中，使用单个RGC-DINO模型融合RGB、红外、深度三模态数据，实现高精度目标检测。

**核心创新**：
- 质量感知门控融合机制（Reliability-Gated Fusion）
- 三模态协同训练
- 课程学习与难例挖掘

---

## 技术架构

### 1. 模型设计

#### 1.1 Backbone
- **RGB主干**：DINO 4-scale ResNet-50
  - 4个feature level输出
  - 通道数：256
  - 参数：enc_layers=6, dec_layers=6, nheads=8
  - num_queries=900, num_feature_levels=4

#### 1.2 Side Encoders

**IR Side Encoder**：3-stage轻量级CNN
```
Conv(1->32, 3x3, s2) -> GN -> SiLU
ResBlock(32->64, s2)
ResBlock(64->128, s2)
ResBlock(128->256, s2)
```
- FPN-like结构：lateral 1×1卷积输出4个256通道特征
- 与RGB主干4个尺度对齐（P3/P4/P5/P6）
- 使用bilinear interpolate + 1×1 conv对齐

**Depth Side Encoder**：结构同IR encoder
- 输入：[log_depth, inverse_depth, valid_mask]
- 3通道输入，输出4个256通道特征图

#### 1.3 质量特征提取器

**离线静态特征**（24维）：
- RGB: brightness_mean/std, edge_density, entropy, laplace_var, local_contrast, overexposed_ratio, underexposed_ratio
- IR: mean, std, entropy, laplace_mean, tophat_mean, blackhat_mean, hot_ratio, cold_ratio
- Depth: valid_ratio, hole_ratio, near_ratio, far_ratio, mean/std(log_depth), mean(inv_depth), edge_density

**在线动态特征**：
- 预计算质量特征的k-means聚类
- 通过dataloader在线查询

#### 1.4 Reliability-Gated Residual Fusion

**参数**：
- hidden_dim: 256
- n_scales: 4（对应P3/P4/P5/P6）
- q_dim: 24（质量特征维度）
- alpha_prior: 0.35
- gate_floor: 0.05
- gate_ceil: 0.90

**融合流程**：
```python
# 1. 质量特征 -> prior
ĝᵢ = softmax(MLP_gate(GAP(F_r), GAP(F_i), GAP(F_d), q̄))

# 2. Gating residual
gᵢ = Normalize(clip((1-α)ĝᵢ + αpᵢ, g_floor, g_ceil))

# 3. 融合
Auxᵢ = gᵢ,ᵣϕᵢ,ᵣ(Fᵢʳᵍᵇ) + gᵢ,ᵢᵣϕᵢ,ᵢᵣ(Fᵢᵈᵉᵖᵗʰ)
Fᵢᶠᵘˢᵉᵈ = Fᵢʳ + ψᵢ([Fᵢʳ, Auxᵢ], Fᵢʳ ⊙ Auxᵢ])
```

**设计原则**：
1. RGB是主模态，不在early stage引入cross-attention堆叠
2. 门控不是纯黑盒，通过质量先验 + 特征门控共同决策
3. 设置floor/ceil限制门控范围，避免极端退化

---

### 2. 跨学科模块落地

#### 2.1 信息论模块：核态信息量估计

**目的**：筛选高质量样本用于质量回归训练。

**实现**：基于k-means聚类质量特征空间，计算样本的"信息意外度"。
- 理论依据：距离质量特征中心越远 = 更意外的模态配合
- 实现位置：`src/model/quality_features.py`
- 集成方式：offline计算，训练时在线查询

#### 2.2 信号处理模块：IR/Depth物理重编码

**IR处理**：
- 不假设"高温=白色"，根据官方定义真实通道没有虚假彩色
- Laplacian是检验曝光异常的关键
- OpenCV实现：Laplacian差分 + top-hat/opening + black-hat closing
- RGB-D文献与DFormer相关工作说明depth的垂直度bias较大，需合理降权

**Depth处理**（3通道）：
```python
uint8[H,W,3] -> collapse(channel0 or mean) -> robust normalize -> Laplacian -> 
top-hat -> black-hat -> thermal saliency -> uint8[H,W]
uint16[H,W] -> clip[300,20000] -> valid_mask -> log_depth -> inverse_depth ->
depth_edge
```
- 实际导入：hole_ratio和valid_mask，depth垂直度bias确认
- IR/Depth side encoder输入：[log_depth, inverse_depth, valid_mask]
- P0必做：IR折后3通道差异 vs 单通道

#### 2.3 跨学科模块：可靠性门控 gain scheduling

**目的**：让模型学习融合增益表，自主权衡模态可信度。

**理论依据**：gain scheduling - 控制论中经典方法，适应性增益调度。

**实现**（ReliabilityGatedFusion）：
```python
# 先用固定alpha和bucket划分训练
g_norm = q_norm 共同预测 g_hat_l
# 最后融合时：concat + 1x1
```

**参数设定**：
- gate_mean不长期固定[>0.9, <0.05, <0.05]
- 保持融合bucket中：IR/Depth gate降级，但不降至0.08级别
- 失败风险：bucket选得不对，IR热量聚类过碎
- Stage B前半拆除或弱化gate head，支持domain adversary

---

### 3. 会棒抗摇模块

#### 3.1 质量特征：小数据抗常偏

**问题**：仅2000个样本，不够分裂，缺乏异质性。

**解决方案**：
1. **信息论先验**：模态质量加权不是"黑盒"，用质量特征引导
   - 计算24维质量静态特征
   - k-means聚类作为prior
   - 训练时同时优化gate和prior
   
2. **信号处理提取域无关特征**：
   - 使用median + MAD代替mean/std
   - 热量统计：bucket/mAP与worst-bucket mAP平衡
   - pH值pHash与轻embedding替代group clustering

3. **第3节稀疏split与异帮样本重复**：
   - 按grouped stratified 3-fold划分
   - 仅用pseudo-domain分布做bucket，不代入fold
   - split_manifest.json保障可复现

#### 3.2 模型抗摇模块：小数据抗常规训练噪声

**问题**：样本少易过拟合，需domain-agnostic特征。

**策略**：
1. **域对抗/nuisance-domain**：
   - k=4个nuisance domains（normal daylight / low light / thermal dominant / sparse-noisy depth）
   - 用GRL实现feature-level adversarial训练
   - 适应式教师网络（Adaptive Teacher）
   - 损失函数：
     ```
     L_domain = CE(D(GRL(f_global; λᵢ)), d_cluster)
     ```
   - λ_i采用warmup：前6 epoch为0，后线性到0.03-0.05
   - 代码位置：`src/model/nuisance_domain_head.py`

2. **Spill与bucket AP对比**：
   - 各bucket AP不长期是否定的
   - 失败风险：连续split或域划分不当

#### 3.3 认知科学模块：由易到难的课程学习

**Difficulty Score定义**：
```python
d = w₁·low_light + w₂·blur + w₃·depth_hole + w₄·small_object + 
    w₅·crowd + w₆·rare_class
```

**Stage划分**：
- Stage A：只保持低难度样本本中的70%
- Stage B：吸收全分布并对rare class做方根重采样
- Stage C：保持全分布，hard sample权重上调

**Curriculum配置**：
- `easy_pool_ratio_stage_a: 0.70`
- `hard_boost_stage_c: 1.25`
- Loss更平滑，Stage B进入AP对比后rare class无下降

---

### 4. 神经结构优化

#### 4.1 决策理论模块：score calibration与top-100排序

**问题**：confidence排序与mAP"信息增益"不一致。

**解决方案**：
1. **Platt scaling与isotonic regression**：
   - 对每个类别拟合score校准器
   - 用3-fold开发集拟合，未见数据集应用
   - 代码位置：`src/engine/calibrate_scores.py`

2. **Top-100截断**：
   - raw_score -> classwise_calibrator -> global sort -> top-100
   - 避免"高参考低量"累积损失

#### 4.2 统计物理/系统工程模块：粗筛制衡稳定性

**Champion版本**：只接受已经过3-fold口径的版本
**Challenger版本**：每次只允许改一个变量
**Fallback版本**：永远保留一个最晚版本+与对应commit

**可复现性保障**：
- config yaml
- checkpoint hash
- git commit
- split manifest hash
- calibrator version
- submit zip hash
- Python/PyTorch/CUDA固定seed
- torch.manual_seed
- deterministic算法

---

## 数据、训练与工程实现

### 5. 数据准备

**Fork IDEA-Research/DINO**作为起点：
- 4-scale/5-scale配置
- 以COCO pretrain做基础
- 按official README微调op配置

**目录结构**：
```
project/
├── configs/
│   ├── rgc_dino_r50_final.yaml
│   ├── rgc_dino_r50_fold0.yaml
│   ├── rgc_dino_r50_fold1.yaml
│   └── rgc_dino_r50_fold2.yaml
├── data/
│   ├── train/
│   │   ├── rgb/
│   │   ├── ir/
│   │   ├── depth/
│   │   └── labels/
│   ├── test/
│   └── splits/
├── src/
│   ├── data/
│   │   ├── io_tri_modal.py
│   │   ├── tri_modal_dataset.py
│   │   ├── transforms_sync.py
│   │   ├── grouped_split.py
│   │   └── sampler_curriculum.py
│   ├── model/
│   │   ├── ir_side_encoder.py
│   │   ├── depth_side_encoder.py
│   │   ├── quality_features.py
│   │   ├── reliability_gated_fusion.py
│   │   ├── nuisance_domain_head.py
│   │   └── build_rgc_dino.py
│   ├── engine/
│   │   ├── train.py
│   │   ├── eval_local.py
│   │   ├── calibrate_scores.py
│   │   └── infer_submit.py
│   └── utils/
├── tools/
│   ├── inspect_dataset.py
│   ├── prepare_splits.py
│   ├── check_submission.py
│   ├── package_submission.py
│   ├── train_fold.sh
│   └── train_final.sh
├── tests/
└── outputs/
```

**数据IO设计**：
- `io_tri_modal.py`：读取RGB、IR、深度（16-bit）
- 使用cv2.IMREAD_UNCHANGED读取depth，禁止ANYDEPTH到8-bit
- `tri_modal_dataset.py`：输出sample字典{"image_id", "rgb", "ir", "depth3", "target", "quality_static", "group_id", "domain_id"}
- `transforms_sync.py`：确保RGB/IR/Depth/bbox同步变换

---

### 6. 训练环节

#### 6.1 环境配置

**基准环境**：
```bash
python=3.7.3, pytorch=1.9.0, cuda=11.1
```

**关键依赖**：
- 构造MultiScaleDeformable Attention需compatible DINO op
- 如无编译困难则保留DINO官方CUDA=11.1依赖
- PyTorch文档明示部分op无完全reproducibility，但可通过固定torch.manual_seed + torch.use_deterministic_algorithms()来抑制不确定性

#### 6.2 训练策略

**三阶段训练**：

| 阶段 | Epoch | 训练集尺度 | 冻结策略 | ModDrop | 域对抗 | 难度强化 | 目标 |
|------|-------|-----------|---------|---------|--------|---------|------|
| Stage A | 6 | 短边640 | 冻结RGB主干早期head | 关或较弱 | 关 | 低 | 让IR/Depth学会"不添乱" |
| Stage B | 28 | 640-800多尺度 | 全模型训练 | 开 | 开 | 中 | 吃满主效应 |
| Stage C | 8-12 | 896/960 | 全模型低LR | 开 | 维持低强度 | 强 | 冲AP75/AP90与榜距 |

**学习率调度**：
- AdamW，主干lr=1e-5, 其余1e-4, weight_decay=1e-4
- clip_max_norm=0.1
- Stage A引入backbone.0.body.conv1 / layer1冻结
- Stage B解冻全参，开启photometric aug
- Stage C降低lr到1e-5/1e-6

**主配置文件示例**（configs/rgc_dino_r50_final.yaml）：
```yaml
experiment:
  name: rgc_dino_r50_final
  seed: 3407
  output_dir: outputs/rgc_dino_r50_final

data:
  root: data/
  num_classes: 12
  split_mode: grouped_stratified_3fold
  folds: 3
  image_max_size: 1333
  keep_ratio: true
  train_scales_stage_a: [640]
  train_scales_stage_b: [640, 672, 704, 736, 768, 800]
  train_scales_stage_c: [896, 960]
  rgb_norm_mean: [0.485, 0.456, 0.406]
  rgb_norm_std: [0.229, 0.224, 0.225]
  depth_clip_mm: [300, 20000]
  ir_collapse: channel0
  preload_quality_cache: true
  grouped_sampler: true
  rare_class_repeat_sqrt: true
  sync_aug:
    hflip_prob: 0.5
    random_resize: true
    random_crop_prob: 0.2
    max_crop_ratio: 0.15
  rgb_aug:
    color_jitter_prob: 0.5
    blur_prob: 0.2
  ir_aug:
    gain_shift_prob: 0.2
    blur_prob: 0.1
  depth_aug:
    hole_sim_prob: 0.15
    noise_prob: 0.1

model:
  family: dino_4scale_r50
  pretrain_model_path: weights/checkpoint0011_4scale.pth
  finetune_ignore: ["label_enc.weight", "class_embed"]
  num_queries: 900
  num_select: 300
  num_feature_levels: 4
  hidden_dim: 256
  enc_layers: 6
  dec_layers: 6
  nheads: 8
  dn_number: 100
  dn_labelbook_size: 64
  dn_box_noise_scale: 0.4
  dn_label_noise_ratio: 0.5
  use_checkpoint_stage_c: true
  use_ema: true
  ema_decay: 0.9997

fusion:
  enable: true
  mode: reliability_gated_residual
  q_dim: 24
  alpha_prior: 0.35
  gate_floor: 0.05
  gate_ceil: 0.90
  gate_loss_weight: 0.04

domain_head:
  enable: true
  num_domains: 4
  lambda_max: 0.05
  warmup_epochs: 6
  cluster_source: quality_features_kmeans

optimizer:
  type: AdamW
  lr: 1.0e-4
  lr_backbone: 1.0e-5
  weight_decay: 1.0e-4
  clip_max_norm: 0.1
  batch_size_per_gpu: 1
  grad_accum_steps: 8
  amp: true
  zero_grad_set_to_none: true

schedule:
  total_epochs: 42
  stage_a_epochs: 6
  stage_b_epochs: 28
  stage_c_epochs: 8
  lr_drop_epochs: [30, 38]

augmentation:
  disable_mosaic: true
  disable_cutmix: true
  keep_modal_alignment: true

curriculum:
  enable: true
  easy_pool_ratio_stage_a: 0.70
  hard_boost_stage_c: 1.25
  difficulty_weights:
    low_light: 1.0
    blur: 0.7
    depth_hole: 0.8
    small_object: 1.0
    crowd: 0.6
    rare_class: 1.0

moddrop:
  enable: true
  mode: soft
  stage_b:
    rgb: 0.05
    ir: 0.12
    depth: 0.15
  stage_c:
    rgb: 0.08
    ir: 0.15
    depth: 0.20
  min_keep_scale:
    rgb: 0.70
    ir: 0.50
    depth: 0.45

loss:
  lambda_gate: 0.04
  lambda_domain: 0.00
  lambda_cross_modal_optional: 0.00
  enable_cross_modal_optional: false

evaluation:
  metric: competition_map_50_95
  iou_thresholds: [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
  interpolation_points: 101
  report_ap50: true
  report_ap75: true
  report_per_class: true
  report_bucket_ap: true

inference:
  nms_iou_threshold: -1
  topk_per_image: 100
  calibrator: oof_classwise_hybrid
  score_floor: 1.0e-4

submission:
  write_empty_txt: true
  enforce_class_id_range: [0, 11]
  normalize_boxes: true
  zip_name: rgc_dino_r50_final_submit.zip

reproducibility:
  save_seed_manifest: true
  save_git_commit: true
  save_git_diff: true
  save_split_manifest: true
  deterministic_final: true
```

**命令行示例**：
```bash
# 准备splits
python tools/prepare_splits.py --config configs/rgc_dino_r50_final.yaml

# 训练fold0
bash tools/train_fold.sh configs/rgc_dino_r50_final.yaml fold0

# 训练fold1
bash tools/train_fold.sh configs/rgc_dino_r50_final.yaml fold1

# 训练fold2
bash tools/train_fold.sh configs/rgc_dino_r50_final.yaml fold2

# 校准scores
python -m src.engine.calibrate_scores --config configs/rgc_dino_r50_final.yaml

# 最终推理+提交
bash tools/train_final.sh configs/rgc_dino_r50_final.yaml
```

---

### 7. 验证、提交与作战计划

#### 7.1 推理与提交链路

**流程**：
1. infer_submit.py：预测到原生口径
2. calibrate：xxyy_abs -> cxcywh_norm -> 应用classwise calibrator -> global sort -> top-100
3. 截断到100 -> 同名TXT
4. writer共用一套parser
5. check_submission.py实战骨架

**提交格式验证**（tools/check_submission.py）：
```python
from __future__ import annotations
from pathlib import Path

NUM_CLASSES = 12

def validate_one_file(path: Path) -> list[str]:
    errs = []
    lines = [x.strip() for x in path.read_text(encoding="utf-8").splitlines()]
    if x.strip()]:
        
        if len(lines) > 100:
            errs.append(f"{path.name}: more than 100 predictions")
        
        for i, line in enumerate(lines):
            parts = line.split()
            if len(parts) != 6:
                errs.append(f"{path.name}:{i+1} expected 6 fields, got {len(parts)}")
                continue
            
            cls_id, cx, cy, w, h, score = parts
            try:
                cls_id = int(cls_id)
                cx, w, h, score = map(float, (cx, cy, w, h, score))
            except ValueError:
                errs.append(f"{path.name}:{i+1} parse error")
                continue
            
            if not (0 <= cls_id < NUM_CLASSES):
                errs.append(f"{path.name}:{i+1} invalid class_id {cls_id}")
            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
                errs.append(f"{path.name}:{i+1} center out of range")
            if not (0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                errs.append(f"{path.name}:{i+1} box size out of range")
            if not (0.0 <= score <= 1.0):
                errs.append(f"{path.name}:{i+1} score out of range")
    
    return errs

def validate_dir(pred_dir: str, expected_ids: list[str]) -> None:
    pred_root = Path(pred_dir)
    all_errs = []
    
    for image_id in expected_ids:
        path = pred_root / f"{image_id}.txt"
        if not path.exists():
            all_errs.append(f"missing file: {path.name}")
            continue
        all_errs.extend(validate_one_file(path))
    
    extra = {p.stem for p in pred_root.glob("*.txt")} - set(expected_ids)
    all_errs.extend([f"unexpected file: {x}.txt" for x in sorted(extra)])
    
    if all_errs:
        raise SystemExit("\n".join(all_errs))
    
    print("submission format check passed")
```

#### 7.2 本地评估器

**实现位置**：`src/engine/eval_local.py`

**核心逻辑**：现实COCO JSON后多不看有备份，必须做原生TXT合法性、空文件治理、每图top-100截断后的结果排序、类别约束[0,11]、101点插值压缩等。

```python
def evaluate_dir(
    pred_dir: str,
    gt_dir: str,
    image_ids: list[str],
    num_classes: int = 12,
    iou_thresholds: list[float] = [0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95],
    interpolation_points: int = 101,
) -> dict:
    """
    返回:
      {
        "map_50_95": float,
        "ap50": float,
        "ap75": float,
        "per_class_ap": {...},
        "per_iou_map": {...},
      }
    """
```

**实况考试必需项**：每次必前后交织的101点插值压缩标准，以及对每个bucket AP分点校准；回滚策略应是off / classwise Platt only / classwise isotonic only / hybrid，失败风险仅hybrid混接。

#### 7.3 实验顺序跟踪表

| 代号 | 改动 | 预期收益 | 成功标准 | 失败表现 | 回滚策略 |
|-----|-----|---------|---------|---------|---------|
| A0 | RGB-only DINO 4-scale R50 | 建基线 | 本地评测与官方口径一致 | loss/evaluator异常 | 不进入A1以前先修链路 |

---

## 单元测试与作战计划

### 8. 单元测试文件

**Day 1即上线的测试文件**：

| 测试文件 | 输入 | 预期输出 | 失败含义 |
|---------|-----|---------|---------|
| test_depth_encoding.py | 合成16-bit depth | log/inv/mask数值正确 | 深度物理链路错误 |
| test_ir_collapse.py | 三通道几乎相同PNG | 输出单通道一致 | IR读取链路 |
| test_bbox_conversion.py | cxcywh_norm / xxyy_abs | 可逆误差在阈内 | 提交框坐标标有风险 |
| test_sync_transform.py | 三模态对对齐样本+框 | 几何变换后仍对齐 | 同步增强有bug |
| test_grouped_split.py | 带group_id样本集 | 无group泄漏 | 验证集污染 |
| test_fusion_shapes.py | 三路feature list | fused shape对齐 | 融合模块接线错误 |
| test_gate_distribution.py | 随机batch | gate和为1，受floor/ceil约束 | 门控实现不稳 |
| test_eval_metric.py | 小型手工样例 | AP与人工算一致 | 本地评测器链错 |
| test_submission_format.py | 伪提交目录 | 非法格式本被拒 | 最终提交格式式风险 |
| test_train_smoke.py | 8张样本+小循环 | 5-10 iter无异常 | 主训练链本未闭环 |

---

## 复现性保障

**完整记录系统**：
- 保存seed manifest
- 保存git commit
- 保存git diff
- 保存split manifest
- deterministic_final模式

**确定性工程**：
- torch.manual_seed
- Python/NumPy种子
- CUDA deterministic algorithms
- 禁用Mosaic/CutMix（破坏模态对齐）

---

## 总结

本工程方案实现了：
1. **三模态协同**：RGB主干 + IR/Depth side encoders
2. **质量感知融合**：Reliability-Gated Residual Fusion
3. **跨学科优化**：信息论、信号处理、控制论、认知科学
4. **工程保障**：单元测试、可复现性、完整验证链路

**预期效果**：
- Stage A: 建立基线，确保IR/Depth不添乱
- Stage B: 充分利用三模态协同，提升主效应
- Stage C: 冲击AP75/AP90高IoU指标，缩小与榜首差距

**关键创新点**：
- 质量先验引导的自适应门控机制
- 域对抗训练增强泛化能力
- 课程学习策略优化训练效率
- 完整的工程化验证与提交流程
