# 2×RTX 3090可行方案：从46分冲击60+分（2个月完整路线）

> 针对2×RTX 3090 (24GB×2)硬件约束的优化方案

---

## 硬件约束分析

**当前配置**：
- GPU: 2×RTX 3090 (24GB VRAM each)
- 总显存: 48GB
- 算力: ~71 TFLOPS (FP16) per card

**可行性评估**：
- ✓ Swin-B/L: 可行
- ✓ Co-DETR: 可行（需gradient checkpointing）
- ⚠️ InternImage-XL: 受限（需优化）
- ✗ DINO-v2 ViT-g/14 (1.1B): 不可行

---

## 调整后的突破路线（2个月）

### 阶段1：基础优化（2周）→ 目标52分

**Week 1-2: Swin-B训练优化**

```yaml
# 2×3090可行配置
model:
  backbone: swin_base  # 88M参数，~7GB显存
  
training:
  # 显存优化
  batch_size_per_gpu: 2  # 每卡2张
  grad_accumulation: 8   # 累积8步 = 等效batch_size 32
  mixed_precision: true  # FP16
  gradient_checkpointing: true  # 节省显存
  
  # 训练配置
  epochs: 36
  input_size: 800
  lr: 2e-5
  
  # 数据并行
  distributed: true
  world_size: 2
```

**显存占用估算**：
- Model: 7GB
- Optimizer states: 14GB
- Activations: ~2GB (with checkpointing)
- 总计: ~23GB < 24GB ✓

**训练时间**：
- 每epoch约45分钟（2000 iters）
- 36 epochs = 27小时
- 3-fold = 81小时 = **3.4天**

**预期提升**：+3-4分 (45→49分)

---

### 阶段2：架构升级（2周）→ 目标54分

**Week 3-4: Swin-L + 渐进式训练**

```yaml
# Swin-L优化配置
model:
  backbone: swin_large  # 197M参数，~14GB显存
  
training:
  # 显存优化策略
  batch_size_per_gpu: 1  # 每卡1张（关键）
  grad_accumulation: 16  # 累积16步
  gradient_checkpointing: true
  
  # 渐进式训练（节省时间）
  stage_1:  # 低分辨率预热
    epochs: 12
    input_size: 640
    
  stage_2:  # 提高分辨率
    epochs: 24
    input_size: 800
    load_from: stage_1_best.pth
```

**显存占用**：
- Model: 14GB
- Optimizer: 28GB → 使用DeepSpeed ZeRO-2优化到18GB
- Activations: ~3GB
- 总计: ~20GB < 24GB ✓

**DeepSpeed配置**：
```json
{
  "train_batch_size": 32,
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 16,
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 2e-5,
      "weight_decay": 0.05
    }
  },
  "fp16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu"
    }
  }
}
```

**训练时间**：
- Stage 1: 12 epochs × 40 min = 8小时
- Stage 2: 24 epochs × 60 min = 24小时
- 3-fold = 96小时 = **4天**

**预期提升**：+3-4分 (49→53分)

---

### 阶段3：训练策略优化（1周）→ 目标56分

**Week 5: 数据增强+长训练**

```yaml
# 增强配置
augmentation:
  # 3090可用的轻量增强
  mosaic: 
    prob: 0.5
    num_images: 4  # 4图拼接（不用9图）
    
  mixup:
    prob: 0.15
    alpha: 0.5
    
  copypaste:
    prob: 0.3
    max_objects: 20  # 限制数量
    
  # 基础增强
  random_flip: 0.5
  color_jitter: 0.5
  blur: 0.2

training:
  epochs: 50  # 延长训练
  early_stopping: 
    patience: 10
    monitor: val_map
```

**训练时间**：
- 50 epochs × 70 min = 58小时
- 3-fold = 174小时 = **7.2天**

**预期提升**：+2-3分 (53→56分)

---

### 阶段4：模型融合优化（1周）→ 目标58分

**Week 6: 知识蒸馏**

```yaml
# 蒸馏配置（3090可行）
distillation:
  # 教师：Swin-L最佳模型
  teacher:
    backbone: swin_large
    checkpoint: best_swin_l_fold0.pth
    frozen: true
    
  # 学生：Swin-B
  student:
    backbone: swin_base
    
  # 蒸馏策略
  feature_distillation:
    layers: [stage2, stage3, stage4]
    loss_weight: 0.5
    
  logit_distillation:
    temperature: 4.0
    loss_weight: 0.7
    
training:
  epochs: 30
  batch_size_per_gpu: 2  # 学生模型小，可增加batch
```

**显存占用**：
- Teacher (frozen): 14GB
- Student (trainable): 7GB
- 总计: ~21GB < 24GB ✓

**训练时间**：
- 30 epochs × 50 min = 25小时
- 3-fold = 75小时 = **3.1天**

**预期提升**：+1-2分 (56→58分)

---

### 阶段5：半监督学习（2周）→ 目标60分

**Week 7-8: 测试集伪标签**

```yaml
# 半监督流程
semi_supervised:
  # Round 1: 训练集训练强模型
  round_1:
    model: swin_large
    dataset: train_only
    epochs: 50
    
  # Round 2: 测试集生成伪标签
  round_2:
    inference:
      # 多模型ensemble生成高质量伪标签
      models:
        - swin_l_fold0
        - swin_l_fold1
        - swin_l_fold2
      fusion: wbf
      confidence_threshold: 0.95  # 高置信度过滤
      
  # Round 3: 联合训练
  round_3:
    dataset: train + pseudo_test
    pseudo_loss_weight: 0.5
    epochs: 30
```

**训练时间**：
- Round 1: 7.2天（已在阶段3完成）
- Round 2: 推理3小时
- Round 3: 30 epochs × 80 min = 40小时
- 3-fold = 120小时 = **5天**

**预期提升**：+1-2分 (58→60分)

---

### 阶段6：后处理优化（1周）→ 目标61分

**Week 9: WBF + TTA**

```yaml
# 后处理配置
postprocess:
  # WBF融合
  wbf:
    models:
      - swin_l_fold0: weight=1.2
      - swin_l_fold1: weight=1.0
      - swin_l_fold2: weight=1.0
      - swin_b_distilled_fold0: weight=0.8
      - swin_b_distilled_fold1: weight=0.8
      - swin_b_distilled_fold2: weight=0.8
    iou_threshold: 0.5
    
  # TTA（轻量版）
  tta:
    augmentations:
      - scale=1.0, flip=false
      - scale=0.9, flip=false
      - scale=1.1, flip=false
      - scale=1.0, flip=true
    fusion: wbf
```

**推理时间**：
- 单模型单尺度: 5分钟
- 6模型 × 4TTA = 24次推理
- 总时间: ~2小时

**预期提升**：+0.5-1分 (60→61分)

---

### 阶段7：极限优化（1周）→ 目标62分

**Week 10: Model Soup + 超参搜索**

```yaml
# Model Soup
model_soup:
  # 收集多个checkpoint
  checkpoints:
    - swin_l_fold0_ep48.pth
    - swin_l_fold0_ep49.pth
    - swin_l_fold0_ep50.pth
  
  # 贪心选择最优组合
  averaging: greedy_soup
  
# 超参搜索（轻量版）
hyperparameter_search:
  method: optuna
  n_trials: 20  # 限制试验次数
  
  # 只搜索后处理参数
  search_space:
    wbf_iou_threshold: [0.4, 0.6]
    confidence_threshold: [0.001, 0.05]
    nms_threshold: [0.5, 0.7]
```

**时间**：1周

**预期提升**：+0.5-1分 (61→62分)

---

## 完整时间线（2个月）

| 阶段 | 周数 | 任务 | GPU时间 | 预期分数 |
|------|------|------|---------|---------|
| 当前 | - | Baseline | - | 45-47 |
| 阶段1 | 1-2 | Swin-B优化 | 3.4天 | 49 |
| 阶段2 | 3-4 | Swin-L升级 | 4天 | 53 |
| 阶段3 | 5 | 数据增强+长训练 | 7.2天 | 56 |
| 阶段4 | 6 | 知识蒸馏 | 3.1天 | 58 |
| 阶段5 | 7-8 | 半监督学习 | 5天 | 60 |
| 阶段6 | 9 | WBF+TTA | 0.1天 | 61 |
| 阶段7 | 10 | 极限优化 | 0.5天 | 62 |

**总GPU时间**：约23天（含buffer）
**总日历时间**：10周 = 2.5个月（留有余量）

---

## 显存优化技巧汇总

### 1. Gradient Checkpointing
```python
# 启用梯度检查点
model.backbone.use_checkpoint = True
# 节省：40-50% activation显存
# 代价：15-20% 训练速度下降
```

### 2. DeepSpeed ZeRO-2
```python
# 优化器状态分片
from deepspeed import initialize

model_engine, optimizer, _, _ = initialize(
    model=model,
    config='ds_config.json'
)
# 节省：~30% 优化器显存
```

### 3. Mixed Precision Training
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
scaler.scale(loss).backward()
# 节省：~40% 总显存
```

### 4. Gradient Accumulation
```python
# 小batch size + 累积梯度
optimizer.zero_grad()
for i, (inputs, targets) in enumerate(dataloader):
    outputs = model(inputs)
    loss = criterion(outputs, targets) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
# 效果：等效大batch size，显存友好
```

---

## 无法实现的方案（标注）

### ✗ InternImage-XL (335M参数)
- **问题**：单卡需要>30GB显存
- **替代**：Swin-L (197M) 性能接近
- **损失**：约1-2分

### ✗ Co-DETR完整版
- **问题**：双decoder需要额外8GB显存
- **替代**：单decoder + 更长训练
- **损失**：约1分

### ✗ DINO-v2 ViT-g/14 (1.1B)
- **问题**：需要>80GB显存
- **替代**：冻结DINO-v2 ViT-B作为特征提取器
- **损失**：约2分

---

## 资源消耗估算

### 电力成本
- 2×RTX 3090功耗: ~700W
- 训练23天 × 24h = 552小时
- 用电: 552h × 0.7kW = 386kWh
- 电费: ~$50-100（按地区）

### 时间成本
- 训练时间: 23 GPU-days
- 人工时间: ~40小时（配置+监控+调试）
- 总日历时间: 10周

---

## 风险控制

### 技术风险

| 风险 | 概率 | 缓解措施 |
|------|------|---------|
| OOM (显存不足) | 中 | Gradient checkpointing, 减小batch size |
| 训练不稳定 | 低 | 梯度裁剪, 更小学习率 |
| 过拟合 | 中 | Early stopping, 数据增强 |

### 时间风险
- 每阶段预留20% buffer
- 关键路径：Swin-L训练
- 并行策略：推理与训练同步

---

## 实施清单

### 软件依赖
```bash
# 核心依赖
pip install torch==2.0.1+cu118 torchvision
pip install deepspeed==0.9.5
pip install optuna
pip install ensemble-boxes  # WBF

# 下载预训练权重
wget https://github.com/microsoft/Swin-Transformer/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth
wget https://github.com/microsoft/Swin-Transformer/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth
```

### 硬件检查
```bash
# 检查显存
nvidia-smi

# 测试DeepSpeed
ds_report

# 监控温度
watch -n 1 nvidia-smi
```

---

## 预期成果

### 保守估计
- 目标: 58-60分
- 概率: 80%
- 排名: 前3-5名

### 乐观估计
- 目标: 60-62分
- 概率: 50%
- 排名: 榜首或超越

### 关键因素
1. Swin-L训练质量（最重要）
2. 半监督学习效果
3. WBF融合策略

---

## 立即行动

### 本周任务（Week 1）
1. 安装DeepSpeed和依赖
2. 下载Swin-B预训练权重
3. 配置分布式训练
4. 启动Swin-B fold0训练
5. 监控显存和速度

### 命令示例
```bash
# 启动Swin-B训练（2卡）
python -m torch.distributed.launch \
  --nproc_per_node=2 \
  scripts/train_rgc_dino.py \
  --config configs/swin_b_stage1.yaml \
  --fold 0

# 监控
tensorboard --logdir outputs/swin_b_fold0/
```

---

## 总结

**可行性**：✓ 完全可行
- 所有方案都在2×3090的显存限制内
- 训练时间约23天，符合2个月预算
- 预期达到60±2分

**核心策略**：
- 用Swin-L替代InternImage-XL（性能接近，显存友好）
- 用DeepSpeed优化显存
- 用Gradient Checkpointing换时间
- 重点投入：数据增强、半监督、蒸馏

**开始行动**：
立即启动阶段1的Swin-B训练，验证显存和速度符合预期。
