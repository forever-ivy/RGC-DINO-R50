# RGC-DINO 项目文档导航

> 面向城市场景视觉多模态目标检测竞赛  
> 当前状态：45.044分 → 目标60+分（2个月）

---

## 🎯 快速开始

**当前最佳方案**：[FINAL_ROADMAP.md](FINAL_ROADMAP.md)
- 硬件：2×RTX 3090
- 时间：2个月（10周）
- 目标：60±2分
- 状态：✓ 完全可行

---

## 📋 核心文档

### 1. 最终执行方案
- **[FINAL_ROADMAP.md](FINAL_ROADMAP.md)** - 2×RTX 3090可行的完整路线图
  - 7个阶段，从45分到62分
  - 显存优化策略（DeepSpeed, Gradient Checkpointing）
  - 详细的时间和资源估算

---

## 📚 归档文档

### 失败教训（Lessons Learned）
- [TTA失败分析](archive/lessons_learned/TTA_FAILURE_ANALYSIS.md)
  - 朴素平均导致-10分
  - 必须用NMS/WBF
  
- [2-fold融合失败](archive/lessons_learned/2FOLD_FAILURE_LESSONS.md)
  - 弱模型拉低强模型
  - 验证集≠测试集
  - 单强模型 > 多弱模型融合

### 历史方案（Historical Plans）
- [原始工程方案](archive/historical_plans/RGC_DINO_ENGINEERING_PLAN.md)
  - 从PDF转换的完整技术方案
  - RGC三模态融合机制
  
- [学术前沿方案](archive/historical_plans/SOTA_BREAKTHROUGH_ROADMAP.md)
  - 理想硬件配置（8×A100）
  - InternImage-XL等大模型方案

### 自动化文档（Automation）
- [自动化总结](archive/automation/AUTOMATION_SUMMARY.md)
- [竞赛自动化](archive/automation/COMPETITION_AUTOMATION.md)
- [快速开始](archive/automation/QUICK_START_AUTOMATION.md)

---

## 🗺️ 项目路线

### 当前进度（2026-06-15）

| 阶段 | 状态 | 分数 | 备注 |
|------|------|------|------|
| Baseline (ResNet-50) | ✓ | 45.044 | fold0单模型 |
| 低阈值调优 | ✗ | 43.955 | 失败 |
| TTA多尺度 | ✗ | 34.872 | 灾难性失败 |
| 2-fold融合 | ✗ | 44.263 | 不升反降 |
| **→ Swin-L升级** | 🔄 | 目标50 | **下一步** |

### 未来计划（10周）

```
Week 1-2: Swin-L训练 → 49-50分
Week 3-4: 数据增强+长训练 → 52-53分
Week 5-6: 知识蒸馏 → 54-55分
Week 7-8: 半监督学习 → 56-58分
Week 9-10: 多模型融合+优化 → 60-62分
```

---

## 🔑 关键教训

### ✓ 成功经验
1. RGC三模态融合机制有效
2. 质量感知门控工作正常
3. ResNet-50达到45分符合预期

### ✗ 失败教训
1. **融合不一定提升** - 弱模型会拉低强模型
2. **验证集≠测试集** - 必须在测试集实测
3. **Backbone是瓶颈** - ResNet-50已到上限47分

### 💡 核心认知
- 单个强模型 > 多个弱模型融合
- 架构升级 > 微调优化
- 实测 > 理论预期

---

## 📊 当前状态

### 模型性能

| Fold | 验证集mAP | 测试集分数 | 训练Epoch | 状态 |
|------|----------|-----------|----------|------|
| fold0 | 0.3279 | 45.044 | 12 | ✓ 最强 |
| fold1 | 0.2446 | 未知 | 12 | ⚠️ 弱（划分异常）|
| fold2 | 0.2872 | <44推测 | 12 | ⚠️ 次强 |

### 硬件资源
- **GPU**: 2×RTX 3090 (24GB×2)
- **算力**: ~71 TFLOPS (FP16) per card
- **限制**: 单卡24GB显存

---

## 🚀 立即行动

### 本周任务
1. ✓ 整理文档结构
2. 准备Swin-L配置文件
3. 下载Swin-L预训练权重
4. 启动fold0训练

### 命令快速参考
```bash
# 下载权重
wget https://github.com/microsoft/Swin-Transformer/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth

# 启动训练（2卡）
python -m torch.distributed.launch \
  --nproc_per_node=2 \
  scripts/train_rgc_dino.py \
  --config configs/swin_l_stage1.yaml \
  --fold 0
```

---

## 📞 快速链接

- 主仓库：`/data1/liuxuan/projects/RGC-DINO-R50`
- 配置文件：`configs/`
- 训练脚本：`scripts/train_rgc_dino.py`
- 输出目录：`outputs/`
- 最终方案：[FINAL_ROADMAP.md](FINAL_ROADMAP.md)

---

## 📝 文档更新日志

- 2026-06-15 23:30: 重组文档结构，创建归档目录
- 2026-06-15 23:00: 2-fold融合失败（44.263分）
- 2026-06-15 22:16: 提交2-fold融合
- 2026-06-15 21:00: TTA失败（34.872分）
- 2026-06-15 13:00: Baseline建立（45.044分）

---

**最后更新**：2026-06-15 23:30  
**下一步**：启动Swin-L训练，目标2周内达到50分
