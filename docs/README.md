# RGC-DINO 项目文档导航

> 面向城市场景视觉多模态目标检测竞赛  
> 当前状态：线上最佳已从 RGC-DINO-R50 baseline 45.044 提升到 **Co-DETR + InternImage-L continue epoch20 的 48.335**；后续主冲榜路线明确切换为 **Co-DETR InternImage-L continuation**，RGC 三模态迁移作为下一阶段提分方向。

---

## 🎯 当前最佳方案

**主线文档**：[FINAL_ROADMAP.md](FINAL_ROADMAP.md)

- 硬件：2×RTX 3090（24GB×2）
- 当前线上最佳：Co-DETR + InternImage-L continue epoch20，**48.335**
- 当前主线：先坚持已验证的 RGB-only Co-DETR InternImage-L continuation/checkpoint-selection 路线
- 下一阶段提分：在 Co-DETR InternImage-L 主线稳定后，再迁移 IR/Depth reliability-gated residual fusion
- 原则：强单模型优先，strict final-TXT validation 过门禁后才提交

**硬件可行性说明**：[2x3090_FEASIBLE_ROADMAP.md](2x3090_FEASIBLE_ROADMAP.md)

- 说明 Co-DETR + InternImage-L 在 2×3090 上的显存压力
- 给出 batch=1/GPU、AMP、gradient checkpointing、梯度累积、分辨率阶梯等策略
- 明确 Co-DETR-R50 sanity → Co-DETR + InternImage-L 低分辨率 → 主训练 → train-all 的执行顺序

---

## 📋 核心文档

### 1. 最终执行方案

- **[FINAL_ROADMAP.md](FINAL_ROADMAP.md)**
  - Co-DETR + InternImage-L 主冲榜路线
  - 阶段化训练计划
  - 验证门禁与提交纪律
  - 禁止路线：测试集伪标签训练、朴素 TTA 平均、弱 fold 融合、简单投票/平均 ensemble

### 2. 2×3090 可行性

- **[2x3090_FEASIBLE_ROADMAP.md](2x3090_FEASIBLE_ROADMAP.md)**
  - 显存风险排序
  - 省显存训练策略
  - Co-DETR + InternImage-L 的执行阶梯

---

## 📚 归档文档

### 失败教训（Lessons Learned）

- [TTA失败分析](archive/lessons_learned/TTA_FAILURE_ANALYSIS.md)
  - 朴素平均导致约 10 分级别掉分
  - TTA 必须使用验证过的 NMS/Soft-NMS/WBF-like 合并策略

- [2-fold融合失败](archive/lessons_learned/2FOLD_FAILURE_LESSONS.md)
  - 弱模型会拉低强模型
  - 验证集和测试集不完全一致
  - 单个强模型优先于多个弱模型融合

### 历史方案（Historical Plans）

- [原始工程方案](archive/historical_plans/RGC_DINO_ENGINEERING_PLAN.md)
  - 从 PDF 转换的 RGC-DINO 技术方案
  - 三模态可靠性门控融合设计

- [学术前沿方案](archive/historical_plans/SOTA_BREAKTHROUGH_ROADMAP.md)
  - 理想硬件下的更激进方案
  - 当前 2×3090 下仅作参考

### 自动化文档（Automation）

- [自动化总结](archive/automation/AUTOMATION_SUMMARY.md)
- [竞赛自动化](archive/automation/COMPETITION_AUTOMATION.md)
- [快速开始](archive/automation/QUICK_START_AUTOMATION.md)

---

## 🗺️ 当前进度

| 阶段 | 状态 | 分数/目标 | 备注 |
|---|---|---:|---|
| RGC-DINO-R50 baseline | 已完成/退为 fallback | 45.044 | 老基线，已被新主线超过 |
| 低阈值调优 | 失败 | 43.955 | 低阈值/密集框高风险 |
| 朴素 TTA 平均 | 失败 | 34.872 | 已归档为禁止路线 |
| 2-fold 融合 | 失败 | 44.263 | 弱 fold 拖累强模型 |
| Swin-L 尝试 | 退为对照/复盘 | 不作为上限判断 | 疑似权重/评估闭环问题 |
| **Co-DETR + InternImage-L continue epoch20** | **当前主线/线上最佳** | **48.335** | strict final-TXT mAP 0.413322，后续只提交超过此门槛的候选 |

---

## 🔑 核心认知

### 已验证经验

1. Co-DETR + InternImage-L continuation 已在线上验证超过老 RGC-DINO baseline：48.335 vs 45.044，正式作为当前主线。
2. 当前已上线高分仍是 RGB-only Co-DETR；IR/Depth/RGC 三模态融合是下一阶段提分方向，不再先退回旧 RGC-DINO 主线。
3. 错误融合比不融合更糟；不能把多模型/多尺度输出简单平均。

### 当前主线原则

```text
强单模型优先
→ Co-DETR + InternImage-L continuation 作为当前主线
→ checkpoint/strict final-TXT sweep 选择 best epoch（当前 epoch20）
→ 只提交 local strict mAP 超过 0.413322、线上基线超过 48.335 的候选
→ 再推进更长训练 / train-all / 单模型 TTA或tile / IR-Depth RGC fusion 迁移
→ 严格 promotion 后提交
```

---

## 🚀 立即行动

### Week 0：地基修复

1. 补齐 checkpoint → validation prediction → mAP 的完整评估闭环。
2. 加入 backbone/detector 预训练权重加载硬门禁。
3. 记录 prediction count / score distribution，防止极端稀疏或极端密集输出。
4. 保证所有候选 ZIP 都有 manifest 和 promotion reason。

### Week 1：Co-DETR-R50 sanity

1. 离线准备 Co-DETR 集成目录。
2. 写最小 Co-DETR-R50 配置。
3. 写 LSF smoke 脚本，只生成脚本，不在交互会话直接跑长任务。
4. 跑 1-3 epoch sanity，确认 loss、mAP、TXT 输出链路。

### Week 2+：Co-DETR + InternImage-L

1. 准备 InternImage-L 公开预训练权重。
2. 写 backbone wrapper 和 key-loading report。
3. 先 640/704 低分辨率训练 12-18 epoch。
4. 验证通过后启动 50-72 epoch 主训练。

---

## 📞 快速链接

- 主仓库：`/data1/liuxuan/projects/RGC-DINO-R50`
- 配置目录：`configs/`
- 训练脚本：`scripts/train_rgc_dino.py`（现有 RGC-DINO 主线；Co-DETR 主线需新增脚本/适配）
- 推理脚本：`scripts/infer_rgc_dino.py`（Co-DETR 主线需新增或适配）
- 输出目录：`outputs/`
- 主线方案：[FINAL_ROADMAP.md](FINAL_ROADMAP.md)
- 2×3090 可行性：[2x3090_FEASIBLE_ROADMAP.md](2x3090_FEASIBLE_ROADMAP.md)

---

## 📝 文档更新日志

- 2026-06-18：主路线调整为 Co-DETR + InternImage-L；删除测试集伪标签训练、多模型简单融合等不合规/高风险默认路线。
- 2026-06-15：重组文档结构，创建归档目录。
- 2026-06-15：2-fold 融合失败（44.263）。
- 2026-06-15：TTA 朴素平均失败（34.872）。
- 2026-06-15：R50 baseline 建立（45.044）。
