# RGC-DINO 项目文档导航

> 面向城市场景视觉多模态目标检测竞赛  
> 当前状态：线上最佳已从 RGC-DINO-R50 baseline 45.044、Co-DETR + InternImage-L continue epoch20 的 48.335、fresh epoch7 raw 的 48.6960，提升到 **Co-DETR + InternImage-L fresh epoch7 + class thresholds 的 48.727**。2026-06-22 train-time high-res fine-tune 已完成但 strict mAP 退步，未进入 test ZIP / promotion / 提交；后续主冲榜路线继续以 48.727 class-threshold anchor 为门槛，保留 **class-wise threshold / hard validation / prediction diagnostics** 作为候选筛选体系。

---

## 🎯 当前最佳方案

**主线文档**：[FINAL_ROADMAP.md](FINAL_ROADMAP.md)

- 硬件：2×RTX 3090（24GB×2）
- 当前线上最佳：Co-DETR + InternImage-L fresh epoch7 + class thresholds，**48.727**（strict final-TXT mAP `0.4262677082771047`，hard-val `0.28694971837472955`）。
- 当前主线：坚持已验证的 Co-DETR InternImage-L continuation/checkpoint-selection 路线，不改道到 YOLO。
- 最新状态：2026-06-22 high-res fine-tune strict best `0.4188436556179077`，低于 fresh epoch7 anchor；hard-val `0.29068978363886794` 小幅高于 baseline，但未过 strict promotion 门槛。
- 立即插入：吸收对手 50.8190 经验，先做 class-wise threshold、NMS/image-side sweet spot、hard validation、prediction diagnostics。
- 下一阶段提分：在 Co-DETR InternImage-L 主线和后处理门禁稳定后，再迁移 IR/Depth reliability-gated residual fusion。
- 原则：强单模型优先，strict final-TXT validation + hard-val/预测分布过门禁后才提交。

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

### 3. 对手经验复盘

- **[OPPONENT_URBAN_VISUAL_RECOGNITION_LESSONS.md](OPPONENT_URBAN_VISUAL_RECOGNITION_LESSONS.md)**
  - 复盘 `YOLO11M + RGB-guided-RDT + class-wise threshold` 50.8190
  - 提取可迁移经验：FP suppression、类别阈值、高分辨率 sweet spot、hard validation、RGB 主导三模态引导
  - 明确不能使用对手权重/提交包，不能主线改道 YOLO

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
| Co-DETR + InternImage-L continue epoch20 | 已完成/被 fresh epoch7 超过 | 48.335 | strict final-TXT mAP 0.413322，旧 Co-DETR anchor |
| Co-DETR + InternImage-L fresh epoch7 | 已完成/被 class-threshold 版本超过 | 48.6960 | strict final-TXT mAP 0.426216676，hard-val 0.286884750，旧 raw fresh anchor |
| **Co-DETR + InternImage-L fresh epoch7 + class thresholds** | **当前主线/线上最佳** | **48.727** | strict final-TXT mAP 0.426267708，hard-val 0.286949718；class thresholds `[0.05,0.02,0.003,0,...]` |
| Co-DETR fresh epoch7 corrected image-side s832 smoke | 已完成/诊断，不提交 | 不提交 | 旧 s768/s832/s896 因 1333 width cap 实际都 resize 到 1333×750；修正 s832 `(1479,832)` 后 results hash 已变化，但 strict 0.422748 < anchor 0.426267708，hard-val 0.293758 小幅更好 |
| Co-DETR fresh epoch7 high-res fine-tune | 已完成/不 promotion | 不提交 | best strict 0.418843656 < 0.426216676；hard-val 0.290689784 小幅更好但主指标退步 |
| 对手 urban-visual-recognition | 外部公开参考/不作为本项目产物 | 50.8190 | YOLO11M + RGB-guided-RDT + class-wise threshold；只学习经验，不使用其权重/提交包 |

---

## 🔑 核心认知

### 已验证经验

1. Co-DETR + InternImage-L continuation 已在线上验证超过老 RGC-DINO baseline：48.727 vs 45.044，正式作为当前主线；当前决策 anchor 是 fresh epoch7 + class thresholds strict mAP `0.4262677082771047`。
2. 当前已上线高分仍是 RGB-only Co-DETR；IR/Depth/RGC 三模态融合是下一阶段提分方向，不再先退回旧 RGC-DINO 主线。
3. 错误融合比不融合更糟；不能把多模型/多尺度输出简单平均。
4. 对手 50.8190 说明，class-wise threshold、误检抑制和高分辨率 sweet spot 在 48-51 分段可能贡献接近 1 分，应先榨干当前强 checkpoint。

### 当前主线原则

```text
强单模型优先
→ Co-DETR + InternImage-L continuation 作为当前主线
→ checkpoint/strict final-TXT sweep 选择 best epoch（当前 fresh epoch7 + class thresholds）
→ 继续执行 NMS / image-side sweet spot / hard-val / prediction diagnostics
→ 只提交 local strict mAP 超过 0.426267708、线上基线超过 48.727 且预测分布合理的候选
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

### 立即插入：对手经验吸收 / 后处理与验证升级

1. 基于 epoch20 best / epoch24 生成低阈值 validation raw prediction cache。
2. 实现并运行 class-wise threshold sweep。
3. 联合搜索 `image_max_side`、NMS IoU、candidate threshold、final top100。
4. 输出 class-wise AP/count/score histogram、boxes/image、top100 截断报告。
5. 构建 hard validation，防止阈值过拟合普通 fold。
6. 只在 strict mAP、hard-val、预测分布都过门禁后提交。
7. 已验证 positive：2026-06-22 fresh epoch7 class-threshold candidate strict `0.4262677082771047`，hard-val `0.28694971837472955`，线上 `48.727`，成为新 anchor。
8. 已验证 image-side 诊断：旧 eval_s768/s832/s896 输出完全一致的根因是 `(1333, side)` 被宽度 cap 主导，三者都实际 resize 到 `1333×750`；修正 s832 为 `(1479,832)` 后输出 hash 改变，但 strict `0.4227484953374498` 低于 anchor，不进入 test ZIP / promotion / 提交。
9. 已验证 negative：2026-06-22 high-res fine-tune best strict `0.4188436556179077` 低于 fresh epoch7 raw/class-threshold anchor；不生成 test ZIP、不 promote、不提交。

### Week 2+：Co-DETR + InternImage-L

1. 准备 InternImage-L 公开预训练权重。
2. 写 backbone wrapper 和 key-loading report。
3. 先 640/704 低分辨率训练 12-18 epoch。
4. 验证通过且完成后处理/验证升级后，启动 50-72 epoch 主训练或高分辨率微调。

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

- 2026-06-22：记录 Co-DETR fresh epoch7 + class thresholds 为当前 48.727 anchor；raw fresh epoch7 为 48.6960；high-res fine-tune strict mAP 退步，未进入 promotion/提交。
- 2026-06-18：主路线调整为 Co-DETR + InternImage-L；删除测试集伪标签训练、多模型简单融合等不合规/高风险默认路线。
- 2026-06-15：重组文档结构，创建归档目录。
- 2026-06-15：2-fold 融合失败（44.263）。
- 2026-06-15：TTA 朴素平均失败（34.872）。
- 2026-06-15：R50 baseline 建立（45.044）。
