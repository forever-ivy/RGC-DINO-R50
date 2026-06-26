# 项目文档导航

> 面向城市场景视觉多模态目标检测竞赛  
> 当前确认主线：**Co-DETR + InternImage-L 单模型**。当前线上最佳为 `GPU1 load-from epoch6 + top100 allocation person0865/light10625/uav0825/boat003`，leaderboard **50.353**，strict final-TXT mAP `0.4379615851682616`，hard-val `0.29545499238138817`。

旧 RGC-DINO/DINO-R50 PDF、历史方案、旧自动化文档、Swin-L 一次性日志和 agent 旧计划已清理。保留的文档只服务三类用途：官方规则、当前 Co-DETR 主线、当前主线需要吸收的复盘经验。

---

## 当前源头文档

### 1. 主路线

- **[FINAL_ROADMAP.md](FINAL_ROADMAP.md)**
  - 当前唯一主冲榜路线：Co-DETR + InternImage-L fresh/continuation + legal top100 allocation。
  - 当前 anchor：50.353 / strict `0.4379615851682616` / hard-val `0.29545499238138817`。
  - 明确 RGC-DINO/Swin/R50 只保留为历史基线、故障回退或对照。
  - 明确禁止：测试集伪标签训练、朴素 TTA 平均、弱 fold 融合、简单投票/平均 ensemble。

### 2. 2×3090 可行性

- **[2x3090_FEASIBLE_ROADMAP.md](2x3090_FEASIBLE_ROADMAP.md)**
  - 说明 Co-DETR + InternImage-L 在 2×RTX 3090 上的显存压力。
  - 给出 batch=1/GPU、AMP、gradient checkpointing、梯度累积、分辨率阶梯等策略。
  - 把当前 50.353 anchor 作为后续 continuation / train-all / high-res / TTA / tile 的提交门槛。

### 3. Co-DETR 训练交接

- **[CODETR_INTERNIMAGE_L_TRAINING_HANDOFF.md](CODETR_INTERNIMAGE_L_TRAINING_HANDOFF.md)**
  - 记录 Co-DETR / InternImage 外部树、公开权重、COCO 导出、preflight、LSF 生成脚本。
  - 只准备手动 `bsub`，不在交互会话里直接启动长训练。

### 4. 对手经验复盘

- **[OPPONENT_URBAN_VISUAL_RECOGNITION_LESSONS.md](OPPONENT_URBAN_VISUAL_RECOGNITION_LESSONS.md)**
  - 复盘公开 `urban-visual-recognition` 的 YOLO11M + RGB-guided-RDT + class-wise threshold 50.8190。
  - 只吸收可迁移经验：FP suppression、类别阈值、top100 allocation、高分辨率 sweet spot、hard validation、RGB 主导三模态引导。
  - 不复用对手权重、提交包或任何外部训练产物；不改道 YOLO。

### 5. 官方规则

- `../doc/official/面向城市场景的视觉多模态目标检测.pdf`
- `../doc/official/赛事流程.pdf`

官方 PDF 是规则与流程来源，必须保留。

---

## 当前进度与 anchor

| 阶段/方向 | 状态 | 分数/指标 | 决策 |
|---|---|---:|---|
| RGC-DINO-R50 baseline | 已完成 | leaderboard 45.044 | 退为 legacy/fallback |
| 低阈值盲调 | 已失败 | leaderboard 43.955 | 禁止作为默认路线 |
| 朴素 TTA 平均 | 已失败 | leaderboard 34.872 | 禁止；只能用验证过的 NMS/WBF-like/单模型融合策略 |
| 2-fold 弱融合 | 已失败 | leaderboard 44.263 | 禁止弱模型拖累强模型 |
| Swin-L | 对照/复盘 | 不作为上限判断 | 不再占用主线资源 |
| Co-DETR + InternImage-L continue epoch20 | 历史 anchor | leaderboard 48.335 / strict 0.413322 | 已被后续超过 |
| Co-DETR + InternImage-L fresh epoch7 raw | 历史 anchor | leaderboard 48.6960 / strict 0.426216676 | 已被 class-threshold 版本超过 |
| Co-DETR + InternImage-L fresh epoch7 + class thresholds | 历史 anchor | leaderboard 48.727 / strict 0.426267708 | 已被 GPU1 ep6 allocation 超过 |
| Co-DETR GPU1 ep6 + person085/light105 | 历史 anchor | leaderboard 50.349 / strict 0.437940274 | 已被当前 anchor 超过 |
| **Co-DETR GPU1 ep6 + person0865/light10625/uav0825/boat003** | **当前 anchor** | **leaderboard 50.353 / strict 0.437961585 / hard-val 0.295454992** | **后续候选必须超过它** |
| corrected image-side s832 smoke | 诊断完成 | strict 0.422748 | 不提交 |
| high-res fine-tune | 诊断完成 | strict 0.418843656 | 不 promotion / 不提交 |

---

## 当前主线原则

```text
强单模型优先
→ Co-DETR + InternImage-L continuation/checkpoint-selection
→ strict final-TXT validation 是唯一有效本地门禁
→ class-wise threshold / legal top100 allocation / NMS / image-side / hard-val / prediction diagnostics
→ 候选必须 strict mAP > 0.4379615851682616，hard-val 不崩，leaderboard baseline >= 50.353
→ 再推进 longer train / train-all / 单模型 TTA 或 tile / IR-Depth RGC fusion 迁移
→ promotion 后才进入 outputs/submissions/
```

### 已确认失败经验（已从旧文档合并）

- 低阈值盲调会引入大量 FP，历史线上从 45.044 降到 43.955。
- 朴素多尺度 TTA 平均会压低真阳性分数、保留单尺度误检，历史线上掉到 34.872。
- 弱 fold / 弱模型融合会拖累强模型，历史 2-fold 融合降到 44.263。
- 验证集和 leaderboard 不完全一致，必须结合 hard-val、预测分布和提交间隔纪律。

---

## 快速入口

### Co-DETR 主线脚本

- 依赖检查：`scripts/check_codetr_environment.py`、`scripts/check_codetr_integration.py`
- 训练准备：`scripts/prepare_codetr_training.py`
- LSF 生成：`scripts/write_bsub_codetr_train.py`、`scripts/write_bsub_codetr_continue.py`
- validation cache：`scripts/cache_codetr_predictions.py`
- 后处理搜索：`scripts/sweep_codetr_class_thresholds.py`、`scripts/sweep_codetr_top100_allocation.py`、`scripts/sweep_codetr_submission_params.py`
- TXT/ZIP 转换：`scripts/codetr_results_to_submission.py`
- test 推理与 promotion：`scripts/run_codetr_test_and_promote.sh`、`scripts/promote_submission_candidate.py`
- 自动提交监控：`scripts/monitor_competition.py`

### Legacy / fallback 脚本

- RGC-DINO 训练：`scripts/train_rgc_dino.py`
- RGC-DINO 推理：`scripts/infer_rgc_dino.py`
- legacy DINO 集成检查：`scripts/check_dino_integration.py`
- legacy LSF 生成：`scripts/write_bsub_train.py`、`scripts/write_bsub_dino_smoke.py`

这些 legacy 脚本仍可用于对照和后续 fusion 迁移，但不再作为主冲榜路线。

---

## 文档更新日志

- 2026-06-26：清理旧方向文档；文档入口收敛到 Co-DETR + InternImage-L 当前主线、官方规则和必要复盘。
- 2026-06-25：记录 Co-DETR InternImage-L GPU1 load-from epoch6 + top100 allocation person0865/light10625/uav0825/boat003 为当前 50.353 anchor。
- 2026-06-22：记录 fresh epoch7 + class thresholds 48.727、high-res fine-tune 退步。
- 2026-06-18：主路线调整为 Co-DETR + InternImage-L；删除测试集伪标签训练、多模型简单融合等不合规/高风险默认路线。
