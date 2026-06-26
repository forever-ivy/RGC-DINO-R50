# Co-DETR + InternImage-L 训练交接清单

> 更新日期：2026-06-26
> 目标：让当前 Co-DETR + InternImage-L 主线在不打断交互会话的前提下，通过轻量 preflight、LSF 脚本和 strict validation 门禁继续推进。
> 状态：`external/Co-DETR`、`external/IDEA-Research-DINO`、`external/InternImage-master` 已在仓库外部目录中；InternImage-L 与 Co-DETR/Co-DINO 公开预训练权重已放在默认路径。R50 sanity 权重如需运行 Stage-0 可另行补齐。

---

## 1. 当前主线定位

当前冲榜 anchor 是：

```text
Co-DETR InternImage-L GPU1 load-from epoch6
+ class thresholds
+ legal top100 allocation person0865/light10625/uav0825/boat003
leaderboard 50.353
strict final-TXT mAP 0.4379615851682616
hard-val 0.29545499238138817
```

后续训练、continuation、high-res、train-all、TTA/tile 或 IR/Depth fusion 迁移，都必须以这个 anchor 为门槛：strict final-TXT mAP 必须超过 `0.4379615851682616`，promotion metadata 中 leaderboard baseline 必须不低于 `50.353`，hard-val/预测分布不能明显退化。

---

## 2. 当前已准备好的文件

### 配置

- `configs/codetr_internimage_l_stage0.yaml`
  项目级规划配置，记录 Co-DETR + InternImage-L 主线、显存策略、验证门禁。

- `configs/codetr_internimage_l_mm_config.py`
  Co-DETR/MMDetection-style Stage-1 训练配置：当前是 RGB-only InternImage-L 桥接，先稳定 detector/backbone/weight loading，再迁移三模态 RGC fusion。

- `configs/codetr_internimage_l_aic2026_test.py`
  AIC2026 test-set 推理配置。

- `configs/codetr_internimage_l_eval_s*.py`
  image-side / resize 诊断配置；旧 `s768/s832/s896` width-cap 问题已记录在主路线文档中。

- `configs/codetr_r50_stage0_mm_config.py`、`configs/codetr_r50_tiny_mm_config.py`
  Co-DETR-R50/tiny sanity 配置，只用于链路检查，不作为最终主线。

### 脚本

- `scripts/check_codetr_environment.py`
  检查 Co-DETR/MMCV/MMDetection 相关运行环境。

- `scripts/check_codetr_integration.py`
  检查 `external/Co-DETR` 目录结构和公开预训练权重路径。

- `scripts/export_codetr_coco.py`、`scripts/export_codetr_test_coco.py`
  导出 Co-DETR 可读取的 COCO RGB 数据集。

- `scripts/write_bsub_codetr_smoke.py`
  生成只做集成检查的 LSF 脚本，不训练。

- `scripts/write_bsub_codetr_train.py`
  生成 Co-DETR stage train LSF 脚本，不自动提交。

- `scripts/write_bsub_codetr_continue.py`
  生成 continuation LSF；默认带 strict mAP / leaderboard baseline 门禁，未超过 anchor 时不进入 test inference/promotion。

- `scripts/continue_codetr_internimage_l_stage1.sh`
  continuation job 内部流程：训练/验证/strict final-TXT sweep/门禁后 test-and-promote。

- `scripts/prepare_codetr_training.py`
  一键执行轻量准备：导出 COCO、生成 smoke/train LSF、做 preflight。

- `scripts/audit_weight_load_report.py`
  检查权重加载报告，防止 backbone 主干大量 missing 还继续长训。

### 后处理与 promotion

- `scripts/cache_codetr_predictions.py`
- `scripts/sweep_codetr_class_thresholds.py`
- `scripts/sweep_codetr_top100_allocation.py`
- `scripts/sweep_codetr_submission_params.py`
- `scripts/codetr_results_to_submission.py`
- `scripts/run_codetr_test_and_promote.sh`
- `scripts/promote_submission_candidate.py`
- `scripts/monitor_competition.py`

---

## 3. 外部资源与默认路径

所有资源必须放在 `/data1/liuxuan/` 下，不能放到 `/home/`。

### 代码树

```text
external/Co-DETR/
external/IDEA-Research-DINO/        # legacy / 对照
external/InternImage-master/        # InternImage 参考代码
```

`external/Co-DETR` 预期至少包含：

```text
external/Co-DETR/tools/train.py
external/Co-DETR/tools/test.py
external/Co-DETR/tools/dist_train.sh
external/Co-DETR/mmdet/
external/Co-DETR/projects/configs/ 或 external/Co-DETR/configs/
external/Co-DETR/ops_dcnv3/
```

### 公开预训练权重

当前主线默认路径：

```text
/data1/liuxuan/checkpoints/internimage/internimage_l_public_pretrain.pth
/data1/liuxuan/checkpoints/codetr/codetr_internimage_l_public_pretrain.pth
```

可选 R50 sanity 权重：

```text
/data1/liuxuan/checkpoints/codetr/codetr_r50_public_pretrain.pth
```

说明：

- InternImage-L backbone 权重用于 backbone 初始化。
- Co-DETR/Co-DINO detection 权重用于 detector/head/transformer 初始化。
- R50 权重只用于 Stage-0 sanity；缺失时不要影响当前 InternImage-L anchor 的整理与后处理复盘。
- 权重必须是公开预训练权重，不能来自外部训练数据的私有模型。

---

## 4. 轻量检查顺序

```bash
source /data1/liuxuan/activate-py310.sh
python scripts/check_codetr_environment.py
python scripts/check_codetr_integration.py --codetr-root external/Co-DETR
python scripts/prepare_codetr_training.py --no-require-weights
```

如要强制检查权重：

```bash
python scripts/prepare_codetr_training.py
```

如果 `prepare_codetr_training.py` 显示 `ready_for_manual_bsub: false`，不要提交训练，先修 blocker。

---

## 5. 手动 LSF 工作流

preflight 通常会生成：

```text
outputs/jobs/codetr_smoke.lsf
outputs/jobs/codetr_internimage_l_stage0_train.lsf
```

只在用户/资源确认后提交：

```bash
bsub < outputs/jobs/codetr_smoke.lsf
# smoke 通过后再提交训练/continuation
bsub < outputs/jobs/codetr_internimage_l_stage0_train.lsf
```

Continuation 优先使用：

```bash
python scripts/write_bsub_codetr_continue.py --output outputs/jobs/codetr_continue.lsf
# bsub < outputs/jobs/codetr_continue.lsf
```

---

## 6. 训练阶段纪律

### Stage-0：RGB Co-DETR sanity

- 目的：验证 Co-DETR 训练/评估/COCO 数据链路。
- 不追求最高分。
- 通常不提交 leaderboard。

### Stage-1：Co-DETR + InternImage-L low-res

- 输入 640/704。
- batch=1/GPU。
- AMP + gradient checkpointing + accumulation。
- 先确认 loss/mAP/预测分布正常。

### Stage-2：长训/continuation 主模型

- 以当前 best checkpoint/recipe 为 anchor。
- 每次候选都必须生成 validation metric、strict final-TXT report、hard-val/预测分布报告。
- 只有超过当前 strict `0.4379615851682616` 且 hard-val 不崩，才允许 test inference / promotion。

---

## 7. 禁止事项

- 不在交互 shell 直接启动长 GPU 训练。
- 不使用测试集伪标签训练。
- 不使用外部训练数据。
- 不做简单投票/平均 ensemble。
- 不复用对手权重、提交包或训练产物。
- 不提交 validation/OOF/debug/empty/partial ZIP。
- 不在没有 local mAP、manifest、prediction distribution report、promotion reason 的情况下提交。

---

## 8. 最短安全路径

```bash
source /data1/liuxuan/activate-py310.sh
python scripts/check_codetr_environment.py
python scripts/check_codetr_integration.py --codetr-root external/Co-DETR
python scripts/prepare_codetr_training.py --no-require-weights
python scripts/write_bsub_codetr_continue.py --output outputs/jobs/codetr_continue.lsf
# 用户确认资源后：bsub < outputs/jobs/codetr_continue.lsf
```

后续所有候选必须通过 strict final-TXT、hard-val/预测分布和 promotion metadata 门禁，才能进入 `outputs/submissions/`。
