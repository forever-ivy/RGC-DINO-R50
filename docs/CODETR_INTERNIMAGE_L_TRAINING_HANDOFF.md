# Co-DETR + InternImage-L 训练交接清单

> 目标：完成所有前期配置，使外部 Co-DETR 代码和公开预训练权重到位后，可以手动 `bsub` 启动模型训练。  
> 状态：仓库侧配置、COCO 导出、preflight、LSF 生成已准备；当前 blocker 是 `external/Co-DETR` 和公开权重文件尚未放置。

---

## 1. 当前已准备好的文件

### 配置

- `configs/codetr_internimage_l_stage0.yaml`  
  项目级规划配置，记录 Co-DETR + InternImage-L 主线、显存策略、验证门禁。

- `configs/codetr_internimage_l_mm_config.py`  
  外部 Co-DETR/MMDetection-style 训练配置骨架。需要在 `external/Co-DETR` 到位后根据实际 config/module 名称微调。

- `configs/codetr_r50_stage0_mm_config.py`  
  Co-DETR-R50 sanity 配置骨架，用于先验证数据/训练/评估链路。

### 脚本

- `scripts/check_codetr_integration.py`  
  检查 `external/Co-DETR` 目录结构和公开预训练权重路径。

- `scripts/export_codetr_coco.py`  
  将 fold 导出为 Co-DETR 可读取的 COCO RGB sanity 数据集。

- `scripts/write_bsub_codetr_smoke.py`  
  生成只做集成检查的 LSF 脚本，不训练。

- `scripts/write_bsub_codetr_train.py`  
  生成 Co-DETR stage-0/stage-1 训练 LSF 脚本，不自动提交。

- `scripts/prepare_codetr_training.py`  
  一键执行轻量准备：导出 COCO、生成 smoke/train LSF、做 preflight。

- `scripts/audit_weight_load_report.py`  
  检查权重加载报告，防止 backbone 主干大量 missing 还继续长训。

### 测试

- `tests/test_codetr_integration.py`
- `tests/test_write_bsub_codetr_train.py`
- `tests/test_weight_audit.py`

---

## 2. 当前 blocker

运行：

```bash
source /data1/liuxuan/activate-py310.sh
python scripts/prepare_codetr_training.py --no-require-weights
```

当前结果：

```text
coco_ready: true
config_exists: true
codetr_status_ok: false
ready_for_manual_bsub: false
blocked_by:
  - external Co-DETR tree and/or required public pretrained weights are missing
```

也就是说：仓库内部准备已经完成，但还不能训练，因为外部 Co-DETR 代码树尚未放置。

---

## 3. 需要放置的外部资源

所有资源必须放在 `/data1/liuxuan/` 下，不能放到 `/home/`。

### Co-DETR 代码树

目标路径：

```text
/data1/liuxuan/projects/RGC-DINO-R50/external/Co-DETR
```

预期至少包含：

```text
external/Co-DETR/tools/train.py
external/Co-DETR/tools/test.py
external/Co-DETR/tools/dist_train.sh
external/Co-DETR/mmdet/
external/Co-DETR/projects/configs/ 或 external/Co-DETR/configs/
```

如服务器网络可用，可在用户明确同意后通过可达源获取；否则由用户下载后上传到 `/data1/liuxuan/`，再运行：

```bash
source /data1/liuxuan/activate-py310.sh
python scripts/install_codetr_from_archive.py /data1/liuxuan/<Co-DETR-archive>.zip --force
```

该脚本只解压公开代码，不编译 CUDA ops、不下载权重、不启动训练。

### 公开预训练权重

当前配置默认路径：

```text
/data1/liuxuan/checkpoints/internimage/internimage_l_public_pretrain.pth
/data1/liuxuan/checkpoints/codetr/codetr_internimage_l_public_pretrain.pth
/data1/liuxuan/checkpoints/codetr/codetr_r50_public_pretrain.pth
```

说明：

- InternImage-L backbone 权重用于 backbone 初始化。
- Co-DETR/Co-DINO detection 权重用于 detector/head/transformer 初始化。
- R50 权重用于 Stage-0 sanity。
- 权重必须是公开预训练权重，不能来自外部训练数据的私有模型。

---

## 4. 放置资源后的检查顺序

### 4.1 轻量 preflight

```bash
source /data1/liuxuan/activate-py310.sh
python scripts/prepare_codetr_training.py
```

期望：

```text
ready_for_manual_bsub: true
```

如果只想先检查代码树、不强制权重：

```bash
python scripts/prepare_codetr_training.py --no-require-weights
```

### 4.2 生成的 LSF

preflight 会生成：

```text
outputs/jobs/codetr_smoke.lsf
outputs/jobs/codetr_internimage_l_stage0_train.lsf
```

### 4.3 手动提交 smoke

```bash
bsub < outputs/jobs/codetr_smoke.lsf
```

该 job 只做集成检查，不训练。

### 4.4 手动提交 stage-0 训练

只有 smoke 通过后再提交：

```bash
bsub < outputs/jobs/codetr_internimage_l_stage0_train.lsf
```

---

## 5. 训练阶段纪律

### Stage-0：RGB Co-DETR sanity

- 目的：验证 Co-DETR 训练/评估/COCO 数据链路。
- 不追求最高分。
- 不建议直接提交 leaderboard。

### Stage-1：Co-DETR + InternImage-L low-res

- 输入 640/704。
- batch=1/GPU。
- AMP + gradient checkpointing + accumulation。
- 先确认 loss/mAP/预测分布正常。

### Stage-2：长训主模型

- 50-72 epoch。
- 704/768/800/832 多尺度。
- EMA、权重加载审计、validation mAP 每阶段落盘。

---

## 6. 禁止事项

- 不在交互 shell 直接启动长 GPU 训练。
- 不使用测试集伪标签训练。
- 不使用外部训练数据。
- 不做简单投票/平均 ensemble。
- 不提交 validation/OOF/debug/empty/partial ZIP。
- 不在没有 local mAP、manifest、prediction distribution report 的情况下提交。

---

## 7. 当前最短可执行路径

```bash
# 1. 放置 external/Co-DETR 和公开预训练权重

# 2. 运行轻量准备
source /data1/liuxuan/activate-py310.sh
python scripts/prepare_codetr_training.py

# 3. smoke 通过后手动提交训练
bsub < outputs/jobs/codetr_smoke.lsf
bsub < outputs/jobs/codetr_internimage_l_stage0_train.lsf
```

如果 `prepare_codetr_training.py` 仍显示 `ready_for_manual_bsub: false`，不要提交训练，先修 blocker。
