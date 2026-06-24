# 对手 urban-visual-recognition 项目复盘与可借鉴清单

> 复盘对象：<https://github.com/oscar030406/urban-visual-recognition><br>
> 对应榜单记录：AIC-2026-35512237 / 魔丸和灵珠 / 2026-06-15 20:38:57 / **50.8190**<br>
> 本地镜像位置：`/data1/liuxuan/projects/urban-visual-recognition-opponent`<br>
> 镜像说明：由于服务器 `git clone` 中途超时，本次通过 GitHub API 下载了仓库文本文件树；未下载 release 中的数据集、权重、提交 ZIP、PDF/DOCX 大文件。<br>
> 检查 commit：`84fc51f26b1b6894e9ed75185154770faf569984`（`main`）
> 复盘日期：2026-06-20

---

## 0. 总结结论

这个对手项目最值得学习的不是“换成 YOLO11M”本身，而是它在小数据、多模态、排行榜反馈不稳定场景下形成的一套**稳健工程策略**：

```text
稳定 RGB 主干
→ IR/Depth 只做轻量可靠性引导，不破坏 RGB 预训练滤波器
→ 高分辨率推理寻找 sweet spot，不盲目拉满
→ 低阈值取候选 + 类别级阈值抑制 FP
→ 严格提交校验
→ 通过平台反馈确认误检敏感，但不把平台当无限验证集
```

对我们当前 `RGC-DINO-R50 / Co-DETR + InternImage-L` 主线的直接启发：

1. **短期最高 ROI：补齐 class-wise threshold sweep / class-wise prediction diagnostics。** 这可能比继续等待 epoch24 自然涨分更有效。
2. **高分辨率推理必须与误检控制绑定。** 对手 `1536 + 极低 conf` 掉分，`1408 + 更保守阈值` 提升；我们也不能只追更大 `image_max_side`。
3. **三模态融合应坚持 RGB 主导 + IR/Depth 可靠性引导。** 对手最稳的是输入层 RGB-guided-RDT；我们应在 Co-DETR/InternImage-L 特征层做更可控的 reliability-gated residual fusion。
4. **不要盲目更大模型、盲目加小目标头、盲目 train-all。** 对手均有负面证据。
5. **需要 hard validation / class-size AP / modality reliability analysis。** 随机验证集与平台趋势不一致，是双方共同风险。

---

## 1. 仓库与产物整理

### 1.1 本地整理位置

对手项目的文本镜像已从当前仓库 `external/` 移出，放到：

```text
/data1/liuxuan/projects/urban-visual-recognition-opponent
```

该目录不在本仓库内，避免污染 `RGC-DINO-R50` 的 git 状态和工程边界。

目录内新增了本地说明文件：

```text
/data1/liuxuan/projects/urban-visual-recognition-opponent/_LOCAL_COPY_NOTE.md
```

说明其来源、commit、下载范围和大文件未下载事实。

### 1.2 未下载内容与规则提醒

对手 release 元数据中包含：

- 11 个数据集 ZIP 分片，总大小约 19GB；
- `final_artifacts.zip`，约 41MB，包含最好权重和最好提交包；
- `SHA256SUMS.txt`。

这些大文件**没有下载**。即便能下载，也不应直接使用对手最好权重或提交 ZIP：

- 对手权重属于外部训练产物/对手产物，不应用于我们的提交；
- 对手提交包不可复用；
- 可学习其公开代码、报告、后处理思想，但我们的训练、推理和提交必须来自本项目本地合法流程。

---

## 2. 对手真实方案

### 2.1 主体方案

对手 README 明确写明：

```text
YOLO11M + RGB-guided-RDT
```

其处理方式是：保留 RGB 三通道形式，用红外和深度生成空间权重，对 RGB 局部亮度做调制。这样既继续利用 YOLO 的公开预训练权重，又引入三模态信息。

证据：

- `/data1/liuxuan/projects/urban-visual-recognition-opponent/README.md:5`
- `/data1/liuxuan/projects/urban-visual-recognition-opponent/docs/阶段性总结.md:13-24`

### 2.2 分数记录

对手公开文档记录：

| 项目 | 平台分数 |
|---|---:|
| 早期稳定提交 | 50.0380 |
| 最好提交 | 50.8190 |
| 提升 | +0.7810 |

证据：

- `/data1/liuxuan/projects/urban-visual-recognition-opponent/README.md:7-25`
- `/data1/liuxuan/projects/urban-visual-recognition-opponent/docs/阶段性总结.md:26-44`
- `/data1/liuxuan/projects/urban-visual-recognition-opponent/docs/实验结果摘要.csv:2-12`

其最好提交包名记录为：

```text
artifacts/final/submission_best_50.8190_person003.zip
```

对应模型权重记录为：

```text
artifacts/final/rgb_guided_rdt_yolo11m_1280_ft2_e70_best.pt
```

注意：这些 artifacts 未纳入 git，也未被本次下载。

---

## 3. RGB-guided-RDT：最值得研究的三模态输入构造

### 3.1 它不是完整三分支融合

对手文档明确说：

> `RGB-guided-RDT` 指以 RGB 为主要视觉信息，同时引入 Infrared 和 Depth 构造多模态输入表示。它不是完整的三分支中期融合结构，但在验证和平台表现上最稳定。

证据：

- `/data1/liuxuan/projects/urban-visual-recognition-opponent/docs/阶段性总结.md:163-170`

这点非常关键：他们的“多模态”不是复杂网络，而是**输入层的 RGB 亮度调制**。

### 3.2 具体公式

核心代码在 `image_ops.py`：

1. IR 转灰度并 CLAHE 增强：

```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
return clahe.apply(gray)
```

证据：

- `/data1/liuxuan/projects/urban-visual-recognition-opponent/src/city_multimodal_detection/image_ops.py:59-66`

2. Depth 归一化并生成 valid mask：

```python
if depth.dtype == np.uint8 or observed_max <= 255.0:
    normalized = to_gray_uint8(depth)
    valid_mask = normalized > 0
else:
    valid_mask = depth_float >= min_depth
    clipped = np.clip(depth_float, min_depth, max_depth)
```

证据：

- `/data1/liuxuan/projects/urban-visual-recognition-opponent/src/city_multimodal_detection/image_ops.py:69-97`

3. IR attention 与 near-depth attention 融合：

```python
ir_attention = _normalize_float01(ir_channel)
near_depth = (1.0 - depth_channel / 255.0) * valid_mask
depth_attention = _normalize_float01(near_depth)
attention = (0.55 * ir_attention + 0.45 * depth_attention) / 1.0
```

4. 对 RGB 做空间增益调制：

```python
gate = base_gate + gain * attention
output = rgb * gate
```

默认：

```text
base_gate = 0.85
gain = 0.30
ir_weight = 0.55
depth_weight = 0.45
```

证据：

- `/data1/liuxuan/projects/urban-visual-recognition-opponent/src/city_multimodal_detection/image_ops.py:188-199`
- `/data1/liuxuan/projects/urban-visual-recognition-opponent/src/city_multimodal_detection/image_ops.py:238-268`
- `/data1/liuxuan/projects/urban-visual-recognition-opponent/configs/rgb_guided_rdt.yaml:14-20`

### 3.3 它为什么稳定

它稳定的原因不是神秘结构，而是非常工程化：

```text
保留 RGB 三通道
→ 保留 COCO / YOLO 预训练滤波器的输入统计
→ IR/Depth 只在空间上增强局部亮度
→ 不强行把 IR/Depth 当作完全等价的图像语义通道
```

这与我们当前方向一致：

```text
RGB primary path
+ IR/Depth auxiliary reliability-gated residual
```

我们不应把 IR/Depth 变成无约束主干输入，而应让它们在可靠时提供残差信号。

### 3.4 对我们可迁移的方式

不建议直接把主线改成“预处理 RGB-guided-RDT + YOLO11M”。更适合我们的迁移方式有三种：

#### 方式 A：作为低成本 ablation

新增一个可选输入预处理：

```text
visible_rdt = RGB * (0.85 + 0.30 * attention(IR_CLAHE, near_depth))
```

用于回答：

- 在同一 Co-DETR / DINO 模型下，RGB-guided-RDT 输入是否优于原 RGB？
- 它是否只对低光/夜间/小目标样本有效？
- 它是否会增加误检？

#### 方式 B：作为 RGC gate 的先验特征

我们已有 `quality_features` 和 `ReliabilityGatedResidualFusion`。可考虑加入：

- IR CLAHE 对比度；
- IR attention mean/std/top percentile；
- depth valid ratio；
- near-depth attention mean/std；
- RGB-guided attention 与 RGB 梯度/亮度的相关性；
- attention 对目标区域/背景区域的差异。

这些可以作为 gate 质量特征，而不是直接改图像。

#### 方式 C：作为可视化诊断工具

将 RGB、IR、Depth、attention、RGB-guided-RDT 生成对照图，用于 hard validation 的样本诊断：

- 哪些样本 attention 强但误检多？
- 哪些样本 depth invalid 太多？
- 哪些类别在 IR/Depth 引导下更容易被增强？

---

## 4. 对手涨分的真正来源：class-wise threshold / FP suppression

### 4.1 分数提升路径

对手文档明确写道：

> 分数提升主要来自推理阶段的后处理校准，尤其是类别级置信度阈值调整，而不是模型主体结构本身明显变强。

证据：

- `/data1/liuxuan/projects/urban-visual-recognition-opponent/docs/阶段性总结.md:44`

平台记录：

| 阶段 | candidate | 改动 | 平台分 |
|---|---|---|---:|
| baseline | `submission_rgb_guided_rdt_yolo11m_ft2_i1408_c0005_iou065_tta_checked.zip` | 早期稳定基线 | 50.0380 |
| global_threshold | `submission_lockedbest_i1408_c0015_iou065_tta_checked.zip` | 更保守全局 conf | 50.6880 |
| classsafe_a | `submission_classsafe_a_rare003_i1408_iou065_tta_checked.zip` | 提高稀有类阈值 | 50.75 |
| classsafe_b | `submission_classsafe_b_rare003_sign002_i1408_iou065_tta_checked.zip` | 稀有类 + sign | 50.77 |
| classsafe_c | `submission_classsafe_c_rare003_sign002_person002_i1408_iou065_tta_checked.zip` | 继续调 person | 50.78 |
| diag_person003 | `submission_diag_person003_from_best_i1408_iou065_tta_checked.zip` | person 阈值提高 | **50.8190** |

证据：

- `/data1/liuxuan/projects/urban-visual-recognition-opponent/docs/实验结果摘要.csv:2-12`
- `/data1/liuxuan/projects/urban-visual-recognition-opponent/docs/阶段性总结.md:323-367`

### 4.2 重要启发

对手从 50.0380 到 50.8190 的主要收益不是重新训练模型，而是：

```text
更合适的推理尺度
+ 更保守全局阈值
+ 类别级阈值
+ 控制低置信度 FP
```

这与我们历史经验一致：

- 低阈值盲调导致分数下降；
- 朴素 TTA 平均导致约 10 分级别暴跌；
- 弱模型融合引入 FP，得不偿失；
- mAP@50:95 对 FP 和排序非常敏感。

### 4.3 对手 class-wise threshold 脚本逻辑

`predict_class_threshold_submit.py` 的核心流程：

1. 读取 class threshold；
2. `base_conf = min(class_conf)`，或使用手动 base-conf；
3. 用 YOLO 生成较多候选：`candidate-max-det=300`；
4. Python 里按类别阈值过滤；
5. 每图最多写 `max-det=100`；
6. manifest 记录过滤前后预测数量。

证据：

- 参数：`/data1/liuxuan/projects/urban-visual-recognition-opponent/scripts/predict_class_threshold_submit.py:19-40`
- threshold 解析：`/data1/liuxuan/projects/urban-visual-recognition-opponent/scripts/predict_class_threshold_submit.py:43-72`
- 过滤：`/data1/liuxuan/projects/urban-visual-recognition-opponent/scripts/predict_class_threshold_submit.py:86-92`
- 推理与过滤：`/data1/liuxuan/projects/urban-visual-recognition-opponent/scripts/predict_class_threshold_submit.py:129-145`
- manifest：`/data1/liuxuan/projects/urban-visual-recognition-opponent/scripts/predict_class_threshold_submit.py:147-174`

`sweep_class_thresholds.py` 的核心流程：

1. 低 base conf 跑一次验证集，缓存 raw predictions；
2. 对 12 类逐类贪心搜索阈值；
3. 目标是 `mAP@50:95`，可加 `box_penalty * boxes_per_image` 惩罚过多框；
4. 输出 JSON/CSV/Markdown，包含 class threshold、class AP、class box counts。

关键默认参数：

```text
imgsz = 1408
base_conf = 0.0005
init_conf = 0.00125
iou = 0.65
candidate_max_det = 300
max_det = 100
augment = true
threshold candidates = 0.00075,0.001,0.00125,0.0015,0.002,0.003
rounds = 2
```

证据：

- `/data1/liuxuan/projects/urban-visual-recognition-opponent/scripts/sweep_class_thresholds.py:29-49`
- raw prediction cache：`/data1/liuxuan/projects/urban-visual-recognition-opponent/scripts/sweep_class_thresholds.py:129-163`
- threshold apply：`/data1/liuxuan/projects/urban-visual-recognition-opponent/scripts/sweep_class_thresholds.py:166-180`
- AP 计算：`/data1/liuxuan/projects/urban-visual-recognition-opponent/scripts/sweep_class_thresholds.py:189-241`
- objective：`/data1/liuxuan/projects/urban-visual-recognition-opponent/scripts/sweep_class_thresholds.py:253-254`
- 贪心搜索：`/data1/liuxuan/projects/urban-visual-recognition-opponent/scripts/sweep_class_thresholds.py:296-318`
- 输出 JSON：`/data1/liuxuan/projects/urban-visual-recognition-opponent/scripts/sweep_class_thresholds.py:331-356`

### 4.4 我们应该如何迁移

当前 `src/rgc_dino/dino_inference.py` 有：

- 全局 `score_threshold`；
- `max_detections`；
- optional classwise NMS；
- `ClasswiseScoreCalibrator`。

但缺少：

```text
per-class hard threshold
candidate_max_detections before final top100
class-wise threshold sweep over validation predictions
class-wise box count / score histogram / AP diagnostics
```

建议新增能力：

```text
scripts/sweep_class_thresholds_rgc_dino.py
scripts/analyze_prediction_distribution.py
scripts/infer_rgc_dino.py --class-score-thresholds thresholds.json
scripts/infer_rgc_dino.py --candidate-max-detections 300 --max-detections 100
```

推荐 pipeline：

```text
checkpoint
→ validation raw prediction cache with very low base threshold
→ class-wise threshold greedy sweep
→ strict final-TXT validation mAP
→ class-wise counts / boxes-per-image / score hist
→ hard validation replay
→ only then package test ZIP
```

这是对手最值得立即借鉴的部分。

---

## 5. 高分辨率推理：找 sweet spot，不是盲目拉满

### 5.1 对手结论

对手尝试过：

```text
1536 + 极低置信度阈值
```

平台反而下降到 48/49 左右。随后改用：

```text
1408 + 更保守阈值
```

平台表现更稳定并刷新成绩。

证据：

- `/data1/liuxuan/projects/urban-visual-recognition-opponent/docs/阶段性总结.md:105-120`
- `/data1/liuxuan/projects/urban-visual-recognition-opponent/docs/阶段性总结.md:315-322`

### 5.2 对我们意味着什么

我们不能简单认为：

```text
更大 image_max_side = 更高 mAP
```

更大分辨率会同时带来：

- 小目标召回提升；
- 更多低置信度候选；
- 更多局部纹理误检；
- top100 截断风险；
- 更高显存/时间成本。

因此我们的高分辨率计划应变成联合 sweep：

```text
image_max_side: 800 / 832 / 896 / 960
nms_iou: 0.55 / 0.65 / 0.75
base_score_threshold: 0.0005 / 0.001 / 0.0015 / 0.003
class-wise thresholds: greedy sweep
max candidate boxes: 300
final max boxes: 100
```

而不是只做 epoch20/24/longer-train 的 checkpoint selection。

---

## 6. 对手失败实验与我们应避免的坑

### 6.1 简单 IR/Depth gate 未稳定超过基线

对手 gate 实验：

```text
auxgate_rdtgate5_unfreeze_lr8e6_a002_e80_seed42_gpu0
mAP@50-95 ≈ 0.4979
```

低于稳定方案约 `0.5005`。

证据：

- `/data1/liuxuan/projects/urban-visual-recognition-opponent/docs/阶段性总结.md:208-228`

对手判断原因：

1. RGB/IR/Depth 虽然空间对齐，但局部边界或目标区域仍可能有偏差；
2. Depth 无效区域会引入噪声；
3. IR/Depth 在不同场景下可靠性不同；
4. 简单 gate 没区分不同尺度特征。

证据：

- `/data1/liuxuan/projects/urban-visual-recognition-opponent/docs/阶段性总结.md:230-239`

对我们启发：

- RGC fusion 不能只是“把 IR/Depth 加进去”；
- 必须保留 zero-init residual、bounded gate、quality features、depth valid mask；
- 要记录 gate 分布和按场景/类别/尺度的效果。

### 6.2 YOLO11L / YOLO11X 不如 YOLO11M

对手发现更大 YOLO 模型不如 YOLO11M，判断瓶颈不在容量，而在：

- 误检；
- 验证集失真；
- 多模态融合方式；
- 小数据过拟合。

证据：

- `/data1/liuxuan/projects/urban-visual-recognition-opponent/docs/阶段性总结.md:245-256`

对我们启发：

- Co-DETR + InternImage-L 已经很强，但继续堆更大 backbone 不一定是正确方向；
- 当前应优先解决后处理、验证、hard-val、class-wise FP、三模态可靠性。

### 6.3 直接加 P2 小目标 head 失败

对手 P2 head 实验：

```text
mAP@50-95 ≈ 0.4021
```

明显低于主线。

证据：

- `/data1/liuxuan/projects/urban-visual-recognition-opponent/docs/阶段性总结.md:258-284`

对我们启发：

- 不要在当前阶段直接大改 Co-DETR head；
- 小目标问题优先从高分辨率、tile inference、class-wise threshold、loss weighting、定位诊断做起；
- 如果改结构，必须有完整初始化策略和验证门禁。

### 6.4 继续训练与 train-all 不一定有效

对手基于最好权重做低 LR 继续训练、冻结微调、全 2000 图训练，结果没有稳定超过原基线；全量训练虽然让重合验证集分数升高，但平台下降。

证据：

- `/data1/liuxuan/projects/urban-visual-recognition-opponent/docs/阶段性总结.md:286-296`

对我们启发：

- epoch24 如果超不过 epoch20，不应继续无限自然长训；
- train-all 只能在 recipe 经 fold/hard-val 证明后做最终模型；
- 没有独立验证的 train-all 分数不可作为提交依据。

### 6.5 1536 + 极低 conf 失败

对手明确把 `1536 高分辨率 + 极低 conf` 列为暂缓/排除方向，因为平台误检增多、表现下降。

证据：

- `/data1/liuxuan/projects/urban-visual-recognition-opponent/docs/阶段性总结.md:371-385`

对我们启发：

- 不要重复低阈值盲调；
- 单模型 TTA/tile 必须先在 validation 上证明，并配套 NMS/class threshold；
- 这与我们 TTA failure lesson 完全一致。

---

## 7. 验证体系：对手最有价值的战略提醒

### 7.1 随机验证集与平台不一致

对手明确说：

> 本地验证集和平台测试集存在分布差异。单纯依赖随机 `1600/400` 划分得到的本地 mAP，不能完全代表平台趋势。

证据：

- `/data1/liuxuan/projects/urban-visual-recognition-opponent/docs/阶段性总结.md:397-401`

他们建议按场景、序列或拍摄来源重新划分，并构建 hard validation，覆盖：

- 夜间；
- 弱光；
- 遮挡；
- 小目标；
- 人群密集；
- 深度无效；
- 红外弱响应。

证据：

- `/data1/liuxuan/projects/urban-visual-recognition-opponent/docs/阶段性总结.md:407-423`

### 7.2 我们应如何落地

我们已有 grouped stratified split，优于对手的随机 1600/400；但仍需补：

```text
hard_val_manifest.jsonl
hard_val_tags.json
per-sample quality stats
per-class AP on hard-val
per-size AP on hard-val
postprocess sweep on normal val + hard val both pass才提交
```

建议 hard-val 标签包括：

```text
low_light
night_like
high_depth_invalid_ratio
weak_ir_contrast
small_object_heavy
crowded_person
occlusion
far_depth_dominated
rare_classes
```

---

## 8. 模态消融：决定 RGC fusion 是否真正有效

对手指出，还需要系统回答 IR 和 Depth 是否分别有效，计划同模型同划分对比：

| 实验 | 输入 |
|---|---|
| A | RGB-only |
| B | IR-only |
| C | Depth-only |
| D | RGB + IR |
| E | RGB + Depth |
| F | RGB + IR + Depth |

证据：

- `/data1/liuxuan/projects/urban-visual-recognition-opponent/docs/阶段性总结.md:425-447`

对我们而言，这一步很关键。否则即使 RGC fusion 训练成功，也无法判断：

- IR 是否只在夜间/弱光有帮助；
- Depth 是否对遮挡/尺度/小目标有帮助；
- IR 与 Depth 同时引入是否互相污染；
- gate 是否学会在无效 depth 下关闭 depth 分支；
- 三模态是否只是提高 recall 但带来更多 FP。

建议最小落地方式：

```text
1. 不先全量长训，先做短训/冻结/低成本 ablation。
2. 在 same checkpoint recipe 下比较 RGB-only、RGB+IR、RGB+Depth、RGB+IR+Depth。
3. 记录每类 AP、每类框数、每图框数、gate 均值/方差、hard-val 表现。
4. 只有证明 IR/Depth 在 hard-val 或特定类/场景有效，才扩大训练投入。
```

---

## 9. 定位质量：mAP@50 与 mAP@50:95 的差距

对手稳定方案本地：

```text
mAP@50     ≈ 0.772
mAP@50-95  ≈ 0.5005
```

证据：

- `/data1/liuxuan/projects/urban-visual-recognition-opponent/docs/阶段性总结.md:19-24`
- `/data1/liuxuan/projects/urban-visual-recognition-opponent/docs/阶段性总结.md:171-178`

他们判断：模型不是完全检测不到，而是高 IoU 阈值下定位精度不足。

建议方向：

- 分析 AP50 vs AP75/AP90；
- class-wise / size-wise AP；
- 小目标 loss weighting；
- 中心点距离、宽高差异、长宽比、对角线相似度等定位辅助损失；
- 优先关注 `uav / ball / light / sign / person` 等小目标或弱类。

证据：

- `/data1/liuxuan/projects/urban-visual-recognition-opponent/docs/阶段性总结.md:478-489`

对我们而言，Co-DETR/DINO 可能定位质量强于 YOLO，但仍应做：

```text
AP50 / AP75 / AP90 gap by class
small / medium / large AP
box center error distribution
box size error distribution
top false positive localization patterns
```

这能指导 tile/high-res/loss/postprocess，而不是盲目改结构。

---

## 10. 和我们当前路线的映射

### 10.1 对手经验与本项目路线对照

| 对手经验 | 证据 | 对我们意味着什么 |
|---|---|---|
| RGB-guided-RDT 最稳定 | README + 阶段总结 | 保持 RGB 主分支，IR/Depth 做可靠性引导 |
| 分数提升主要来自后处理 | 阶段总结 + CSV | 立即补 class-wise threshold sweep |
| 1408 比 1536+低 conf 稳 | 阶段总结 | 高分辨率必须配 FP 控制 |
| 简单 gate 未稳定超过基线 | gate 实验 | RGC 必须 reliability-aware、per-level、zero-init、bounded |
| YOLO11L/X 不如 M | 模型规模复测 | 盲目堆模型不是当前最高 ROI |
| P2 head 大幅下降 | P2 实验 | 不要直接大改检测头 |
| train-all 平台下降 | continue/train-all 复盘 | train-all 只能作为 recipe 证明后的最终步骤 |
| random val 不可靠 | hard-val 建议 | 本项目也要 hard validation |

### 10.2 对当前 repo 的具体触点

| 目标 | 当前最相关文件/模块 | 建议 |
|---|---|---|
| class-wise threshold | `src/rgc_dino/dino_inference.py`, `scripts/infer_rgc_dino.py` | 新增 per-class hard threshold 和 candidate max det |
| threshold sweep | `src/rgc_dino/metrics.py`, `scripts/evaluate_predictions.py` | 新增 sweep 脚本，缓存 raw predictions |
| prediction diagnostics | `src/rgc_dino/submission.py`, `submission_manifest.py` | 增加 class-wise count/score histogram |
| RGB-guided-RDT ablation | `src/rgc_dino/dino_dataset.py`, `quality_features.py` | 可选输入预处理或 gate quality feature |
| RGC fusion | `src/rgc_dino/models/rgc_fusion.py`, `side_encoder.py` | 加入 IR/Depth saliency stats，记录 gate 分布 |
| hard-val | `src/rgc_dino/splits.py`, `training_splits.py` | 构建 hard-val manifest，不替代现有 fold |
| high-res/tile | `scripts/infer_rgc_dino.py` | image side/NMS/class-threshold 联合 sweep |

---

## 11. 优先级建议

### P0：立刻实现 class-wise threshold sweep

这是对手最直接、最可复用、最可能短期提分的经验。

目标：

```text
epoch20 best / epoch24
→ low-conf val raw predictions
→ class-wise threshold sweep
→ strict final-TXT mAP
→ class-wise count/score/AP diagnostics
```

预期收益：

- 如果当前模型已有足够候选框，可能带来 0.5-1.5 分级别提升；
- 即便不提升，也能明确哪些类低置信度 FP 最严重。

风险：

- 小验证集可能过拟合阈值；
- 必须在 hard-val 或额外 split 上复核；
- 不应通过大量平台提交做单类别诊断。

### P1：高分辨率推理 + class-wise threshold 联合 sweep

建议 sweep：

```text
image_max_side: 800 / 832 / 896 / 960
base_score_threshold: 0.0005 / 0.001 / 0.0015 / 0.003
nms_iou_threshold: 0.55 / 0.65 / 0.75
candidate_max_detections: 300
final_max_detections: 100
```

只保留 strict final-TXT mAP 和预测分布都正常的候选。

### P2：构建 hard validation 与 class/size diagnostics

先不需要复杂模型改动，先让评价体系更可信。

输出：

```text
hard_val_manifest.jsonl
hard_val_tags.json
class_size_ap_report.md
prediction_distribution_report.json
```

### P3：RGB-guided-RDT 低成本 ablation

做一个可开关的预处理/特征统计实验：

```text
rgb_guided_rdt_input: true/false
ir_clahe_stats: true
depth_near_saliency_stats: true
```

用于判断它对 Co-DETR/InternImage-L 是否也有帮助。

### P4：RGC-Co-DETR-InternImage-L fusion 继续推进

吸收对手结论：不要三套完整 backbone，不要强融合；坚持：

```text
RGB main branch
IR/Depth lightweight side encoder
quality-aware per-level gate
zero-init residual
bounded auxiliary contribution
depth valid mask
hard-val and gate diagnostics
```

---

## 12. 明确不建议做的事

### 12.1 不要使用对手权重或提交包

公开代码可以学习，但对手 release 权重/提交 ZIP 不应进入我们的训练或提交链路。

### 12.2 不要主线切到 YOLO11M

YOLO11M 到 50.819 说明它是强 baseline，但它与头部 58/61 仍有距离。它适合做：

```text
对照线 / fallback / 工程参考 / 后处理参考
```

不应替代当前 Co-DETR + InternImage-L 主线。

### 12.3 不要盲目平台单类别诊断

对手通过多次平台提交找到 `person003`。我们应将该流程本地化，避免把 leaderboard 当验证集。

### 12.4 不要重复 1536 + 极低 conf

对手和我们历史都证明：低阈值/高候选框如果没有 FP 控制，会掉分。

### 12.5 不要直接改检测头

P2 head 的失败说明，大改 head 需要完整初始化和训练策略。当前更稳的是后处理、高分辨率、tile、loss/diagnostics。

---

## 13. 建议形成的下一步任务清单

### 任务 1：class-wise threshold 支持

- 修改 `dino_result_to_detection_labels` 或其调用路径；
- 支持 JSON/list/dict per-class threshold；
- 区分 candidate threshold 与 final class threshold；
- manifest 写入 class thresholds 和过滤前后预测数。

### 任务 2：validation raw prediction cache

- 保存每张图每个候选框：class、score、xyxy/xywhn；
- 支持不同 `image_max_side`、NMS、candidate max det；
- 避免重复跑 GPU 推理。

### 任务 3：greedy threshold sweep

- 以 `mAP@50:95` 为目标；
- 可加 `box_penalty * boxes_per_image`；
- 输出 class thresholds、class AP、class counts、boxes/image。

### 任务 4：prediction diagnostics

- 每类预测数量；
- 每类 score histogram；
- 每图框数分布；
- top100 截断前后损失；
- AP50/AP75/AP90 gap；
- small/medium/large AP。

### 任务 5：hard validation

- 生成 hard-val 标签；
- 对所有后处理 sweep 做 normal-val + hard-val 双验证；
- 防止阈值过拟合 normal-val。

### 任务 6：RGB-guided-RDT ablation

- 实现可选预处理或质量特征；
- 先小规模验证，不进入默认提交路径；
- 分析它在哪些场景有效。

---

## 14. 最终建议

对手项目对我们最有价值的借鉴可以概括为：

```text
短期：class-wise threshold / 高分辨率 sweet spot / FP suppression
中期：hard validation / class-size diagnostics / modality reliability analysis
长期：RGB 主导 + IR/Depth 可靠性引导的轻量特征级融合
```

当前最应该马上做的不是换模型，也不是继续无限等待 epoch24，而是：

```text
把 Co-DETR/InternImage-L 当前 best checkpoint 的后处理诊断补齐：
low-conf candidate → class-wise threshold sweep → high-res/NMS 联合 sweep → strict final-TXT validation → promotion。
```

这条路线与本项目 `docs/FINAL_ROADMAP.md` 的原则一致：强单模型优先、禁止朴素平均/弱融合、提交必须有 local strict mAP 和预测分布证据。对手的 50.8190 说明，在 48-51 分段，**后处理与误检控制足以贡献接近 1 分**；而要追 58/61，仍需在这个基础上推进 Co-DETR InternImage-L 的高分辨率、tile、train-all 和可靠三模态 RGC fusion。
