# TTA多尺度推理失败分析

> 日期：2026-06-15  
> 失败结果：34.872分（从45.044暴跌-10.17分）  
> 根本原因：朴素平均融合策略错误

---

## 失败事实

- **Baseline**: 45.044分
- **TTA结果**: 34.872分
- **损失**: -10.17分（-22.6%）
- **排名**: 17/19

---

## 技术原因

### 致命代码
```python
# infer_rgc_dino_tta.py
def fuse_boxes_avg(all_boxes, iou_threshold):
    for cluster in clusters:
        score = np.mean([b[5] for b in cluster])  # ✗ 致命错误
```

### 为什么朴素平均导致灾难？

**1. 反向激励机制**
```
真目标（3个尺度都检出）：
  - s640: 0.92
  - s720: 0.88  
  - s800: 0.90
  → 平均后: 0.90  ✗ 置信度降低

误检（只有1个尺度）：
  - s640: 0.65
  → 保持: 0.65   ✗ 置信度不变

结果: 真目标优势缩小 → precision暴跌
```

**2. mAP计算机制**
- mAP按score排序计算precision-recall曲线
- 真目标score被拉低 → 排名下降
- 误检score保持 → 排名相对上升
- 高recall区间precision崩溃

**3. 与低阈值叠加**
- th=0.001已让precision下降1分
- 错误融合进一步放大 → 额外-9分

---

## 正确做法

### 方案1: NMS（Non-Maximum Suppression）
```python
score = max([b[5] for b in cluster])  # ✓ 保留最高分
```
- 多尺度检出 → 保持最强信号
- 单尺度误检 → 不会被强化

### 方案2: WBF（Weighted Boxes Fusion）
```python
score = sum([b[5] for b in cluster]) / normalization_factor
```
- 多尺度检出 → 提升置信度
- 理论最优

### 方案3: Voting
```python
if len(cluster) >= min_votes:
    score = max([b[5] for b in cluster])
```
- 只保留N个尺度都检出的框
- 提升precision

---

## 核心教训

**✗ 错误认知**：
- "多尺度推理一定提升"
- "平均就是融合"
- "未验证就盲目提交"

**✓ 正确认知**：
- **融合策略比推理本身更关键**
- **错误的融合比不融合更糟**
- **必须在验证集先验证**

---

## 已修正

✓ `fuse_multifold_predictions.py` 已改为NMS策略  
✓ 下次融合会用正确方法

---

**教训**：融合算法的选择可以让分数暴跌10分，必须极度谨慎。
