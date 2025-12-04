# Loss 爆炸问题修复

## 问题现象

```
Epoch 17: Loss ≈ 4.x, mAP=4.7% ✓ 正常

Epoch 18:
  Iter 10:  Loss=10.305  ← 突然爆炸！
  Iter 80:  Loss=12.074  ← 最高点
  Iter 400: Loss=8.584   ← 慢慢下降但仍高

典型的梯度爆炸症状！
```

## 🔍 根本原因

### 原因1：Gumbel-Softmax 数值不稳定 ⚠️ **最可能**

**Gumbel 噪声公式**：
```python
gumbel_noise = -log(-log(rand() + 1e-9) + 1e-9)
```

**问题**：
- 两次 log 运算，数值范围可能很大
- 训练后期 score 分布变尖锐，Gumbel 放大不稳定性
- 可能产生极大的值 → softmax 溢出 → 梯度爆炸

### 原因2：移除 no_grad 后梯度累积

- 之前 attention 在 no_grad 里，梯度被"截断"
- 现在梯度完全流通，累积到 Backbone
- 某些 batch 的梯度可能异常大

### 原因3：学习率调度

Epoch 18 时 lr=2.50e-04，可能对当前的梯度幅度太大。

## ✅ 已应用的修复

### 修复1：禁用 Gumbel-Softmax

**文件**：`configs/RGBNT201/DeMo_SDTPS.yml`

```yaml
# 修改前
SDTPS_USE_GUMBEL: True

# 修改后
SDTPS_USE_GUMBEL: False  # 暂时禁用，避免数值不稳定
```

**效果**：
- ✅ 使用确定性的 Top-K 选择
- ✅ 避免 Gumbel 噪声的数值问题
- ⚠️ 失去"可微采样"的优势（但先保证稳定训练）

## 🚀 建议的后续修复

### 修复2：添加 Gradient Clipping（推荐）

在 `engine/processor.py` 的训练循环中：

```python
# Line 78 之后
scaler.scale(loss).backward()

# 添加 gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # ← 添加这一行

scaler.step(optimizer)
scaler.update()
```

**效果**：
- ✅ 防止梯度爆炸
- ✅ 训练更稳定
- ✅ 可以重新尝试启用 Gumbel

### 修复3：降低学习率

如果仍然不稳定，可以：

```yaml
SOLVER:
  BASE_LR: 0.0002  # 从 0.00035 降低到 0.0002
```

### 修复4：Gumbel 温度退火

如果要重新启用 Gumbel，使用温度退火：

```python
# 训练初期用高温度（更平滑）
tau = max(0.5, 1.0 - epoch * 0.02)  # Epoch 1: tau=1.0, Epoch 25: tau=0.5

soft_mask = F.softmax((score + gumbel_noise) / tau, dim=1)
```

## 🎯 立即行动

### 1. 停止当前训练

```bash
Ctrl+C
```

### 2. 删除不稳定的 checkpoints

```bash
rm ../DeMo_1*.pth  # Epoch 10-19 的 checkpoints
```

或者从一个稳定的 checkpoint 继续（如果有）。

### 3. 重新开始训练（使用修复后的配置）

```bash
python train_net.py --config_file configs/RGBNT201/DeMo_SDTPS.yml
```

**修复后的配置**：
- ✅ WARMUP_ITERS: 3（学习率快速增长）
- ✅ SDTPS_USE_GUMBEL: False（避免数值不稳定）
- ⚠️ 建议添加 gradient clipping

## 📊 预期效果

禁用 Gumbel 后，应该看到：

```
Epoch 1: Loss=4.x→3.x, Acc=10-20%
Epoch 2: Loss=3.x→2.x, Acc=20-35%
Epoch 3: Loss=2.x→1.x, Acc=35-50%
...
不应该再有 Loss 爆炸
```

## 📝 后续优化

训练稳定后，可以尝试：

1. **重新启用 Gumbel + gradient clipping**
2. **调整 Gumbel 温度**（降低到 0.5）
3. **使用温度退火策略**

---

**提交**：commit `53f46d9`
**下一步**：停止训练，重新开始，观察是否稳定
