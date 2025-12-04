# SDTPS 训练问题修复

## 问题现象

```
Epoch 4: Loss=4.529, Acc=5%
→ 几乎没有学习（201类随机猜测 ≈ 0.5%）
```

## 🎯 根本原因分析

### 原因1：`with torch.no_grad()` 阻断了 Backbone 学习 ⚠️ **最关键**

**问题**：
我们在所有 attention 计算中使用了 `with torch.no_grad()`，这在原 SEPS 论文中是合理的，但在我们的场景中有问题。

**场景对比**：

| 特性 | 原 SEPS | 我们的 DeMo |
|------|---------|------------|
| 任务 | 图像-文本检索 | 多模态重识别 |
| Backbone 训练 | **冻结**（不需要更新） | **Finetune**（需要学习） |
| 特征空间 | 预训练好的固定空间 | 需要学习判别性特征 |
| Attention 作用 | 选择已有的好特征 | 指导 Backbone 提取判别特征 |

**梯度流动被阻断**：

```
Loss → SDTPS特征 → Aggregation (有梯度✅) → TokenSparse (有梯度✅)
                                                  ↓
                                            Attention (NO GRAD ❌)
                                                  ↓
                                            Backbone (收不到梯度❌)
```

**结果**：
- Backbone 不知道应该提取什么样的特征
- SDTPS 只能基于 Backbone 现有的（可能不好的）特征工作
- 形成恶性循环

### 原因2：`test_sign` 未检查 `USE_SDTPS`

**问题代码**（processor.py line 47）：
```python
test_sign = cfg.MODEL.HDM or cfg.MODEL.ATM  # ← 缺少 USE_SDTPS
```

虽然这只影响评估，但应该修复。

## ✅ 已应用的修复

### 修复1：移除 `no_grad`（核心修复）

**文件**：`modeling/sdtps_complete.py`

**修改前**：
```python
def _compute_self_attention(self, patches, global_feat):
    with torch.no_grad():  # ← 阻断梯度
        patches_norm = F.normalize(patches, dim=-1)
        ...
```

**修改后**：
```python
def _compute_self_attention(self, patches, global_feat):
    # 移除 no_grad 以允许梯度传播到 Backbone
    patches_norm = F.normalize(patches, dim=-1)
    global_norm = F.normalize(global_feat, dim=-1)
    self_attn = (patches_norm * global_norm).sum(dim=-1)
    return self_attn
```

**效果**：
- ✅ Backbone 现在能从 attention 获得梯度
- ✅ 端到端学习判别性特征
- ⚠️ 显存占用会增加（保存中间梯度）

### 修复2：更新 `test_sign`

**文件**：`engine/processor.py`

**修改前**：
```python
test_sign = cfg.MODEL.HDM or cfg.MODEL.ATM
```

**修改后**：
```python
test_sign = cfg.MODEL.HDM or cfg.MODEL.ATM or cfg.MODEL.USE_SDTPS
```

## 🚀 重新开始训练

### 方式1：从头训练（推荐）

```bash
# 删除旧的checkpoints
rm ../DeMo_*.pth

# 重新开始训练
python train_net.py --config_file configs/RGBNT201/DeMo_SDTPS.yml
```

### 方式2：从 Epoch 4 继续（如果想快速验证）

```bash
# 使用已有的 checkpoint
python train_net.py --config_file configs/RGBNT201/DeMo_SDTPS.yml \
    TEST.WEIGHT ../DeMo_4.pth
```

但建议**从头训练**，因为前4个epoch的学习可能是错误的。

## 📊 预期改善

修复后应该看到：

### Epoch 1-2
```
Loss: 6.x → 5.x → 4.x
Acc: 1% → 5% → 15%
```

### Epoch 5-10
```
Loss: 4.x → 3.x → 2.x
Acc: 15% → 30% → 50%
```

### Epoch 20+
```
Loss: < 2.0
Acc: > 60%
```

如果修复后仍然不收敛，考虑：

## 🔧 备选修复方案

### 方案A：减少压缩比例

**当前**：128 → 64 → 25 (19.5%) - 可能太激进

**建议**：
```yaml
SDTPS_SPARSE_RATIO: 0.7  # 70%
SDTPS_AGGR_RATIO: 0.5    # 50%
# 最终：128 → 90 → 45 (35%)
```

### 方案B：先只用 SDTPS 特征，不用原始特征

修改 `make_model.py`：
```python
if self.USE_SDTPS:
    # 只返回 SDTPS 分支
    return sdtps_score, sdtps_feat
```

减少分支数量，聚焦训练 SDTPS。

### 方案C：Warm-up SDTPS

前几个 epoch 先训练原始分支，再逐步启用 SDTPS。

### 方案D：检查学习率

为 SDTPS 模块设置更大的学习率：

```python
# 在 solver/make_optimizer.py 中
sdtps_params = []
other_params = []

for name, param in model.named_parameters():
    if 'sdtps' in name:
        sdtps_params.append(param)
    else:
        other_params.append(param)

optimizer = torch.optim.Adam([
    {'params': other_params, 'lr': cfg.SOLVER.BASE_LR},
    {'params': sdtps_params, 'lr': cfg.SOLVER.BASE_LR * 10},  # SDTPS 用10倍学习率
])
```

## 📝 监控指标

重新训练时关注：

1. **Loss 下降速度**：应该在 Epoch 1-5 快速下降
2. **Acc 增长速度**：应该在 Epoch 1-5 快速增长到 20%+
3. **梯度范数**：检查 SDTPS 和 Backbone 的梯度是否正常
4. **特征质量**：可视化 SDTPS 选择的 patches

## ✅ 立即行动

1. **停止当前训练**
2. **删除旧 checkpoints**
3. **重新训练**（修复已应用）
4. **监控前5个epoch**
5. **如果仍不收敛**，尝试备选方案

---

**修复时间**：2025-12-04 23:55
**关键修复**：移除 `with torch.no_grad()`
**预期效果**：Loss 快速下降，Acc 快速上升
