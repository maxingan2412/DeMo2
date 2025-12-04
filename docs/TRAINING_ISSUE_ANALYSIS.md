# SDTPS 训练问题分析

## 问题现象

```
Epoch 4:
  Loss: 4.529（非常高）
  Acc: 0.050（5%，几乎随机）

201个类别，随机猜测准确率 ≈ 0.5%
5% 说明模型几乎没有学到任何东西
```

## 🔍 关键问题：Attention 的 no_grad 阻断了学习路径

### 问题分析

我们在所有 attention 计算中使用了 `with torch.no_grad()`：

```python
def _compute_self_attention(self, patches, global_feat):
    with torch.no_grad():  # ← 关键问题！
        patches_norm = F.normalize(patches, dim=-1)
        global_norm = F.normalize(global_feat, dim=-1)
        return (patches_norm * global_norm).sum(dim=-1)
```

**这导致的梯度流动**：

```
Backbone → patches, global_feat
              ↓ (no_grad)
          attention scores
              ↓ (no_grad)
          综合 score
              ↓ (有梯度)
          MLP predictor
              ↓
          选择 + 聚合
              ↓
          loss
```

**问题**：
- ❌ Backbone 无法从 attention 获得梯度反馈
- ❌ Backbone 不知道应该提取什么样的特征才能让 attention 工作得更好
- ❌ 只有 MLP predictor 和 aggregation 有梯度，但它们依赖 Backbone 的特征质量

### 原论文 vs 我们的场景

| 特性 | 原 SEPS（图像-文本检索） | 我们 DeMo（重识别） |
|------|----------------------|------------------|
| Backbone | 预训练的 ViT，**冻结** | 预训练的 ViT，**需要 finetune** |
| 任务 | 检索（固定特征空间） | 分类（学习判别特征） |
| Attention 目的 | 找到与文本对应的 patch | 找到判别性 patch |
| 是否需要 Backbone 学习 | ❌ 不需要 | ✅ **需要** |

**关键差异**：
- 原 SEPS 的 Backbone 已经训练好，只需要"选择"正确的 patches
- 我们的 Backbone 需要学习提取判别性特征，**需要梯度指导**！

## 🎯 解决方案

### 方案1：移除 attention 计算中的 `no_grad`（推荐）

```python
def _compute_self_attention(self, patches, global_feat):
    # 移除 with torch.no_grad()
    patches_norm = F.normalize(patches, dim=-1)
    global_norm = F.normalize(global_feat, dim=-1)
    return (patches_norm * global_norm).sum(dim=-1)
```

**优点**：
- ✅ Backbone 能获得梯度，学习判别性特征
- ✅ 端到端训练
- ✅ 更适合重识别任务

**缺点**：
- ⚠️ 显存占用增加（需要保存中间梯度）
- ⚠️ 计算稍慢

### 方案2：只移除 self-attention 的 `no_grad`

```python
def _compute_self_attention(self, patches, global_feat):
    # 允许梯度
    patches_norm = F.normalize(patches, dim=-1)
    global_norm = F.normalize(global_feat, dim=-1)
    return (patches_norm * global_norm).sum(dim=-1)

def _compute_cross_attention(self, patches, cross_global):
    with torch.no_grad():  # 保留 cross-attention 的 no_grad
        ...
```

### 方案3：增加 warm-up 阶段

前几个 epoch 先训练 Backbone 和 直接拼接分支，再启用 SDTPS：

```python
if epoch < 5:
    # 只用 ori_score, ori 计算损失
    loss = loss_fn(ori_score, ori, target)
else:
    # 同时用 SDTPS 和 ori
    loss = loss_fn(sdtps_score, sdtps_feat, target) + loss_fn(ori_score, ori, target)
```

### 方案4：调整 sparse/aggr 比例

当前压缩太激进：128 → 25 (19.5%)

可以尝试：
```yaml
SDTPS_SPARSE_RATIO: 0.7  # 70%
SDTPS_AGGR_RATIO: 0.5    # 50%
# 最终：128 → 90 → 45 (35%)
```

### 方案5：检查是否真的在使用 SDTPS 特征

检查 processor.py line 47:
```python
test_sign = cfg.MODEL.HDM or cfg.MODEL.ATM  # ← 没有检查 USE_SDTPS！
```

虽然这只影响评估，但可能有其他地方也有类似问题。

## 🔧 立即尝试的修复

### 修复1：移除 no_grad（最重要）

在 `modeling/sdtps_complete.py` 中：

```python
def _compute_self_attention(self, patches, global_feat):
    if global_feat.dim() == 2:
        global_feat = global_feat.unsqueeze(1)

    # 移除 with torch.no_grad()
    patches_norm = F.normalize(patches, dim=-1)
    global_norm = F.normalize(global_feat, dim=-1)
    self_attn = (patches_norm * global_norm).sum(dim=-1)
    return self_attn
```

### 修复2：更新 test_sign

在 `engine/processor.py` line 47：

```python
test_sign = cfg.MODEL.HDM or cfg.MODEL.ATM or cfg.MODEL.USE_SDTPS
```

## 📊 对比实验建议

1. **对照组**：训练原始 DeMo（HDM+ATM），看收敛速度
2. **实验组1**：SDTPS + 移除 no_grad
3. **实验组2**：SDTPS + 更宽松的压缩比例
4. **实验组3**：SDTPS + warm-up

## 总结

**最可能的原因**：`with torch.no_grad()` 阻断了 Backbone 的学习

**立即修复**：移除 attention 计算中的 `no_grad`

**原因**：
- 原 SEPS 是检索任务，Backbone 冻结
- 我们是重识别任务，Backbone 需要 finetune
- No_grad 阻止了 Backbone 学习提取判别性特征
