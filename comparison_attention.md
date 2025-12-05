# Attention 计算对比

## 原版（seps_modules_reviewed_v2_enhanced.py）

```python
with torch.no_grad():  # ✅ 关键：不计算梯度！
    img_spatial_glo_norm = F.normalize(
        img_spatial_embs.mean(dim=1, keepdim=True), dim=-1
    )
    img_spatial_self_attention = (
        img_spatial_glo_norm * img_spatial_embs_norm
    ).sum(dim=-1)
```

## 我的版本（modeling/sdtps.py）

```python
# ❌ 缺少 with torch.no_grad()
patches_norm = F.normalize(patches, dim=-1)
global_norm = F.normalize(global_feat, dim=-1)
self_attn = (patches_norm * global_norm).sum(dim=-1)
```

## 对比表

| 特性 | 原版 | 我的版本 | 一致性 |
|------|------|----------|--------|
| L2 归一化 | ✅ | ✅ | ✅ 一致 |
| 点积相似度 | ✅ | ✅ | ✅ 一致 |
| `with torch.no_grad()` | ✅ 有 | ❌ 没有 | ❌ **不一致** |
| 可学习参数 | ❌ 无 | ❌ 无 | ✅ 一致 |

## 影响分析

### ✅ 没有可学习参数
- 原版和我的版本都**没有可学习参数**
- 只是简单的归一化 + 点积运算
- 这一点是一致的 ✅

### ❌ 缺少 `no_grad` 的影响
1. **显存占用增加**：会保存中间梯度
2. **计算速度变慢**：需要计算额外的梯度
3. **可能干扰训练**：attention score 的梯度会传播回特征

## 结论

**核心逻辑一致，但缺少 `no_grad` 优化**

修复方法：在两个函数中添加 `with torch.no_grad():`
