# SDTPS 形状一致性说明

## ✅ 当前状态：形状保证一致

### 保证机制

三个模态 `RGB_enhanced`, `NI_enhanced`, `TI_enhanced` 的形状完全一致，因为：

1. **相同的 Backbone**：所有模态使用 `self.BACKBONE`
2. **相同的输入尺寸**：`(B, 3, 256, 128)`
3. **相同的 patch 数量**：N = 128
4. **相同的 sparse_ratio**：在 `MultiModalSDTPS.__init__` 中只定义了一个 `self.sparse_ratio`
5. **确定性的 K 值**：`K = ceil(N × sparse_ratio)` 对所有模态相同

### 验证结果

```
输入 patch 数量: N = 128
sparse_ratio: 0.6
选中数量: K = ceil(128 × 0.6) = 77
输出形状: (B, 78, 512)  # 77 个选中 + 1 个 extra token

RGB_enhanced: (4, 78, 512) ✅
NI_enhanced:  (4, 78, 512) ✅
TI_enhanced:  (4, 78, 512) ✅
```

## ⚠️ 潜在风险场景（未来修改需注意）

### 场景 1：不同模态使用不同的 sparse_ratio

如果未来想为不同模态设置不同的选择比例：

```python
# ❌ 可能导致形状不一致
self.rgb_sparse = TokenSparse(sparse_ratio=0.6)
self.nir_sparse = TokenSparse(sparse_ratio=0.7)  # 不同的比例
self.tir_sparse = TokenSparse(sparse_ratio=0.5)
```

后果：
- RGB: K = ceil(128 × 0.6) = 77 → 输出 (B, 78, C)
- NIR: K = ceil(128 × 0.7) = 90 → 输出 (B, 91, C)
- TIR: K = ceil(128 × 0.5) = 64 → 输出 (B, 65, C)
- ❌ **形状不一致，无法拼接！**

### 场景 2：不同模态使用不同的 Backbone

```python
# ❌ 可能导致 patch 数量不同
RGB_cash = self.rgb_backbone(RGB)    # 输出 128 patches
NI_cash = self.nir_backbone(NI)      # 输出 196 patches（不同配置）
TI_cash = self.tir_backbone(TI)      # 输出 128 patches
```

后果：
- 即使 sparse_ratio 相同，K 值也会不同

### 场景 3：不同的输入图像尺寸

```python
# ❌ 不同尺寸导致不同的 patch 数量
RGB = torch.randn(B, 3, 256, 128)   # patch 数量 = (256/16) × (128/16) = 128
NI = torch.randn(B, 3, 224, 224)    # patch 数量 = (224/16) × (224/16) = 196
TI = torch.randn(B, 3, 256, 128)    # patch 数量 = 128
```

## 🛡️ 安全建议

### 建议 1：保持当前设计（推荐）

**最安全的做法**：保持所有模态使用：
- 相同的 Backbone
- 相同的输入尺寸
- 相同的 sparse_ratio

### 建议 2：如果需要不同的 sparse_ratio

方案 A：**强制对齐输出数量**

```python
# 在 MultiModalSDTPS.forward() 中
# 计算最小的 K 值
K_rgb = rgb_select.shape[1]
K_nir = nir_select.shape[1]
K_tir = tir_select.shape[1]
K_min = min(K_rgb, K_nir, K_tir)

# 截断到相同长度
RGB_enhanced = torch.cat([rgb_select[:, :K_min], rgb_extra], dim=1)
NI_enhanced = torch.cat([nir_select[:, :K_min], nir_extra], dim=1)
TI_enhanced = torch.cat([tir_select[:, :K_min], tir_extra], dim=1)
```

方案 B：**使用自适应池化**

```python
# 将不同数量的 tokens 池化到固定数量
def adaptive_token_pool(tokens, target_num):
    """将 (B, N, C) 池化到 (B, target_num, C)"""
    B, N, C = tokens.shape
    # 使用插值或学习的聚合
    return F.adaptive_avg_pool1d(
        tokens.transpose(1, 2),
        target_num
    ).transpose(1, 2)

RGB_enhanced = adaptive_token_pool(rgb_enhanced, target_num=64)
NI_enhanced = adaptive_token_pool(nir_enhanced, target_num=64)
TI_enhanced = adaptive_token_pool(tir_enhanced, target_num=64)
```

方案 C：**分别处理，不拼接**

```python
# 不要求形状一致，分别处理
RGB_feat = RGB_enhanced.mean(dim=1)  # (B, C)
NI_feat = NI_enhanced.mean(dim=1)    # (B, C)
TI_feat = TI_enhanced.mean(dim=1)    # (B, C)

# 拼接全局特征
final_feat = torch.cat([RGB_feat, NI_feat, TI_feat], dim=-1)  # (B, 3C)
```

## 📝 当前实现的安全性

**当前代码是完全安全的**，因为：

1. ✅ 硬编码使用相同的 `self.sparse_ratio`
2. ✅ 三个 TokenSparse 模块共享相同的配置
3. ✅ 所有模态通过同一个 Backbone
4. ✅ 输入尺寸由配置文件统一控制

只要不修改这些核心设计，形状就**永远一致**！

## 🔍 快速验证方法

运行验证脚本：
```bash
python verify_shape_consistency.py
```

该脚本会：
1. 检查三个模态的 patch 数量
2. 验证选中的 token 数量
3. 确认输出形状一致性
4. 输出详细的形状信息

## 总结

**Q: RGB_enhanced, NI_enhanced, TI_enhanced 形状是否一致？**
**A: 是的，完全一致！** ✅

原因：当前设计天然保证了形状一致性，无需额外的对齐机制。
