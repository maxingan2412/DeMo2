# 余弦门控机制改进说明

## 专业评审反馈与改进

### 评审要点

1. **归一化问题**：余弦 ∈ [-1, 1] 未归一化直接进 sigmoid，可能导致 head 间尺度漂移
2. **概率性问题**：attn 已 softmax（和为 1），乘以 gate 后不再归一化（和 < 1）
3. **初始化问题**：可能导致训练初期过度稀疏

---

## 改进实现

### 1. 可选的 LayerNorm（稳定逐 head 尺度）

```python
# 参数
use_gate_norm: bool = False  # 默认关闭

# 实现
if self.use_gate_norm:
    gate_logits = gate_logits.permute(0, 2, 1)  # (B, num_heads, N) → (B, N, num_heads)
    gate_logits = self.gate_norm(gate_logits)   # LayerNorm on num_heads 维度
    gate_logits = gate_logits.permute(0, 2, 1)  # → (B, num_heads, N)
```

**何时使用：**
- ✅ 当 N（patch 数量）可变时
- ✅ 当担心 head 间尺度漂移时
- ❌ 固定 N 且追求简洁时

**效果：**
- 稳定逐 head 的门控分布
- 防止某些 head 的 gate 值过高/过低

---

### 2. 可选的注意力重新归一化（保持概率性）

```python
# 参数
renormalize_attn: bool = False  # 默认关闭

# 实现
if self.renormalize_attn:
    # 在 N 维度重新归一化，确保每个 head 的权重和为 1
    attn_gated = attn_gated / (attn_gated.sum(dim=-1, keepdim=True) + 1e-8)
```

**数学分析：**

```
原始 attn: softmax(Q @ K^T) → sum(attn, dim=N) = 1 ✓
门控后: attn_gated = attn * gate
        sum(attn_gated) = sum(attn * gate) ≠ 1 (通常 < 1)
重新归一化: attn_gated / sum(attn_gated) → sum = 1 ✓
```

**何时使用：**
- ✅ 当后续模块假设"注意力和为 1"时
- ✅ 需要严格的概率解释时
- ❌ 对于 TokenSparse 的阈值/排序用法（不需要）

**当前场景分析：**
```python
# TokenSparse 的使用方式
score = (s_im + s_m2 + s_m3) / 3  # 平均三个分数
score_sorted, indices = torch.sort(score, descending=True)
keep_policy = indices[:, :num_keep]  # Top-K 选择
```
→ **不需要重新归一化**（基于排序，非概率）

---

### 3. 改进的初始化策略（避免过稀疏）

#### 旧初始化
```python
gate_scale = 1.0
gate_bias = 0.0

# 效果（cosine ∈ [-1, 1]）
gate = sigmoid(1.0 * cosine + 0.0)
     = sigmoid(cosine)
     ∈ [sigmoid(-1), sigmoid(1)]
     ≈ [0.27, 0.73]
```

**问题：**
- ❌ 当 cosine = -1 时，gate = 0.27 → 过度抑制
- ❌ 可能导致初期选中的 tokens 太少

#### 新初始化（改进）
```python
gate_scale = 0.5  # 较小，曲线更平缓
gate_bias = 0.5   # 正值，整体提升

# 效果
gate = sigmoid(0.5 * cosine + 0.5)
     ∈ [sigmoid(0.5*(-1)+0.5), sigmoid(0.5*1+0.5)]
     = [sigmoid(0), sigmoid(1)]
     ≈ [0.50, 0.73]
```

**改进：**
- ✅ 最小门控值从 0.27 → 0.50（更保守）
- ✅ 曲线更平缓（scale=0.5 vs 1.0）
- ✅ 避免初期过度稀疏

#### 可视化对比

```
Cosine    Old gate    New gate    Difference
─────────────────────────────────────────────
 -1.0      0.27        0.50        +0.23 ↑
 -0.5      0.38        0.56        +0.18 ↑
  0.0      0.50        0.62        +0.12 ↑
 +0.5      0.62        0.69        +0.07 ↑
 +1.0      0.73        0.73        +0.00 →
```

**结论：**
- 负余弦值（不相关的 patch）门控值显著提升
- 正余弦值（相关的 patch）基本不变
- 整体效果：**更保守，避免过度抑制**

---

## 推荐配置

### 配置 1：默认（推荐）
```python
CrossModalAttention(
    embed_dim=512,
    num_heads=4,
    use_gate_norm=False,      # 关闭 LayerNorm（简洁）
    renormalize_attn=False,   # 关闭重新归一化（基于排序）
)
```

**适用场景：**
- ✅ 固定图像尺寸（N 不变）
- ✅ TokenSparse 的阈值选择
- ✅ 追求简洁和效率

**初始化：**
- gate_scale = 0.5
- gate_bias = 0.5
- 初始门控值：[0.50, 0.73]

---

### 配置 2：稳定版（多尺度任务）
```python
CrossModalAttention(
    embed_dim=512,
    num_heads=4,
    use_gate_norm=True,       # 启用 LayerNorm
    renormalize_attn=False,
)
```

**适用场景：**
- ✅ 多尺度输入（N 可变）
- ✅ 需要稳定的逐 head 分布
- ⚠️ 增加少量计算开销

---

### 配置 3：严格概率（特殊需求）
```python
CrossModalAttention(
    embed_dim=512,
    num_heads=4,
    use_gate_norm=False,
    renormalize_attn=True,    # 启用重新归一化
)
```

**适用场景：**
- ✅ 后续模块假设"注意力和为 1"
- ✅ 需要严格概率解释
- ❌ **当前 TokenSparse 不需要**

---

## 实验建议

### 消融实验：初始化策略

```bash
# 旧初始化 (scale=1.0, bias=0.0)
python train_net.py --config_file configs/RGBNT201/DeMo_SDTPS_DGAF_ablation.yml \
    --init_scale 1.0 --init_bias 0.0

# 新初始化 (scale=0.5, bias=0.5) - 当前默认
python train_net.py --config_file configs/RGBNT201/DeMo_SDTPS_DGAF_ablation.yml

# 更保守 (scale=0.3, bias=0.7)
python train_net.py --config_file configs/RGBNT201/DeMo_SDTPS_DGAF_ablation.yml \
    --init_scale 0.3 --init_bias 0.7
```

### 消融实验：归一化选项

```bash
# 默认（无额外归一化）
python train_net.py --config_file configs/RGBNT201/DeMo_SDTPS_DGAF_ablation.yml

# 启用 LayerNorm
# 需要修改代码，将 use_gate_norm=True 传入

# 启用重新归一化
# 需要修改代码，将 renormalize_attn=True 传入
```

---

## 数值稳定性分析

### 梯度流

```
cosine_sim (可梯度)
    ↓
gate_logits = cosine * scale + bias
    ↓
[可选] LayerNorm(gate_logits)
    ↓
gate = sigmoid(gate_logits)
    ↓
attn_gated = attn * gate
    ↓
[可选] renormalize(attn_gated)
    ↓
score = mean(attn_gated, dim=heads)
    ↓
Loss
```

**梯度路径：**
- ✅ cosine_sim 可梯度，能反向调节 patches/global
- ✅ scale/bias 可学习
- ✅ gate 通过 sigmoid 有梯度
- ✅ 完整梯度流畅通

---

## 总结

### 当前实现的科学性

✅ **功能正确**
- 余弦相似度 → 可学习变换 → 门控注意力
- 逐 head 自适应
- 逐元素点乘（非线性交互）

✅ **数值稳定**
- 可选 LayerNorm 防止尺度漂移
- 可选重新归一化保持概率性
- 改进的初始化避免过稀疏

✅ **梯度友好**
- 完整的反向传播路径
- Sigmoid 平滑可导
- 无数值爆炸风险

### 专业评审结论

**"实现是科学的，功能上符合'余弦门控注意力'"**

**建议：**
1. ✅ 默认配置（无 LayerNorm，无 renorm）适合当前任务
2. ⚠️ 若训练初期过稀疏，已调整初始化策略
3. ✅ 广播逻辑和梯度流都正确

---

## 文件更新

- `modeling/sdtps.py` - 主文件（已更新）
- `modeling/sdtps_complete.py` - 同步
- `modeling/sdtps_fixed.py` - 同步
- `ARCHITECTURE_SUMMARY.txt` - 架构摘要
- `docs/PROJECT_OVERVIEW.md` - 详细文档

---

**最后更新**: 2025-12-09  
**Commit**: 83bc831
