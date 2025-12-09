# DeMo2 项目架构总览

**最后更新**: 2025-12-09

## 核心架构演进

### 原始 DeMo 架构
```
Backbone → HDM (解耦) → ATMoE (专家混合) → Classifier
```

### 当前 DeMo2 架构
```
Backbone → [SACR] → [LIF] → SDTPS (稀疏选择) → DGAF (自适应融合) → Classifier
         可选      可选           核心                  核心
```

---

## 1. SDTPS 模块（核心改进）

### 文件位置
- `modeling/sdtps.py` (主文件)
- `modeling/sdtps_complete.py` (完整版，与 sdtps.py 相同)
- `modeling/sdtps_fixed.py` (备份版本)

### 功能
多模态稀疏 token 选择，通过 **软 masking** 保持空间结构

### 关键特性

#### 1.1 CrossModalAttention
```python
class CrossModalAttention(nn.Module):
    """
    Q/K 反向的 Cross-Attention + 逐 head 余弦门控

    计算流程:
    1. 余弦相似度: cos_sim = cosine(patches, global)  # (B, N)
    2. Q/K 投影:
       Q = W_q @ global   # (B, 1, C) - 查询者
       K = W_k @ patches  # (B, N, C) - 被检索者
    3. Attention:
       attn = softmax(Q @ K^T / sqrt(d), dim=N)  # 在 patch 维度 softmax
    4. 逐 head 余弦门控:
       gate = sigmoid(cosine * scale[h] + bias[h])  # 每个 head 独立
       attn_gated = attn * gate
    5. 多头平均:
       score = mean(attn_gated, dim=heads)
    """
```

**参数：**
- Q/K 投影: `embed_dim × embed_dim × 2` ≈ 524K params
- 逐 head 门控: `num_heads × 2` = 8 params
- **总计**：每个 CrossModalAttention ≈ 524K params

#### 1.2 TokenSparse
```python
class TokenSparse(nn.Module):
    """
    简化的稀疏选择：直接置零未选中的 tokens

    - 移除了 MLP predictor
    - 移除了 token 提取和聚合
    - 得分公式: score = (s_im + s_m2 + s_m3) / 3
    - 输出: (B, N, C) 而非 (B, K+1, C)
    """
```

**优势：**
- ✅ 保持空间结构 `(B, N, C)`
- ✅ 无额外参数（除了 Cross-Attention）
- ✅ 兼容后续模块（DGAF V3）

#### 1.3 MultiModalSDTPS
```python
class MultiModalSDTPS(nn.Module):
    """
    为 RGB/NIR/TIR 三个模态分别进行 token selection

    每个模态使用 3 个 CrossModalAttention:
    - self_attn: 模态自身
    - cross_attn_m2: 与第二模态的交叉
    - cross_attn_m3: 与第三模态的交叉

    总共: 9 个 CrossModalAttention
    """
```

**参数统计：**
```
Total params: 4,727,880
  - Q/K projections: 4,727,808
  - Gate params (9×8): 72
Gate overhead: 0.0015%
```

### 输入输出
```python
# 输入
RGB_cash: (B, N, C) - backbone 的 patch features
RGB_global: (B, C) - backbone 的 global feature

# 输出
RGB_enhanced: (B, N, C) - soft masked tokens（未选中的置零）
rgb_mask: (B, N) - 二值 mask（1=选中，0=丢弃）
```

---

## 2. DGAF 模块

### 版本对比

| 版本 | 输入 | 输出 | 特点 |
|------|------|------|------|
| **V1** | `(B, C)` × 3 | `(B, 3C)` | 需要预先聚合 tokens |
| **V3** | `(B, N, C)` × 3 | `(B, 3C)` | 内置 attention pooling |

### 当前推荐：DGAF V3
```python
# V3 直接处理 SDTPS 输出的 tokens
dgaf_feat = self.dgaf(RGB_enhanced, NI_enhanced, TI_enhanced)
# (B, N, C) × 3 → (B, 3C)
```

**优势：**
- ✅ 直接使用 SDTPS 的 `(B, N, C)` 输出
- ✅ 内置 attention pooling，保留空间信息
- ✅ 不需要 GLOBAL_LOCAL 处理

---

## 3. GLOBAL_LOCAL 机制

### 作用范围
**仅在以下情况生效：**
1. DGAF V1 版本
2. 不使用 DGAF 的简单拼接

### 功能
```python
if self.GLOBAL_LOCAL:
    # 从 SDTPS 输出的 tokens 进行 pooling
    RGB_local = self.pool(RGB_enhanced.permute(0, 2, 1)).squeeze(-1)  # (B, C)

    # 与 backbone 的 global 融合降维
    RGB_sdtps = self.rgb_reduce(torch.cat([RGB_global, RGB_local], dim=-1))
    # (B, 2C) → (B, C)
else:
    # 默认：mean pooling
    RGB_sdtps = RGB_enhanced.mean(dim=1)  # (B, N, C) → (B, C)
```

### 当前状态
- **DGAF V3 配置**：`GLOBAL_LOCAL: False`（不需要）
- **DGAF V1 配置**：`GLOBAL_LOCAL: True`（可选）

---

## 4. 完整数据流

### 训练流程 (SDTPS + DGAF V3)
```
输入图像 (B, 3, 256, 128)
    ↓
Backbone (ViT-B-16)
    ↓
RGB_cash: (B, 128, 512)  # patch features
RGB_global: (B, 512)     # global feature
    ↓
[可选: SACR/MultiModalSACR]
    ↓
[可选: LIF quality-aware weighting]
    ↓
SDTPS (MultiModalSDTPS)
    ├─ 计算余弦相似度 (B, N)
    ├─ CrossModalAttention (带逐 head 门控)
    ├─ 综合得分 → Top-K mask
    └─ 输出: RGB_enhanced (B, 128, 512) - soft masked
    ↓
DGAF V3
    ├─ 输入: 3 × (B, N, C) tokens
    ├─ Multi-head attention pooling
    └─ 输出: (B, 3C=1536)
    ↓
Bottleneck + Classifier
    ↓
Loss: ID Loss + Triplet Loss
```

### 推理流程
```
与训练相同，但:
- 支持缺失模态模拟 (TEST.MISS)
- 可选择返回模式:
  * return_pattern=1: 仅原始特征
  * return_pattern=2: 仅 SDTPS+DGAF 特征
  * return_pattern=3: 两者拼接 (默认)
```

---

## 5. 配置参数详解

### SDTPS 核心参数

| 参数 | 类型 | 推荐值 | 说明 |
|------|------|--------|------|
| `USE_SDTPS` | bool | `True` | 启用 SDTPS |
| `SDTPS_SPARSE_RATIO` | float | `0.6-0.7` | token 保留比例 |
| `SDTPS_CROSS_ATTN_TYPE` | str | `'attention'` | `'attention'` 或 `'cosine'` |
| `SDTPS_CROSS_ATTN_HEADS` | int | `4` | Cross-Attention 头数 |
| `SDTPS_USE_GUMBEL` | bool | `False` | Gumbel-Softmax（不稳定，不推荐） |
| `SDTPS_GUMBEL_TAU` | float | `5.0` | Gumbel 温度 |
| `SDTPS_LOSS_WEIGHT` | float | `2.0` | SDTPS 分支损失权重 |

### 已废弃参数（保留兼容性）

| 参数 | 原用途 | 当前状态 |
|------|--------|----------|
| `SDTPS_BETA` | MLP predictor 权重 | 已移除 MLP，不使用 |
| `SDTPS_AGGR_RATIO` | TokenAggregation 比例 | 已移除聚合层，不使用 |

### DGAF 参数

| 参数 | 类型 | 推荐值 | 说明 |
|------|------|--------|------|
| `USE_DGAF` | bool | `True` | 启用 DGAF |
| `DGAF_VERSION` | str | `'v3'` | `'v1'` 或 `'v3'` |
| `DGAF_NUM_HEADS` | int | `8` | V3 的注意力头数 |
| `DGAF_TAU` | float | `1.0` | 门控温度 |
| `DGAF_INIT_ALPHA` | float | `0.5` | 初始融合权重 |

### GLOBAL_LOCAL 参数

| 配置 | `GLOBAL_LOCAL` 设置 |
|------|-------------------|
| SDTPS + DGAF V3 | `False`（不需要） |
| SDTPS + DGAF V1 | `True`（可选） |
| SDTPS 单独使用 | `False`（默认 mean） |

---

## 6. 配置文件组织

### 按数据集分类

#### RGBNT201 (行人重识别)
- `DeMo.yml` - 原始 DeMo (HDM+ATM)
- `DeMo_SDTPS.yml` - 仅 SDTPS
- `DeMo_SDTPS_DGAF_ablation.yml` - **SDTPS + DGAF V3 (推荐)**
- `DeMo_SDTPS_DGAF_ablation_test.yml` - 测试配置（小 batch）
- 其他：LIF, SACR 组合配置

#### RGBNT100 (车辆重识别)
- `DeMo.yml` - 原始 DeMo
- `DeMo_SDTPS_DGAF_ablation.yml` - **SDTPS + DGAF V3**

#### MSVR310 (车辆重识别)
- `DeMo.yml` - 原始 DeMo
- `DeMo_SDTPS_DGAF_ablation.yml` - **SDTPS + DGAF V3**

### 推荐配置

**最佳性能：**
```bash
python train_net.py --config_file configs/RGBNT201/DeMo_SDTPS_DGAF_ablation.yml
```

**快速测试：**
```bash
python train_net.py --config_file configs/RGBNT201/DeMo_SDTPS_DGAF_ablation_test.yml
```

---

## 7. 模块参数对比

### SDTPS vs 原始 HDM+ATM

| 模块 | 参数量 | 输入 | 输出 | 特点 |
|------|--------|------|------|------|
| **HDM** | ~1M | `(B,N,C)×3` | 7 个解耦特征 | 复杂，参数多 |
| **ATMoE** | ~500K | 7 个特征 | `(B,C)` | 基于注意力的门控 |
| **SDTPS** | ~4.7M | `(B,N,C)×3` | `(B,N,C)×3` | 稀疏选择，保持空间 |
| **DGAF V3** | ~3.1M | `(B,N,C)×3` | `(B,3C)` | 双门控融合 |

### 关键区别
- **HDM+ATM**: 解耦 → 门控混合 → 聚合
- **SDTPS+DGAF**: 稀疏选择 → 自适应融合 → 聚合

---

## 8. 训练命令

### 基础训练
```bash
# RGBNT201 (行人)
python train_net.py --config_file configs/RGBNT201/DeMo_SDTPS_DGAF_ablation.yml

# RGBNT100 (车辆)
python train_net.py --config_file configs/RGBNT100/DeMo_SDTPS_DGAF_ablation.yml

# MSVR310 (车辆)
python train_net.py --config_file configs/MSVR310/DeMo_SDTPS_DGAF_ablation.yml
```

### 命令行覆盖参数
```bash
# 修改稀疏比例
python train_net.py --config_file configs/RGBNT201/DeMo_SDTPS_DGAF_ablation.yml \
    MODEL.SDTPS_SPARSE_RATIO 0.8

# 使用余弦相似度而非 Cross-Attention
python train_net.py --config_file configs/RGBNT201/DeMo_SDTPS_DGAF_ablation.yml \
    MODEL.SDTPS_CROSS_ATTN_TYPE cosine

# 启用 GLOBAL_LOCAL（如果使用 DGAF V1）
python train_net.py --config_file configs/RGBNT201/DeMo_SDTPS_DGAF_ablation.yml \
    MODEL.DGAF_VERSION v1 MODEL.GLOBAL_LOCAL True
```

### 消融实验
```bash
# 仅 SDTPS，不使用 DGAF
python train_net.py --config_file configs/RGBNT201/DeMo_SDTPS.yml

# 仅 DGAF，不使用 SDTPS
python train_net.py --config_file configs/RGBNT201/DeMo_DGAF.yml

# SDTPS + DGAF（完整版）
python train_net.py --config_file configs/RGBNT201/DeMo_SDTPS_DGAF_ablation.yml
```

---

## 9. 测试命令

### 基础测试
```bash
# 修改 test_net.py:43 指定模型路径
python test_net.py --config_file configs/RGBNT201/DeMo_SDTPS_DGAF_ablation.yml
```

### 缺失模态测试
```bash
# 缺失 RGB
python test_net.py --config_file configs/RGBNT201/DeMo_SDTPS_DGAF_ablation.yml \
    TEST.MISS r

# 缺失 NIR
python test_net.py --config_file configs/RGBNT201/DeMo_SDTPS_DGAF_ablation.yml \
    TEST.MISS n

# 缺失 RGB+NIR
python test_net.py --config_file configs/RGBNT201/DeMo_SDTPS_DGAF_ablation.yml \
    TEST.MISS rn
```

---

## 10. 关键代码位置

### 模型定义
- `modeling/make_model.py:220-330` - 训练 forward
- `modeling/make_model.py:390-520` - 推理 forward

### SDTPS 集成
```python
# modeling/make_model.py:278-329
if self.USE_SDTPS:
    RGB_enhanced, NI_enhanced, TI_enhanced, ... = self.sdtps(...)

    if self.USE_DGAF:
        if self.DGAF_VERSION == 'v3':
            # 直接使用 tokens
            sdtps_feat = self.dgaf(RGB_enhanced, NI_enhanced, TI_enhanced)
        else:
            # V1: 需要聚合
            if self.GLOBAL_LOCAL:
                # pool + global 融合
            else:
                # mean pooling
            sdtps_feat = self.dgaf(RGB_sdtps, NI_sdtps, TI_sdtps)
```

### GLOBAL_LOCAL 集成
```python
# modeling/make_model.py:292-303 (训练)
# modeling/make_model.py:480-495 (推理)
if self.GLOBAL_LOCAL:
    RGB_local = self.pool(RGB_enhanced.permute(0, 2, 1)).squeeze(-1)
    RGB_sdtps = self.rgb_reduce(torch.cat([RGB_global, RGB_local], dim=-1))
```

---

## 11. 常见问题

### Q1: GLOBAL_LOCAL 何时启用？
**A:** 仅在 DGAF V1 或不使用 DGAF 时有用。DGAF V3 直接用 tokens，设置为 `False`。

### Q2: SDTPS_BETA 和 SDTPS_AGGR_RATIO 还有用吗？
**A:** 已废弃。移除了 MLP predictor 和 TokenAggregation 层。保留参数仅为兼容性。

### Q3: 推荐使用 'attention' 还是 'cosine'？
**A:** 推荐 `'attention'`。Cross-Attention 有可学习参数，表达能力更强，逐 head 门控只增加 72 个参数（0.0015%）。

### Q4: sparse_ratio 如何选择？
**A:**
- 0.6: 保留 60% tokens，更激进的稀疏化
- 0.7: 保留 70% tokens，**推荐**（平衡性能和效率）
- 0.8: 保留 80% tokens，更保守

---

## 12. 性能优化建议

### 内存优化
```yaml
DATALOADER:
  NUM_INSTANCE: 2-4  # 减少 instance 数量
SOLVER:
  IMS_PER_BATCH: 4-8  # 减少 batch size
```

### 训练速度优化
```yaml
MODEL:
  SDTPS_SPARSE_RATIO: 0.6  # 更激进的稀疏化
  DGAF_NUM_HEADS: 4  # 减少 DGAF 的头数
DATALOADER:
  NUM_WORKERS: 8-14  # 根据 CPU 核心数调整
```

---

## 13. 最近的代码改动

### 2025-12-09 改动总结

1. **SDTPS 简化** (`21f61b8`)
   - 移除 MLP predictor
   - 移除 token extraction
   - 输出改为 `(B, N, C)` 保持空间结构

2. **Cross-Attention 增强** (`9a18c62`, `c2549be`, `019ece5`)
   - Q/K 反向：global 查询 patches
   - 逐 head 余弦门控：scale 和 bias
   - 门控注意力：`attn * sigmoid(cosine * scale + bias)`

3. **GLOBAL_LOCAL 修正** (`695db1b`)
   - 移到 SDTPS 之后
   - 仅对 DGAF V1 生效
   - 使用 backbone 的 RGB_global + pooled RGB_enhanced

4. **配置文件更新** (`9543901`)
   - 添加 `SDTPS_CROSS_ATTN_TYPE` 和 `SDTPS_CROSS_ATTN_HEADS`
   - 更新 `GLOBAL_LOCAL` 设置
   - 标注废弃参数

---

## 14. Git 提交历史

```
019ece5 - Implement per-head cosine gating with affine transformation
c2549be - Refactor gate mechanism: element-wise gating with lightweight MLP
af84e4b - Sync all SDTPS variants and add backward compatibility
9a18c62 - Add Cross-Attention with cosine similarity gating to SDTPS
21f61b8 - Simplify SDTPS to preserve spatial structure with soft masking
695db1b - Fix GLOBAL_LOCAL logic: only apply to DGAF V1/non-DGAF aggregation
9543901 - Update all SDTPS config files for new architecture (当前)
```

---

## 15. 下一步工作

### 建议实验
1. **Baseline**: SDTPS + DGAF V3 (当前配置)
2. **消融 1**: 仅 SDTPS (`USE_DGAF: False`)
3. **消融 2**: SDTPS (cosine) vs SDTPS (attention)
4. **消融 3**: 不同 sparse_ratio (0.5, 0.6, 0.7, 0.8)
5. **完整版**: SDTPS + DGAF + SACR + LIF

### 配置文件对应
```bash
# Baseline
configs/RGBNT201/DeMo_SDTPS_DGAF_ablation.yml

# 消融 1
configs/RGBNT201/DeMo_SDTPS.yml

# 完整版（需要手动配置）
# USE_SACR: True, USE_LIF: True
```

---

**注意**: 本文档基于 2025-12-09 的代码状态生成。如有进一步改动，请更新此文档。
