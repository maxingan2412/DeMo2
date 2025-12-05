# SDTPS 影响不大的问题分析与修复

## 问题现象

**用户观察**：
> "打开或者关闭 USE_SDTPS 的时候实验的结果没有太大的变化"

## 🔍 根本原因分析

### 原因1：**SDTPS 参数量只占 0.41%** ⚠️⚠️⚠️

**参数量统计**：
```
总参数：88.72M
  ├─ Backbone: 86.15M (97.1%)  ← 主体
  ├─ SDTPS: 0.37M (0.41%)      ← 太小了！
  ├─ 分类器 × 2: 0.62M (0.7%)
  └─ 其他: 1.58M (1.8%)

SDTPS 详细：
  ├─ rgb_sparse (MLP): 0.066M
  ├─ rgb_aggr (Aggregation): 0.056M
  ├─ nir_sparse: 0.066M
  ├─ nir_aggr: 0.056M
  ├─ tir_sparse: 0.066M
  └─ tir_aggr: 0.056M
  总计: 0.365M
```

**关键结论**：
- ❗ SDTPS 只占总参数的 **0.41%**
- ❗ 即使 SDTPS 完全不工作，模型还有 **99.59%** 的参数在学习
- ❗ 这就解释了为什么开关 SDTPS 影响不大！

**对比 HDM+ATM**：
- HDM+ATM 的参数量应该更大（有7个expert的权重）
- 所以开关 HDM+ATM 影响可能更明显

### 原因2：**两个分支损失权重相等** ⚠️⚠️

**当前损失计算**（processor.py）：
```python
output = [sdtps_score, sdtps_feat, ori_score, ori]

loss = loss_fn(sdtps_score, sdtps_feat)  # 权重 = 1.0
     + loss_fn(ori_score, ori)           # 权重 = 1.0

# 相当于：
loss = 1.0 × loss_sdtps + 1.0 × loss_ori
```

**问题**：
- **原始分支**：使用完整的 patches（128个），信息丰富
- **SDTPS分支**：压缩到 25 patches，信息损失
- **权重相等**：模型会优先学习信息更丰富的原始分支

**结果**：
- 模型主要优化原始分支
- SDTPS 分支贡献被稀释
- 开关 SDTPS 影响不大

### 原因3：**原始分支信息更完整**

**特征对比**：

| 分支 | 特征来源 | Patches 数量 | 信息完整度 |
|------|---------|------------|----------|
| 原始 | RGB_global + NI_global + TI_global | 128 (全部) | 100% |
| SDTPS | mean(RGB_enhanced) + ... | 25 (19.5%) | ~80%? |

**信息流对比**：

```
原始分支:
  128 patches → global pooling → concat → classifier
  信息损失：仅 pooling

SDTPS分支:
  128 patches → select 64 → aggregate 25 → mean pooling → concat → classifier
  信息损失：selection + aggregation + pooling
```

---

## ✅ 应用的修复

### 修复1：增大 SDTPS 分支的损失权重

**添加配置参数**（config/defaults.py）：
```python
_C.MODEL.SDTPS_LOSS_WEIGHT = 2.0  # SDTPS 分支权重（原始分支=1.0）
```

**修改损失计算**（engine/processor.py）：
```python
for i in range(0, len(output), 2):
    loss_tmp = loss_fn(score=output[i], feat=output[i+1], ...)
    # SDTPS 分支（i=0）使用更大权重
    if cfg.MODEL.USE_SDTPS and i == 0:
        loss_tmp = loss_tmp * cfg.MODEL.SDTPS_LOSS_WEIGHT  # × 2.0
    loss = loss + loss_tmp

# 新的损失：
loss = 2.0 × loss_sdtps + 1.0 × loss_ori
```

**配置文件**（configs/RGBNT201/DeMo_SDTPS.yml）：
```yaml
SDTPS_LOSS_WEIGHT: 2.0  # SDTPS 权重为原始分支的2倍
```

**效果**：
- ✅ 强制模型更关注 SDTPS 分支
- ✅ 补偿 SDTPS 的信息损失
- ✅ SDTPS 的影响会更明显

### 可选修复（根据实验结果调整）

#### 选项A：进一步增大权重

```yaml
SDTPS_LOSS_WEIGHT: 3.0  # 或更大
```

#### 选项B：只使用 SDTPS 分支

修改 make_model.py，只返回 SDTPS 特征：
```python
if self.USE_SDTPS:
    return sdtps_score, sdtps_feat  # 只返回 SDTPS，不返回 ori
```

#### 选项C：增大 SDTPS 的参数量

修改 sdtps_complete.py：
```python
# 当前：512 → 128 → 1
self.score_predictor = nn.Sequential(
    nn.Linear(512, 512),  # ← 增大隐藏层
    nn.GELU(),
    nn.Linear(512, 128),
    nn.GELU(),
    nn.Linear(128, 1),
    nn.Sigmoid(),
)
```

---

## 📊 预期效果

### 修复前（权重1:1）
```
loss = loss_sdtps + loss_ori
     = 1.5 + 1.5 = 3.0

梯度分配：sdtps 50%, ori 50%
→ 模型同等对待两个分支
→ ori 分支信息更完整，自然占优
```

### 修复后（权重2:1）
```
loss = 2.0 × loss_sdtps + loss_ori
     = 2.0 × 1.5 + 1.5 = 4.5

梯度分配：sdtps 67%, ori 33%
→ 模型更关注 SDTPS 分支
→ 强制学习压缩后的判别特征
```

---

## 🚀 建议的实验方案

### 实验1：验证权重的影响

```bash
# 测试不同的权重
python train_net.py --config_file configs/RGBNT201/DeMo_SDTPS.yml \
    MODEL.SDTPS_LOSS_WEIGHT 1.0  # 基线

python train_net.py --config_file configs/RGBNT201/DeMo_SDTPS.yml \
    MODEL.SDTPS_LOSS_WEIGHT 2.0  # 2倍

python train_net.py --config_file configs/RGBNT201/DeMo_SDTPS.yml \
    MODEL.SDTPS_LOSS_WEIGHT 3.0  # 3倍
```

### 实验2：只用 SDTPS 分支

修改代码，删除原始分支，看 SDTPS 单独的性能。

### 实验3：增大 SDTPS 参数量

增大 MLP 的隐藏层维度，提升学习能力。

---

## 📝 诊断结论

**您的观察完全正确！**

**根本原因**：
1. ✅ SDTPS 参数量太小（0.41%）
2. ✅ 两个分支权重相等（1:1）
3. ✅ 原始分支信息更完整

**不是 SDTPS 不工作，而是它的贡献被原始分支"淹没"了**！

**类比**：
```
总分100分的考试：
  - 主科（Backbone + ori）：99.59分
  - 附加题（SDTPS）：0.41分

即使附加题全对或全错，总分变化都很小！
```

---

## ✅ 已应用的修复

1. ✅ 添加 `SDTPS_LOSS_WEIGHT` 配置参数
2. ✅ 修改损失计算，SDTPS 分支权重 × 2.0
3. ✅ 配置文件设置为 2.0

**下次训练时**，SDTPS 的影响应该会更明显！

---

**建议**：先用权重2.0训练，看效果。如果还不明显，可以：
- 增大到 3.0 或 5.0
- 或者只用 SDTPS 分支
