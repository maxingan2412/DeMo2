# 12组消融实验说明

## 实验设计

**总实验数**: 12组
**设计**: 3个数据集 × 4种架构配置 = 12组实验

### 实验矩阵

| # | 数据集 | 配置 | USE_SDTPS | USE_DGAF | DGAF_VERSION | GLOBAL_LOCAL | CROSS_ATTN |
|---|--------|------|-----------|----------|--------------|--------------|------------|
| 1 | RGBNT201 | Baseline | False | False | - | False | - |
| 2 | RGBNT201 | SDTPS only | True | False | - | False | attention |
| 3 | RGBNT201 | DGAF V3 only | False | True | v3 | False | - |
| 4 | RGBNT201 | SDTPS+DGAF V3 | True | True | v3 | False | attention |
| 5 | RGBNT100 | Baseline | False | False | - | False | - |
| 6 | RGBNT100 | SDTPS only | True | False | - | False | attention |
| 7 | RGBNT100 | DGAF V3 only | False | True | v3 | False | - |
| 8 | RGBNT100 | SDTPS+DGAF V3 | True | True | v3 | False | attention |
| 9 | MSVR310 | Baseline | False | False | - | False | - |
| 10 | MSVR310 | SDTPS only | True | False | - | False | attention |
| 11 | MSVR310 | DGAF V3 only | False | True | v3 | False | - |
| 12 | MSVR310 | SDTPS+DGAF V3 | True | True | v3 | False | attention |

---

## 4种架构配置详解

### 1. Baseline（基准）
```yaml
MODEL.USE_SDTPS: False
MODEL.USE_DGAF: False
MODEL.GLOBAL_LOCAL: False
```

**数据流:**
```
Backbone → RGB_global (B,C) × 3 → concat → ori (B,3C) → Classifier
```

**输出**: `(ori_score, ori)`

---

### 2. SDTPS only
```yaml
MODEL.USE_SDTPS: True
MODEL.USE_DGAF: False
MODEL.GLOBAL_LOCAL: False
MODEL.SDTPS_CROSS_ATTN_TYPE: 'attention'
MODEL.SDTPS_CROSS_ATTN_HEADS: 4
```

**数据流:**
```
Backbone → RGB_cash (B,N,C)
    ↓
SDTPS (soft masking) → RGB_enhanced (B,N,C)
    ↓
mean pooling → RGB_sdtps (B,C) × 3 → concat → sdtps_feat (B,3C)
    ↓
Classifier
```

**输出**: `(sdtps_score, sdtps_feat, ori_score, ori)`

**关键点**:
- 不用 DGAF，直接 mean pooling
- 不用 GLOBAL_LOCAL

---

### 3. DGAF V3 only
```yaml
MODEL.USE_SDTPS: False
MODEL.USE_DGAF: True
MODEL.DGAF_VERSION: 'v3'
MODEL.GLOBAL_LOCAL: False
```

**数据流:**
```
Backbone → RGB_cash (B,N,C) × 3
    ↓
DGAF V3 (attention pooling) → dgaf_feat (B,3C)
    ↓
Classifier
```

**输出**: `(dgaf_score, dgaf_feat, ori_score, ori)`

**关键点**:
- DGAF V3 直接处理 tokens
- 不用 GLOBAL_LOCAL

---

### 4. SDTPS + DGAF V3
```yaml
MODEL.USE_SDTPS: True
MODEL.USE_DGAF: True
MODEL.DGAF_VERSION: 'v3'
MODEL.GLOBAL_LOCAL: False
MODEL.SDTPS_CROSS_ATTN_TYPE: 'attention'
MODEL.SDTPS_CROSS_ATTN_HEADS: 4
```

**数据流:**
```
Backbone → RGB_cash (B,N,C)
    ↓
SDTPS (soft masking) → RGB_enhanced (B,N,C)
    ↓
DGAF V3 (attention pooling) → sdtps_feat (B,3C)
    ↓
Classifier
```

**输出**: `(sdtps_score, sdtps_feat, ori_score, ori)`

**关键点**:
- SDTPS 稀疏选择后，DGAF V3 融合
- 两者都不需要 GLOBAL_LOCAL

---

## 代码修改点

### 1. 支持 Baseline 模式
```python
# modeling/make_model.py:374-377
else:
    # Baseline: 无任何模块
    if self.USE_LIF and lif_loss is not None:
        return (ori_score, ori, lif_loss)
    return (ori_score, ori)
```

### 2. DGAF 非 V3 版本支持 GLOBAL_LOCAL
```python
# modeling/make_model.py:333-356
elif self.USE_DGAF:
    if self.DGAF_VERSION == 'v3':
        dgaf_feat = self.dgaf(RGB_cash, NI_cash, TI_cash)
    else:
        # V1: 需要聚合
        if self.GLOBAL_LOCAL:
            # pool(cash) + global → reduce
            RGB_local = self.pool(RGB_cash.permute(0, 2, 1)).squeeze(-1)
            RGB_dgaf = self.rgb_reduce(torch.cat([RGB_global, RGB_local], dim=-1))
        else:
            # 仅使用 global
            RGB_dgaf = RGB_global

        dgaf_feat = self.dgaf(RGB_dgaf, NI_dgaf, TI_dgaf)
```

---

## 运行脚本

### 一键运行所有12组实验
```bash
bash scripts/run_ablation_4arch_12exp.sh
```

### 运行单个数据集（手动）
```bash
# RGBNT201 - Baseline
CUDA_VISIBLE_DEVICES=0 python train_net.py \
    --config_file configs/RGBNT201/DeMo_SDTPS_DGAF_ablation.yml \
    MODEL.USE_SDTPS False MODEL.USE_DGAF False MODEL.GLOBAL_LOCAL False

# RGBNT201 - SDTPS only
CUDA_VISIBLE_DEVICES=1 python train_net.py \
    --config_file configs/RGBNT201/DeMo_SDTPS_DGAF_ablation.yml \
    MODEL.USE_SDTPS True MODEL.USE_DGAF False MODEL.GLOBAL_LOCAL False

# RGBNT201 - DGAF V3 only
CUDA_VISIBLE_DEVICES=2 python train_net.py \
    --config_file configs/RGBNT201/DeMo_SDTPS_DGAF_ablation.yml \
    MODEL.USE_SDTPS False MODEL.USE_DGAF True MODEL.DGAF_VERSION v3

# RGBNT201 - SDTPS + DGAF V3
CUDA_VISIBLE_DEVICES=3 python train_net.py \
    --config_file configs/RGBNT201/DeMo_SDTPS_DGAF_ablation.yml \
    MODEL.USE_SDTPS True MODEL.USE_DGAF True MODEL.DGAF_VERSION v3
```

---

## 预期结果

### 性能对比（预期）
```
Baseline < SDTPS only ≈ DGAF V3 only < SDTPS + DGAF V3
```

### 参数对比
| 配置 | SDTPS | DGAF | 总参数 |
|------|-------|------|--------|
| Baseline | 0 | 0 | ~86M |
| SDTPS only | ~4.7M | 0 | ~91M |
| DGAF V3 only | 0 | ~3.1M | ~89M |
| SDTPS+DGAF V3 | ~4.7M | ~3.1M | ~94M |

---

## 输出目录结构

```
logs/
├── RGBNT201_4arch_ablation_YYYYMMDD_HHMMSS/
│   ├── 01_baseline.log
│   ├── 02_sdtps_only.log
│   ├── 03_dgaf_v3_only.log
│   └── 04_sdtps_dgaf_v3.log
├── RGBNT100_4arch_ablation_YYYYMMDD_HHMMSS/
│   ├── 01_baseline.log
│   ├── 02_sdtps_only.log
│   ├── 03_dgaf_v3_only.log
│   └── 04_sdtps_dgaf_v3.log
└── MSVR310_4arch_ablation_YYYYMMDD_HHMMSS/
    ├── 01_baseline.log
    ├── 02_sdtps_only.log
    ├── 03_dgaf_v3_only.log
    └── 04_sdtps_dgaf_v3.log
```

---

## 注意事项

1. **GLOBAL_LOCAL 设置**: 所有实验都设为 False
   - Baseline: 不需要
   - SDTPS only: 使用 mean pooling
   - DGAF V3 only: 直接处理 tokens
   - SDTPS + DGAF V3: 两者都不需要

2. **DGAF 版本**: 统一使用 V3
   - V3 直接接受 (B,N,C) 输入
   - 兼容 SDTPS 输出
   - 不需要 GLOBAL_LOCAL

3. **Cross-Attention**: SDTPS 统一使用 'attention'
   - 带逐 head 余弦门控
   - 4 个注意力头

---

**创建时间**: 2025-12-09
**脚本位置**: `scripts/run_ablation_4arch_12exp.sh`
