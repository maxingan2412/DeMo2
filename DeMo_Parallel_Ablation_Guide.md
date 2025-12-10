# DeMo_Parallel 消融实验指南

## 概述

`run_ablation_parallel_201.sh` 脚本用于测试 DeMo_Parallel 架构的所有分支组合。

---

## 架构说明

DeMo_Parallel 包含3个并行分支，每个分支输出3个模态特征（RGB/NIR/TIR）：

```
Backbone
  ├─→ SDTPS  → 3个特征 (token selection + mean pooling)
  ├─→ DGAF   → 3个特征 (adaptive fusion with tokens)
  └─→ Fused  → 3个特征 (global-local fusion)

总计: 9个特征 → 9个分类头
```

---

## 7种消融配置

### 第一轮（4个GPU并行）

#### 1. Fused only (3个头)
```bash
权重: SDTPS=0.0, DGAF=0.0, FUSED=1.0
作用: 测试 global-local fusion 的基线性能
```

#### 2. SDTPS only (3个头)
```bash
权重: SDTPS=1.0, DGAF=0.0, FUSED=0.0
作用: 测试 token selection 的效果
```

#### 3. DGAF only (3个头)
```bash
权重: SDTPS=0.0, DGAF=1.0, FUSED=0.0
作用: 测试 adaptive fusion 的效果
```

#### 4. SDTPS + DGAF (6个头)
```bash
权重: SDTPS=1.0, DGAF=1.0, FUSED=0.0
作用: 测试两个主要模块的协同效果
```

### 第二轮（3个GPU并行）

#### 5. SDTPS + Fused (6个头)
```bash
权重: SDTPS=1.0, DGAF=0.0, FUSED=1.0
作用: token selection + baseline fusion
```

#### 6. DGAF + Fused (6个头)
```bash
权重: SDTPS=0.0, DGAF=1.0, FUSED=1.0
作用: adaptive fusion + baseline fusion
```

#### 7. Full - 所有分支 (9个头)
```bash
权重: SDTPS=1.0, DGAF=1.0, FUSED=1.0
作用: 完整架构，测试是否过拟合
```

---

## 使用方法

### 基础运行
```bash
bash scripts/run_ablation_parallel_201.sh
```

**生成：**
```
logs/RGBNT201_parallel_ablation_20251210_153022/
├── 01_fused_only.log
├── 02_sdtps_only.log
├── 03_dgaf_only.log
├── 04_sdtps_dgaf.log
├── 05_sdtps_fused.log
├── 06_dgaf_fused.log
└── 07_full_9heads.log
```

---

### 带标识运行
```bash
bash scripts/run_ablation_parallel_201.sh test_v1
```

**生成：**
```
logs/RGBNT201_parallel_ablation_test_v1_20251210_153022/
├── 01_fused_only_test_v1.log
├── 02_sdtps_only_test_v1.log
...
└── 07_full_9heads_test_v1.log
```

---

## 实验执行流程

### 第一轮（~12小时，50 epochs）
- GPU 0: Fused only
- GPU 1: SDTPS only
- GPU 2: DGAF only
- GPU 3: SDTPS+DGAF

**等待第一轮完成后，自动启动第二轮**

### 第二轮（~12小时，50 epochs）
- GPU 0: SDTPS+Fused
- GPU 1: DGAF+Fused
- GPU 2: Full (9 heads)

**总时间: ~24小时（串行两轮）**

---

## 分析目标

### 1. 单分支性能排序
- 对比配置 1、2、3
- 识别最强的单分支
- 预期: DGAF > SDTPS > Fused

### 2. 分支互补性
- 对比配置 4、5、6 与对应的单分支
- 检验组合是否优于单独使用
- 预期: SDTPS+DGAF 最强（token selection + adaptive fusion）

### 3. 过拟合检查
- 对比配置 7 (9头) 与最佳双分支
- 如果 9头 < 6头，说明过拟合
- 如果 9头 > 6头，说明9头有效

### 4. 最优配置
- 综合性能、参数量、训练时间
- 推荐最佳生产配置

---

## 预期结果示例

| 配置 | 分类头数 | mAP | Rank-1 | 说明 |
|------|---------|-----|--------|------|
| Fused only | 3 | 74.0 | 76.5 | Baseline |
| SDTPS only | 3 | 75.5 | 78.0 | Token selection |
| DGAF only | 3 | 77.0 | 80.0 | Adaptive fusion 🏆 |
| SDTPS+DGAF | 6 | 78.5 | 81.5 | 组合提升 🏆🏆 |
| SDTPS+Fused | 6 | 76.5 | 79.0 | 中等 |
| DGAF+Fused | 6 | 77.5 | 80.5 | 较好 |
| Full (9头) | 9 | 78.0 | 81.0 | 略低于最佳（过拟合？）|

**推荐配置: SDTPS+DGAF (6头)** - 性能最佳，参数适中

---

## 手动运行单个实验

如果只想测试某个特定配置：

### Fused only
```bash
CUDA_VISIBLE_DEVICES=0 python train_net.py \
  --config_file configs/RGBNT201/DeMo_Parallel.yml \
  MODEL.SDTPS_LOSS_WEIGHT 0.0 \
  MODEL.DGAF_LOSS_WEIGHT 0.0 \
  MODEL.FUSED_LOSS_WEIGHT 1.0
```

### SDTPS + DGAF (推荐)
```bash
CUDA_VISIBLE_DEVICES=0 python train_net.py \
  --config_file configs/RGBNT201/DeMo_Parallel.yml \
  MODEL.SDTPS_LOSS_WEIGHT 1.0 \
  MODEL.DGAF_LOSS_WEIGHT 1.0 \
  MODEL.FUSED_LOSS_WEIGHT 0.0
```

### Full (9头)
```bash
CUDA_VISIBLE_DEVICES=0 python train_net.py \
  --config_file configs/RGBNT201/DeMo_Parallel.yml \
  MODEL.SDTPS_LOSS_WEIGHT 1.0 \
  MODEL.DGAF_LOSS_WEIGHT 1.0 \
  MODEL.FUSED_LOSS_WEIGHT 1.0
```

---

## 注意事项

### 显存要求
- 9头模式: 需要 ~16GB 显存
- 6头模式: 需要 ~12GB 显存
- 3头模式: 需要 ~10GB 显存

**显存不足时：**
```bash
# 减小 batch size
SOLVER.IMS_PER_BATCH 32

# 减小实例数
DATALOADER.NUM_INSTANCE 2
```

### 训练时间
- 3头: ~4小时 (50 epochs)
- 6头: ~6小时 (50 epochs)
- 9头: ~8小时 (50 epochs)

### 权重设置建议
- 主分支: 1.0
- 辅助分支: 0.5-1.0
- 关闭分支: 0.0

---

## 结果分析

运行完成后，查看每个配置的最佳性能：

```bash
grep "Best mAP\|Best Rank-1" logs/RGBNT201_parallel_ablation_*/0*.log
```

对比不同组合的性能提升。

---

## 相关文件

- 脚本: `scripts/run_ablation_parallel_201.sh`
- 配置: `configs/RGBNT201/DeMo_Parallel.yml`
- 模型: `modeling/make_model.py` (DeMo_Parallel 类)
- 设计文档: `DeMo_Parallel_Design.md`
- 快速参考: `DeMo_Parallel_QuickRef.md`
