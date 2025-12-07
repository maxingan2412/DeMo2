# 消融实验结果汇总

## 实验结果总表（含模型复杂度）

| 实验配置 | Best mAP (%) | Best Rank-1 (%) | Params (M) | GFLOPs | 备注 |
|---------|-------------|-----------------|------------|--------|------|
| Baseline | 72.3 | 75.5 | 87.99 | 34.28 | 基准模型 |
| SACR only | 74.0 | 77.6 | 96.38 | 34.28 | +8.4M params |
| SDTPS only | **76.0** | **80.1** | 88.62 | 34.33 | +0.6M params |
| LIF only | 72.0 | 77.2 | 88.16 | 34.28 | +0.2M params |
| SACR + LIF | 73.3 | 74.6 | 96.55 | 34.28 | |
| SACR + SDTPS | 74.8 | 78.0 | 97.02 | 37.55 | |
| SDTPS + LIF | 71.7 | 75.1 | 88.79 | 35.11 | |
| Full (SACR + SDTPS + LIF) | 74.7 | 78.8 | 97.19 | 38.33 | |

## 超参数敏感性分析

### LIF模块参数（Full模型下测试）

| 参数设置 | Best mAP (%) | Best Rank-1 (%) | Params (M) | GFLOPs |
|---------|-------------|-----------------|------------|--------|
| LIF_BETA = 0.2 | 74.6 | **79.3** | 97.19 | 38.33 |
| LIF_BETA = 0.6 | **75.5** | 76.9 | 97.19 | 38.33 |
| LIF_LOSS_WEIGHT = 0.1 | 70.6 | 73.6 | 97.19 | 38.33 |
| LIF_LOSS_WEIGHT = 0.3 | 73.9 | 78.0 | 97.19 | 38.33 |

### SDTPS模块参数（Full模型下测试）

| 参数设置 | Best mAP (%) | Best Rank-1 (%) | Params (M) | GFLOPs |
|---------|-------------|-----------------|------------|--------|
| SDTPS_LOSS = 1.0 | **76.6** | **80.0** | 97.19 | 38.33 |
| SDTPS_LOSS = 3.0 | 69.7 | 74.3 | 97.19 | 38.33 |
| SDTPS_SPARSE = 0.5 | 72.3 | 76.4 | 97.18 | 38.32 |
| SDTPS_SPARSE = 0.8 | 75.8 | 79.4 | 97.19 | 38.34 |

## 模块架构分析

### SACR (Scale-Adaptive Contextual Refinement)
- **功能**: 多尺度空洞卷积 + 通道注意力
- **参数增量**: ~8.4M (主要来自 dilation_rates=[2,3,4] 的 3x3 卷积)
- **计算增量**: 几乎为0 (仅在特征图上做轻量操作)
- **工作原理**:
  - 将 (B, N, D) reshape 为 (B, D, H, W)
  - 并行 1x1 + 空洞3x3 卷积
  - 通道注意力加权
- **适用场景**: 需要扩大感受野时使用

### SDTPS (Sparse and Dense Token-Aware Patch Selection)
- **功能**: 基于跨模态注意力的 Token 选择
- **参数增量**: ~0.6M (仅 MLP Score Predictor)
- **计算增量**: ~0.05 GFLOPs
- **工作原理**:
  - 计算自注意力 s_im (patch 与自身全局的相似度)
  - 计算跨模态注意力 s_m2, s_m3 (patch 与其他模态全局的相似度)
  - MLP 预测重要性 s_pred
  - 综合得分: score = (1-2β)·s_pred + β·(s_m2 + s_m3 + 2·s_im)
  - Top-K 选择保留显著 patch，丢弃的 patch 加权聚合为 extra_token
- **关键参数**: sparse_ratio (保留比例), beta (注意力权重)

### LIF (Local Illumination-aware Fusion)
- **功能**: 基于图像质量的自适应加权
- **参数增量**: ~0.2M (3个轻量 QualityPredictor)
- **计算增量**: 几乎为0
- **工作原理**:
  - RGB: 预测亮度质量
  - NIR: 预测清晰度 (Laplacian方差)
  - TIR: 预测热对比度 (局部标准差)
  - 质量图 resize 到 patch grid，softmax 归一化为权重
  - 对 patch 特征进行逐位置加权
- **关键参数**: beta (温度系数), loss_weight (自监督损失权重)

## 模块冲突分析

### 为什么 SDTPS + LIF 效果差？
1. **选择 vs 加权冲突**: SDTPS 通过 Top-K 选择丢弃低分 patch，而 LIF 会给所有 patch 赋权重
2. **信号干扰**: LIF 基于图像质量（低级特征），SDTPS 基于跨模态语义相关性（高级特征）
3. **处理顺序**: 如果 LIF 先加权后 SDTPS 选择，会改变 SDTPS 的得分分布

### 为什么 Full 模型不如 SDTPS only？
1. SACR 增加参数量，但效果被 SDTPS 的 token 选择稀释
2. 三模块串联导致梯度传播路径复杂，优化困难
3. 模块间存在功能重叠（都是特征增强）

## 最佳配置总结

| 排名 | 配置 | mAP | Rank-1 | 推荐场景 |
|-----|-----|-----|--------|---------|
| 1 | SDTPS_LOSS=1.0 | **76.6%** | 80.0% | 追求最高性能 |
| 2 | SDTPS only | 76.0% | **80.1%** | 性能与简洁平衡 |
| 3 | SDTPS_SPARSE=0.8 | 75.8% | 79.4% | 保留更多 patch |
| 4 | LIF_BETA=0.6 | 75.5% | 76.9% | 需要质量感知 |

## 建议的后续实验

1. **SDTPS only + SDTPS_LOSS=1.0**: 在单独 SDTPS 上使用最优损失权重
2. **SDTPS only + SDTPS_SPARSE=[0.75, 0.85]**: 探索最佳稀疏比例
3. **SDTPS only + 更大的 SDTPS_BETA**: 增加跨模态注意力的权重
4. **SACR + SDTPS + 调参**: 不使用 LIF，调优 SACR 和 SDTPS 的组合

---
*实验数据来源：logs/ 目录下的消融实验日志文件*
*更新时间：2025-12-07*
