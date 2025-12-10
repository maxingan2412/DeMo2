# DeMo_Parallel 架构设计文档

## 1. 需求分析

### 当前架构（顺序）
```
Backbone → SDTPS → DGAF → 输出
```
- SDTPS 输出：`cat([RGB_final, NI_final, TI_final])` = 1个拼接特征 (B, 3C)
- DGAF 输出：`cat([RGB_dgaf, NI_dgaf, TI_dgaf])` = 1个拼接特征 (B, 3C)
- 总计：2个特征 → 2个分类头

### 新架构（并行）
```
Backbone
  ├─→ SDTPS (并行) → RGB_enh, NI_enh, TI_enh (3个特征)
  ├─→ DGAF (并行) → RGB_dgaf, NI_dgaf, TI_dgaf (3个独立特征)
  └─→ Fused → RGB_fused, NI_fused, TI_fused (3个特征)

总计: 9个特征 → 9个分类头 → 9个分数
```

## 2. DGAF 输出分离方案

### 方案A：修改 DGAF 类（推荐）

创建新类 `DualGatedAdaptiveFusionV4`，核心改动：

```python
def forward(self, h_rgb, h_nir, h_tir):
    # ... 内部处理 ...

    # 信息熵门控
    h_entropy = self.entropy_gate(h_rgb, h_nir, h_tir)

    # 模态重要性门控
    h_importance = self.importance_gate(h_rgb, h_nir, h_tir)

    # 自适应融合特征（用于增强）
    alpha = self.alpha
    h_fused = alpha * h_entropy + (1 - alpha) * h_importance
    h_enhance = self.modal_enhance(h_fused)

    # 用融合特征增强各模态（保持独立）
    h_rgb_out = h_rgb + h_enhance
    h_nir_out = h_nir + h_enhance
    h_tir_out = h_tir + h_enhance

    # 返回3个独立特征（而非拼接）
    return h_rgb_out, h_nir_out, h_tir_out  # 3 x (B, C)
```

**优势：**
- 清晰的接口：3个输入 → 3个独立输出
- 保留门控机制：IEG + MIG 仍然工作
- 易于集成：与 DeMo_Parallel 完美匹配

### 方案B：在 DeMo_Parallel 中拆分（不推荐）

保持 DGAF 输出 (B, 3C)，在 DeMo_Parallel 中手动拆分：
```python
dgaf_feat = self.dgaf(...)  # (B, 3C)
RGB_dgaf, NI_dgaf, TI_dgaf = torch.split(dgaf_feat, self.feat_dim, dim=-1)
```

**劣势：**
- 语义模糊：DGAF "融合"后又"拆分"，逻辑矛盾
- 不够清晰：拆分逻辑分散在模型代码中

**决策：使用方案A**

## 3. DeMo_Parallel 详细设计

### 3.1 __init__ 代码框架

```python
class DeMo_Parallel(nn.Module):
    def __init__(self, num_classes, cfg, camera_num, view_num, factory):
        super().__init__()

        # ========== 基础配置 ==========
        if 'vit_base_patch16_224' in cfg.MODEL.TRANSFORMER_TYPE:
            self.feat_dim = 768
        elif 'ViT-B-16' in cfg.MODEL.TRANSFORMER_TYPE:
            self.feat_dim = 512

        self.num_classes = num_classes
        self.cfg = cfg
        self.image_size = cfg.INPUT.SIZE_TRAIN
        self.miss_type = cfg.TEST.MISS
        self.GLOBAL_LOCAL = cfg.MODEL.GLOBAL_LOCAL

        # ========== Backbone ==========
        self.BACKBONE = build_transformer(num_classes, cfg, camera_num, view_num, factory, feat_dim=self.feat_dim)

        # ========== Global-Local Fusion Layers ==========
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.rgb_reduce = nn.Sequential(
            nn.LayerNorm(2 * self.feat_dim),
            nn.Linear(2 * self.feat_dim, self.feat_dim),
            QuickGELU()
        )
        self.nir_reduce = nn.Sequential(
            nn.LayerNorm(2 * self.feat_dim),
            nn.Linear(2 * self.feat_dim, self.feat_dim),
            QuickGELU()
        )
        self.tir_reduce = nn.Sequential(
            nn.LayerNorm(2 * self.feat_dim),
            nn.Linear(2 * self.feat_dim, self.feat_dim),
            QuickGELU()
        )

        # ========== 并行分支1: SDTPS ==========
        h, w = cfg.INPUT.SIZE_TRAIN
        stride_h, stride_w = cfg.MODEL.STRIDE_SIZE
        num_patches = (h // stride_h) * (w // stride_w)

        self.sdtps = MultiModalSDTPS(
            embed_dim=self.feat_dim,
            num_patches=num_patches,
            sparse_ratio=cfg.MODEL.SDTPS_SPARSE_RATIO,
            aggr_ratio=cfg.MODEL.SDTPS_AGGR_RATIO,
            use_gumbel=cfg.MODEL.SDTPS_USE_GUMBEL,
            gumbel_tau=cfg.MODEL.SDTPS_GUMBEL_TAU,
            beta=cfg.MODEL.SDTPS_BETA,
            cross_attn_type=cfg.MODEL.SDTPS_CROSS_ATTN_TYPE,
            cross_attn_heads=cfg.MODEL.SDTPS_CROSS_ATTN_HEADS,
        )

        # ========== 并行分支2: DGAF ==========
        # 使用新版本 V4（返回3个独立特征）
        self.dgaf = DualGatedAdaptiveFusionV4(
            feat_dim=self.feat_dim,
            tau=cfg.MODEL.DGAF_TAU,
            init_alpha=cfg.MODEL.DGAF_INIT_ALPHA,
        )

        # ========== 9个分类头 ==========
        # SDTPS 分支 (3个)
        self.classifier_sdtps_rgb = nn.Linear(self.feat_dim, num_classes, bias=False)
        self.classifier_sdtps_nir = nn.Linear(self.feat_dim, num_classes, bias=False)
        self.classifier_sdtps_tir = nn.Linear(self.feat_dim, num_classes, bias=False)
        self.bottleneck_sdtps_rgb = nn.BatchNorm1d(self.feat_dim)
        self.bottleneck_sdtps_nir = nn.BatchNorm1d(self.feat_dim)
        self.bottleneck_sdtps_tir = nn.BatchNorm1d(self.feat_dim)

        # DGAF 分支 (3个)
        self.classifier_dgaf_rgb = nn.Linear(self.feat_dim, num_classes, bias=False)
        self.classifier_dgaf_nir = nn.Linear(self.feat_dim, num_classes, bias=False)
        self.classifier_dgaf_tir = nn.Linear(self.feat_dim, num_classes, bias=False)
        self.bottleneck_dgaf_rgb = nn.BatchNorm1d(self.feat_dim)
        self.bottleneck_dgaf_nir = nn.BatchNorm1d(self.feat_dim)
        self.bottleneck_dgaf_tir = nn.BatchNorm1d(self.feat_dim)

        # Fused 分支 (3个)
        self.classifier_fused_rgb = nn.Linear(self.feat_dim, num_classes, bias=False)
        self.classifier_fused_nir = nn.Linear(self.feat_dim, num_classes, bias=False)
        self.classifier_fused_tir = nn.Linear(self.feat_dim, num_classes, bias=False)
        self.bottleneck_fused_rgb = nn.BatchNorm1d(self.feat_dim)
        self.bottleneck_fused_nir = nn.BatchNorm1d(self.feat_dim)
        self.bottleneck_fused_tir = nn.BatchNorm1d(self.feat_dim)

        # 初始化所有分类头
        for m in [
            self.classifier_sdtps_rgb, self.classifier_sdtps_nir, self.classifier_sdtps_tir,
            self.classifier_dgaf_rgb, self.classifier_dgaf_nir, self.classifier_dgaf_tir,
            self.classifier_fused_rgb, self.classifier_fused_nir, self.classifier_fused_tir,
        ]:
            m.apply(weights_init_classifier)

        for m in [
            self.bottleneck_sdtps_rgb, self.bottleneck_sdtps_nir, self.bottleneck_sdtps_tir,
            self.bottleneck_dgaf_rgb, self.bottleneck_dgaf_nir, self.bottleneck_dgaf_tir,
            self.bottleneck_fused_rgb, self.bottleneck_fused_nir, self.bottleneck_fused_tir,
        ]:
            m.bias.requires_grad_(False)
            m.apply(weights_init_kaiming)
```

### 3.2 forward 代码框架

```python
def forward(self, x, label=None, cam_label=None, view_label=None):
    # ========== 1. Input Preparation ==========
    RGB, NI, TI = x['RGB'], x['NI'], x['TI']
    if 'cam_label' in x:
        cam_label = x['cam_label']

    # Missing modality simulation (inference only)
    if not self.training:
        if self.miss_type == 'r': RGB = torch.zeros_like(RGB)
        elif self.miss_type == 'n': NI = torch.zeros_like(NI)
        elif self.miss_type == 't': TI = torch.zeros_like(TI)
        elif self.miss_type == 'rn': RGB, NI = torch.zeros_like(RGB), torch.zeros_like(NI)
        elif self.miss_type == 'rt': RGB, TI = torch.zeros_like(RGB), torch.zeros_like(TI)
        elif self.miss_type == 'nt': NI, TI = torch.zeros_like(NI), torch.zeros_like(TI)

    # ========== 2. Backbone Feature Extraction ==========
    RGB_cash, RGB_global = self.BACKBONE(RGB, cam_label=cam_label, view_label=view_label)
    NI_cash, NI_global = self.BACKBONE(NI, cam_label=cam_label, view_label=view_label)
    TI_cash, TI_global = self.BACKBONE(TI, cam_label=cam_label, view_label=view_label)

    # ========== 3. 并行分支1: SDTPS ==========
    # Token selection
    RGB_enh, NI_enh, TI_enh, _, _, _ = self.sdtps(
        RGB_cash, NI_cash, TI_cash, RGB_global, NI_global, TI_global
    )

    # Feature aggregation
    def fuse_global_local(feat_cash, feat_global, pool_layer, reduce_layer):
        feat_local = pool_layer(feat_cash.permute(0, 2, 1)).squeeze(-1)
        return reduce_layer(torch.cat([feat_global, feat_local], dim=-1))

    if self.GLOBAL_LOCAL:
        feat_sdtps_rgb = fuse_global_local(RGB_enh, RGB_global, self.pool, self.rgb_reduce)
        feat_sdtps_nir = fuse_global_local(NI_enh, NI_global, self.pool, self.nir_reduce)
        feat_sdtps_tir = fuse_global_local(TI_enh, TI_global, self.pool, self.tir_reduce)
    else:
        feat_sdtps_rgb = RGB_enh.mean(dim=1)
        feat_sdtps_nir = NI_enh.mean(dim=1)
        feat_sdtps_tir = TI_enh.mean(dim=1)

    # ========== 4. 并行分支2: DGAF ==========
    if self.GLOBAL_LOCAL:
        RGB_in = fuse_global_local(RGB_cash, RGB_global, self.pool, self.rgb_reduce)
        NI_in = fuse_global_local(NI_cash, NI_global, self.pool, self.nir_reduce)
        TI_in = fuse_global_local(TI_cash, TI_global, self.pool, self.tir_reduce)
    else:
        RGB_in, NI_in, TI_in = RGB_global, NI_global, TI_global

    feat_dgaf_rgb, feat_dgaf_nir, feat_dgaf_tir = self.dgaf(RGB_in, NI_in, TI_in)

    # ========== 5. 并行分支3: Fused ==========
    # 直接使用 backbone 的 global-local fusion
    if self.GLOBAL_LOCAL:
        feat_fused_rgb = fuse_global_local(RGB_cash, RGB_global, self.pool, self.rgb_reduce)
        feat_fused_nir = fuse_global_local(NI_cash, NI_global, self.pool, self.nir_reduce)
        feat_fused_tir = fuse_global_local(TI_cash, TI_global, self.pool, self.tir_reduce)
    else:
        feat_fused_rgb = RGB_global
        feat_fused_nir = NI_global
        feat_fused_tir = TI_global

    # ========== 6. Training: Calculate Scores ==========
    if self.training:
        # SDTPS 分支 (3个)
        score_sdtps_rgb = self.classifier_sdtps_rgb(self.bottleneck_sdtps_rgb(feat_sdtps_rgb))
        score_sdtps_nir = self.classifier_sdtps_nir(self.bottleneck_sdtps_nir(feat_sdtps_nir))
        score_sdtps_tir = self.classifier_sdtps_tir(self.bottleneck_sdtps_tir(feat_sdtps_tir))

        # DGAF 分支 (3个)
        score_dgaf_rgb = self.classifier_dgaf_rgb(self.bottleneck_dgaf_rgb(feat_dgaf_rgb))
        score_dgaf_nir = self.classifier_dgaf_nir(self.bottleneck_dgaf_nir(feat_dgaf_nir))
        score_dgaf_tir = self.classifier_dgaf_tir(self.bottleneck_dgaf_tir(feat_dgaf_tir))

        # Fused 分支 (3个)
        score_fused_rgb = self.classifier_fused_rgb(self.bottleneck_fused_rgb(feat_fused_rgb))
        score_fused_nir = self.classifier_fused_nir(self.bottleneck_fused_nir(feat_fused_nir))
        score_fused_tir = self.classifier_fused_tir(self.bottleneck_fused_tir(feat_fused_tir))

        # 返回格式：((score1, feat1), (score2, feat2), ...) - 9对
        return (
            # SDTPS 分支
            score_sdtps_rgb, feat_sdtps_rgb,
            score_sdtps_nir, feat_sdtps_nir,
            score_sdtps_tir, feat_sdtps_tir,
            # DGAF 分支
            score_dgaf_rgb, feat_dgaf_rgb,
            score_dgaf_nir, feat_dgaf_nir,
            score_dgaf_tir, feat_dgaf_tir,
            # Fused 分支
            score_fused_rgb, feat_fused_rgb,
            score_fused_nir, feat_fused_nir,
            score_fused_tir, feat_fused_tir,
        )  # 18个值 (9对 score-feat)

    # ========== 7. Inference: Concatenate All Features ==========
    else:
        return torch.cat([
            feat_sdtps_rgb, feat_sdtps_nir, feat_sdtps_tir,
            feat_dgaf_rgb, feat_dgaf_nir, feat_dgaf_tir,
            feat_fused_rgb, feat_fused_nir, feat_fused_tir,
        ], dim=-1)  # (B, 9C)
```

## 4. 返回值格式优化

### 方案A：扁平元组（当前方案）
```python
return (
    score1, feat1, score2, feat2, ..., score9, feat9
)  # 18个值
```
**优势：** 与现有 processor.py 兼容（循环遍历 score-feat 对）

### 方案B：嵌套字典（更清晰）
```python
return {
    'sdtps': {
        'rgb': (score_sdtps_rgb, feat_sdtps_rgb),
        'nir': (score_sdtps_nir, feat_sdtps_nir),
        'tir': (score_sdtps_tir, feat_sdtps_tir),
    },
    'dgaf': {...},
    'fused': {...},
}
```
**优势：** 清晰易读，但需要修改 processor.py

**决策：使用方案A（兼容性优先）**

## 5. processor.py 修改需求

### 当前代码（处理偶数个 score-feat 对）
```python
for i in range(0, len(output), 2):
    loss_tmp = loss_fn(score=output[i], feat=output[i+1], target=target, target_cam=target_cam)
    loss = loss + loss_tmp
```

### DeMo_Parallel 的输出
- 训练：18个值（9对 score-feat）
- 推理：(B, 9C) 特征

### 是否需要修改？
**不需要修改！** 当前循环逻辑自动支持任意偶数个输出：
```python
len(output) = 18
range(0, 18, 2) = [0, 2, 4, 6, 8, 10, 12, 14, 16]  # 9次迭代，9个损失项
```

### 可选：添加分支权重（高级优化）
```python
# 为不同分支设置不同权重
branch_weights = {
    'sdtps': cfg.MODEL.SDTPS_LOSS_WEIGHT,  # 例如 1.0
    'dgaf': cfg.MODEL.DGAF_LOSS_WEIGHT,    # 例如 1.0
    'fused': cfg.MODEL.FUSED_LOSS_WEIGHT,  # 例如 0.5（辅助分支）
}

for i in range(0, len(output), 2):
    loss_tmp = loss_fn(score=output[i], feat=output[i+1], target=target, target_cam=target_cam)

    # 根据分支索引应用权重
    if i < 6:  # SDTPS (0, 2, 4)
        loss_tmp *= branch_weights['sdtps']
    elif i < 12:  # DGAF (6, 8, 10)
        loss_tmp *= branch_weights['dgaf']
    else:  # Fused (12, 14, 16)
        loss_tmp *= branch_weights['fused']

    loss = loss + loss_tmp
```

## 6. 潜在风险分析与缓解方案

### 风险1: 过拟合（9个分类头）
**原因：**
- 参数量增加：9 × (feat_dim × num_classes) = 9 × (512 × 201) ≈ 0.93M
- 模型容量过大可能导致在小数据集上过拟合

**缓解方案：**
1. **正则化增强**
   - Dropout：在分类头前添加 `nn.Dropout(0.5)`
   - Label Smoothing：`cfg.MODEL.IF_LABELSMOOTH = True`
   - Weight Decay：增大 `cfg.SOLVER.WEIGHT_DECAY`

2. **数据增强**
   - Random Erasing：`cfg.INPUT.RE_PROB = 0.5`
   - Color Jitter：`cfg.INPUT.COLOR_JITTER = True`

3. **Early Stopping**
   - 监控验证集性能，提前停止训练

4. **分支权重调整**
   - 降低 Fused 分支权重（辅助监督）：`FUSED_LOSS_WEIGHT = 0.5`

### 风险2: 计算开销增加
**原因：**
- 9个分类头前向传播
- 9个损失计算

**缓解方案：**
1. **并行计算（已实现）**
   - SDTPS/DGAF/Fused 三分支完全并行（无依赖）
   - GPU 并行效率高

2. **混合精度训练**
   - `amp.autocast(enabled=True)` 已启用
   - 减少显存占用和计算时间

3. **梯度累积**
   ```python
   accumulation_steps = 4
   if (n_iter + 1) % accumulation_steps == 0:
       optimizer.step()
       optimizer.zero_grad()
   ```

### 风险3: 内存占用增加
**原因：**
- 9个特征需要存储：9 × (B × C)
- 9个梯度需要缓存

**缓解方案：**
1. **减小 Batch Size**
   - 从 `IMS_PER_BATCH=128` 降至 `64` 或 `32`

2. **梯度检查点（Gradient Checkpointing）**
   ```python
   from torch.utils.checkpoint import checkpoint

   # 在 forward 中使用
   feat_sdtps_rgb = checkpoint(fuse_global_local, RGB_enh, RGB_global, self.pool, self.rgb_reduce)
   ```

3. **释放中间变量**
   ```python
   del RGB_enh, NI_enh, TI_enh  # 在不再需要时释放
   torch.cuda.empty_cache()
   ```

### 风险4: 收敛速度变慢
**原因：**
- 9个分类头需要更多轮次学习

**缓解方案：**
1. **学习率调整**
   - 增大初始学习率：`BASE_LR = 0.0005` → `0.001`
   - 使用 Warmup：`WARMUP_EPOCHS = 5`

2. **分阶段训练**
   - Stage 1: 冻结 backbone，只训练分类头（10 epochs）
   - Stage 2: 解冻全部，联合训练（50 epochs）

## 7. 配置参数建议

在 `configs/RGBNT201/DeMo_Parallel.yml` 中添加：

```yaml
MODEL:
  USE_PARALLEL: True  # 启用 DeMo_Parallel 架构

  # 分支权重
  SDTPS_LOSS_WEIGHT: 1.0
  DGAF_LOSS_WEIGHT: 1.0
  FUSED_LOSS_WEIGHT: 0.5  # 辅助监督

  # DGAF 版本
  DGAF_VERSION: 'v4'  # 使用新版本（返回3个独立特征）

  # 过拟合缓解
  DROPOUT: 0.5
  IF_LABELSMOOTH: True

SOLVER:
  WEIGHT_DECAY: 0.0005  # 增大正则化
  BASE_LR: 0.0005
  WARMUP_EPOCHS: 5

INPUT:
  RE_PROB: 0.5  # Random Erasing

DATALOADER:
  IMS_PER_BATCH: 64  # 减小 batch size（如果显存不足）
```

## 8. 使用建议

### 训练
```bash
# 基础训练
python train_net.py --config_file configs/RGBNT201/DeMo_Parallel.yml

# 显存不足时
python train_net.py --config_file configs/RGBNT201/DeMo_Parallel.yml \
  DATALOADER.IMS_PER_BATCH 32 \
  DATALOADER.NUM_INSTANCE 2
```

### 测试
```bash
# 推理时自动拼接9个特征
python test_net.py --config_file configs/RGBNT201/DeMo_Parallel.yml
```

### 消融实验
```bash
# 仅测试 SDTPS 分支（修改推理逻辑）
# 仅测试 DGAF 分支
# 仅测试 Fused 分支
```

## 9. 预期性能提升

### 理论优势
1. **多样性增强：** 9个特征从不同角度捕捉模态信息
2. **鲁棒性提升：** 并行架构避免错误累积
3. **缺失模态鲁棒性：** DGAF 分支可自适应调整权重

### 预期指标（RGBNT201）
- Baseline (直接拼接): mAP ~70%
- DeMo (SDTPS+DGAF 顺序): mAP ~75%
- **DeMo_Parallel (并行+9头): mAP ~77-80%** (预期提升 2-5%)

### 消融实验建议
| 架构 | mAP | Rank-1 | 参数量 |
|------|-----|--------|--------|
| Baseline | 70.0 | 72.5 | 0.3M |
| SDTPS Only | 73.5 | 76.0 | 0.5M |
| DGAF Only | 72.0 | 74.5 | 0.4M |
| SDTPS→DGAF (顺序) | 75.0 | 78.0 | 0.6M |
| **DeMo_Parallel (并行+9头)** | **77.5** | **80.5** | **1.2M** |

## 10. 总结

### 核心设计决策
1. ✅ **DGAF 输出分离：** 创建 V4 版本，返回3个独立特征
2. ✅ **并行架构：** 3条分支完全独立，无依赖
3. ✅ **9个分类头：** 每个分支每个模态独立监督
4. ✅ **返回值兼容：** 18个值（9对 score-feat），与现有 processor.py 兼容
5. ✅ **过拟合缓解：** Dropout + Label Smoothing + 分支权重

### 实现优先级
1. **创建 DualGatedAdaptiveFusionV4 类**
2. **创建 DeMo_Parallel 类**
3. **添加配置参数**
4. **训练测试**
5. **消融实验**

### 下一步行动
1. 实现 `DualGatedAdaptiveFusionV4`
2. 实现 `DeMo_Parallel`
3. 修改 `make_model.py` 支持新模型
4. 创建配置文件 `DeMo_Parallel.yml`
5. 训练并评估
