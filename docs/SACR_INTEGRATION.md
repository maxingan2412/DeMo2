# SACR 模块集成文档

## 模块介绍

**SACR (Scale-Adaptive Contextual Refinement)**
- 来源：AerialMind 论文
- 功能：扩展感受野 + 通道注意力，增强特征表示
- 特点：输入输出形状相同，即插即用

## 集成位置

```
Backbone → patch 特征 (B, 128, 512)
   ↓
GLOBAL_LOCAL → 融合 local + global
   ↓
SACR → 多尺度上下文增强 ← 新增！
   ↓
SDTPS → Token 选择和聚合
   ↓
Classifier
```

## 配置参数

### config/defaults.py

```python
_C.MODEL.USE_SACR = False  # 是否使用 SACR
_C.MODEL.SACR_DILATION_RATES = [6, 12, 18]  # 空洞卷积膨胀率
```

### configs/RGBNT201/DeMo_SACR_SDTPS.yml

```yaml
MODEL:
  USE_SACR: True
  SACR_DILATION_RATES: [6, 12, 18]
  USE_SDTPS: True
  SDTPS_LOSS_WEIGHT: 2.0
```

## 实现细节

### 初始化（make_model.py）

```python
if self.USE_SACR:
    # 计算 patch grid 尺寸
    patch_h = 256 / 16 = 16
    patch_w = 128 / 16 = 8

    # 为三个模态分别创建 SACR
    self.rgb_sacr = SACR(token_dim=512, height=16, width=8)
    self.nir_sacr = SACR(token_dim=512, height=16, width=8)
    self.tir_sacr = SACR(token_dim=512, height=16, width=8)
```

### 前向传播

```python
# 训练和推理都相同
if self.USE_SACR:
    RGB_cash = self.rgb_sacr(RGB_cash)  # (B, 128, 512) → (B, 128, 512)
    NI_cash = self.nir_sacr(NI_cash)
    TI_cash = self.tir_sacr(TI_cash)
```

## 参数量分析

### 对比

| 配置 | 总参数 | 新增模块 | 占比 |
|------|--------|---------|------|
| Baseline | 88.04M | 0M | 0% |
| 只有 SDTPS | 88.72M | 0.37M | 0.41% |
| **SACR + SDTPS** | **113.90M** | **25.55M** | **22.43%** |

### SACR 单个模块

```
每个 SACR 模块: 8.39M
  ├─ conv1x1: ~0.26M
  ├─ atrous_conv (r=6): ~2.36M
  ├─ atrous_conv (r=12): ~2.36M
  ├─ atrous_conv (r=18): ~2.36M
  ├─ fusion: ~1.05M
  └─ channel_attn: ~0.001M

3个模态: 8.39M × 3 = 25.18M
```

## 预期效果

### 解决 SDTPS 影响不大的问题

**之前的问题**：
- SDTPS 只占 0.41%，影响被稀释
- 两个分支权重相等，ori 分支占主导

**现在的改进**：
1. ✅ **SACR 增加 25.18M 参数**（22.11%）
2. ✅ **SDTPS 损失权重增大到 2.0**
3. ✅ **两个模块协同工作**

**预期**：
- SACR 增强 patch 特征的判别性
- SDTPS 基于增强后的特征选择和聚合
- 更大的参数量 → 更强的学习能力
- 更大的损失权重 → 更明显的影响

### 完整的改进链

```
改进1: SACR 扩展感受野
  → patch 特征包含更多上下文信息
  → 更适合后续的 token selection

改进2: SDTPS 跨模态引导
  → 利用其他模态信息选择显著 patches
  → 压缩冗余信息

改进3: 损失权重 2.0
  → 强制模型关注 SACR+SDTPS 分支
  → 补偿信息压缩的影响
```

## 训练命令

```bash
# SACR + SDTPS 完整版
python train_net.py --config_file configs/RGBNT201/DeMo_SACR_SDTPS.yml

# 对照实验：只有 SDTPS
python train_net.py --config_file configs/RGBNT201/DeMo_SDTPS.yml

# 对照实验：原始 DeMo (HDM+ATM)
python train_net.py --config_file configs/RGBNT201/DeMo.yml
```

## 测试

```bash
python test_sacr_sdtps.py
```

预期输出：
- ✓ 模型创建成功
- ✓ 参数量：113.90M (SACR 22.11%, SDTPS 0.32%)
- ✓ 训练和推理输出正常
- ✓ 梯度传播正常

## 论文创新点

现在可以强调的完整方法：

1. **跨模态引导的 Token 选择**（SDTPS 核心）
2. **多尺度上下文增强**（SACR 贡献）
3. **两阶段特征压缩**（TokenSparse + Aggregation）
4. **端到端可训练**（移除 no_grad）
5. **缺失模态鲁棒**（多模态设计）

## 可能的消融实验

1. Baseline（不用 SACR, 不用 SDTPS）
2. 只用 SACR
3. 只用 SDTPS
4. SACR + SDTPS（完整版）
5. HDM + ATM（原始 DeMo）

通过对比可以说明各模块的贡献。
