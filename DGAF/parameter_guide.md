# DGAF 参数对比与调优指南

## 1. 原论文参数设置

### 1.1 原论文超参数 (CMU-MOSI)

| 参数 | 值 | 说明 |
|------|-----|------|
| Batch Size | 32 | 训练批次大小 |
| Learning Rate (BERT) | 5e-5 | BERT 微调学习率 |
| Learning Rate (其他) | 1e-4 | 其他组件学习率 |
| Weight Decay | 0.01 | 权重衰减 |
| VAT Steps | 5 | Virtual Adversarial Training 步数 |
| BiLSTM Layers | 3 | 音频/视觉编码器层数 |
| Early Stopping | 8 epochs | 基于验证集 MAE |
| Optimizer | AdamW | 优化器 |
| LR Scheduler | Cosine Annealing | 学习率调度 |

### 1.2 原论文超参数 (CMU-MOSEI)

| 参数 | 值 | 说明 |
|------|-----|------|
| Batch Size | 128 | 更大的数据集使用更大批次 |
| Weight Decay | 0.05 | 更强的正则化 |
| VAT Steps | 3 | 较少的对抗训练步数 |
| BiLSTM Layers | 2 | 较浅的编码器 |
| Early Stopping | 4 epochs | 更快的早停 |

### 1.3 DGAF 特定参数 (推断)

论文未明确给出，但从公式推断：

| 参数 | 推测值 | 说明 |
|------|--------|------|
| τ (温度) | 1.0 | 熵门控温度，控制权重尖锐度 |
| α (初始) | 0.5 | IEG 和 MIG 的平衡，可学习 |

---

## 2. 我们的参数设置

### 2.1 默认配置

```yaml
MODEL:
  USE_DGAF: True
  DGAF_TAU: 1.0         # 熵门控温度
  DGAF_INIT_ALPHA: 0.5  # α 初始值
```

### 2.2 为什么这样设置？

#### 温度 τ = 1.0

- **原因**: 原论文使用标准的 softmax 公式，τ=1.0 是默认值
- **效果**: 适中的权重分布，不会过于尖锐或平滑
- **对比**:
  - τ=0.1: 权重接近 one-hot，可能过于激进
  - τ=5.0: 权重接近均匀，失去自适应效果

#### 初始 α = 0.5

- **原因**: 论文强调 IEG 和 MIG 互补，需要平衡两者
- **效果**: 训练开始时两个门控贡献相等，让模型自己学习最优比例
- **对比**:
  - α=0.7: 偏向 IEG (可靠性)
  - α=0.3: 偏向 MIG (重要性)

### 2.3 参数适配考虑

| 方面 | 原论文场景 | 我们的场景 | 适配 |
|------|-----------|-----------|------|
| 特征来源 | BERT/BiLSTM | ViT | 统一维度，无需调整 |
| 模态数量 | 3 | 3 | 保持一致 |
| 特征质量 | 不同编码器 | 共享 backbone | 熵差异可能更小，可尝试降低 τ |
| 任务类型 | 回归 | 度量学习 | 关注类间/类内区分度 |

---

## 3. 调优建议

### 3.1 温度 τ 的调优

```bash
# 实验不同温度值
for tau in 0.5 1.0 2.0; do
    python train_net.py --config_file configs/RGBNT201/DeMo_DGAF.yml \
        --exp_name "dgaf_tau${tau}" \
        MODEL.DGAF_TAU ${tau}
done
```

**预期**:
- τ 较小 (0.5): 权重更尖锐，低熵模态获得显著更高权重
- τ 较大 (2.0): 权重更平滑，三个模态权重接近

**观察指标**:
- 训练时打印 entropy_weights 分布
- 如果权重几乎 one-hot → τ 过小
- 如果权重接近 [0.33, 0.33, 0.33] → τ 过大

### 3.2 初始 α 的调优

```bash
# 实验不同初始 α 值
for alpha in 0.3 0.5 0.7; do
    python train_net.py --config_file configs/RGBNT201/DeMo_DGAF.yml \
        --exp_name "dgaf_alpha${alpha}" \
        MODEL.DGAF_INIT_ALPHA ${alpha}
done
```

**预期**:
- α=0.7: 更依赖可靠性评估，适合模态质量差异大的场景
- α=0.3: 更依赖重要性学习，适合模态语义差异大的场景

### 3.3 模态缺失鲁棒性测试

```bash
# 测试模态缺失时的表现
for miss in r n t rn rt nt; do
    python test_net.py --config_file configs/RGBNT201/DeMo_DGAF.yml \
        TEST.MISS ${miss}
done
```

**预期**:
- DGAF 应该比简单 concat 更鲁棒
- 缺失模态的特征（全零）熵高，自动获得低权重

---

## 4. 监控与调试

### 4.1 添加监控代码

在 `dual_gated_fusion.py` 的 forward 中添加：

```python
if self.training and torch.rand(1) < 0.01:  # 1% 概率打印
    print(f"[DGAF] α={self.alpha.item():.4f}")
    print(f"[DGAF] IEG weights: RGB={entropy_weights[:, 0].mean():.4f}, "
          f"NIR={entropy_weights[:, 1].mean():.4f}, "
          f"TIR={entropy_weights[:, 2].mean():.4f}")
    print(f"[DGAF] MIG gates: RGB={gates[:, 0].mean():.4f}, "
          f"NIR={gates[:, 1].mean():.4f}, "
          f"TIR={gates[:, 2].mean():.4f}")
```

### 4.2 检查 α 变化

```python
# 在训练脚本中添加
if epoch % 5 == 0 and hasattr(model.module, 'dgaf'):
    alpha = model.module.dgaf.alpha.item()
    logger.info(f"[Epoch {epoch}] DGAF α = {alpha:.4f}")
```

### 4.3 异常情况处理

| 现象 | 可能原因 | 解决方案 |
|------|---------|---------|
| α 训练后仍为 0.5 | 学习率太小 | 增加 DGAF 参数的学习率 |
| α 趋近于 0 或 1 | 单一门控主导 | 检查特征质量，调整 τ |
| 权重几乎均匀 | τ 过大 | 降低 τ |
| 权重几乎 one-hot | τ 过小 | 增加 τ |

---

## 5. 与其他模块的交互

### 5.1 DGAF + SDTPS (推荐)

```yaml
MODEL:
  USE_SDTPS: True
  USE_DGAF: True
  USE_SACR: False  # DGAF 替代 SACR 的功能
  USE_LIF: False   # DGAF 替代 LIF 的功能
```

**优势**:
- SDTPS: Token 级别的选择和增强
- DGAF: 特征级别的自适应融合
- 两者互补，分别处理局部和全局

### 5.2 单独使用 DGAF

```yaml
MODEL:
  USE_SDTPS: False
  USE_DGAF: True  # 仅在原始全局特征上融合
```

**注意**: 这种配置需要修改 make_model.py，因为当前 DGAF 仅在 SDTPS 后使用。

### 5.3 禁用 DGAF (baseline)

```yaml
MODEL:
  USE_SDTPS: True
  USE_DGAF: False  # 使用简单 concat
```

---

## 6. 参数敏感性分析

基于原论文消融实验推断：

| 组件 | 移除后影响 | 敏感性 |
|------|-----------|--------|
| IEG | Acc-7 下降 3.33% | 高 |
| MIG | Acc-7 下降 2.34% | 中 |
| α (固定) | Acc-7 下降 2.04% | 中 |

**结论**:
- IEG (熵门控) 是核心组件，τ 参数较敏感
- MIG (重要性门控) 是辅助组件，参数相对不敏感
- α 可学习比固定更好，但初始值影响不大
