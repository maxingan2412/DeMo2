# Dual-Gated Adaptive Fusion (DGAF) 模块文档

## 目录
1. [原论文概述](#1-原论文概述)
2. [原论文 DGAF 结构](#2-原论文-dgaf-结构)
3. [我们的适配实现](#3-我们的适配实现)
4. [参数设置说明](#4-参数设置说明)
5. [使用方法](#5-使用方法)

---

## 1. 原论文概述

### 论文信息
- **标题**: Beyond Simple Fusion: Adaptive Gated Fusion for Robust Multimodal Sentiment Analysis
- **简称**: AGFN (Adaptive Gated Fusion Network)
- **任务**: 多模态情感分析 (Multimodal Sentiment Analysis, MSA)
- **模态**: 文本 (Text)、音频 (Audio)、视觉 (Visual)
- **数据集**: CMU-MOSI, CMU-MOSEI

### 核心问题
原论文指出简单融合方法的问题：
1. **模态质量差异**: 不同模态可能存在噪声、缺失或语义冲突
2. **隐式假设失效**: 简单 concat 假设所有模态同等可靠，与现实不符
3. **细微情感难以捕捉**: 如讽刺场景中，文本正面但语音/视觉负面

### 解决方案
AGFN 提出 **双门控融合机制 (Dual-Gated Fusion)**：
- **信息熵门控 (IEG)**: 评估模态可靠性
- **模态重要性门控 (MIG)**: 学习样本级别的模态重要性
- **自适应平衡**: 可学习参数 α 平衡两个门控

---

## 2. 原论文 DGAF 结构

### 2.1 整体架构

```
原论文架构：
┌─────────────────────────────────────────────────────────────┐
│  Text ──→ BERT Encoder ──→ h_T (文本特征)                   │
│  Audio ──→ COVAREP + BiLSTM ──→ h_A (音频特征)              │
│  Visual ──→ FACET + BiLSTM ──→ h_V (视觉特征)               │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│              Cross-Modal Attention (可选)                    │
│         增强跨模态交互，得到增强后的 h_T', h_A', h_V'        │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│            Dual-Gated Adaptive Fusion (DGAF)                │
│                                                             │
│   输入: h_T, h_A, h_V (各 (B, C) 维度)                      │
│   输出: h_fused (B, C) 融合特征                             │
└─────────────────────────────────────────────────────────────┘
                           ↓
                    Prediction Head
                           ↓
                    Sentiment Score
```

### 2.2 信息熵门控 (Information Entropy Gate, IEG)

**原理**: 低熵特征 = 高确定性 = 高可靠性 → 给予更高权重

**数学公式**:
```
1. 计算每个模态的信息熵:
   H(h_m) = -Σ p(h_m) * log(p(h_m))

2. 计算投影后的 logits:
   z_m = Proj(h_m)  # 线性投影

3. 熵调制得分:
   score_m = z_m * exp(-H(h_m) / τ)
   # τ 是温度参数，控制熵影响的尖锐程度

4. Softmax 归一化:
   w_m = Softmax(score_m)

5. 加权融合:
   h_entropy = Σ w_m * h_m
```

**公式 (论文 Eq. 4)**:
$$h_{\text{entropy}} = \sum_{m\in\{T,A,V\}} \text{Softmax}_m\left(z_m \cdot e^{-H(h_m)/\tau}\right) h_m$$

### 2.3 模态重要性门控 (Modality Importance Gate, MIG)

**原理**: 学习样本级别的模态重要性，通过门控因子动态调整每个模态的贡献

**数学公式**:
```
1. 拼接所有模态特征:
   h_concat = Concat(h_T, h_A, h_V)  # (B, 3C)

2. 生成门控因子:
   g = σ(MLP(h_concat))  # (B, 3), σ 是 Sigmoid

3. 门控缩放:
   h_T_gated = g[:, 0] * h_T
   h_A_gated = g[:, 1] * h_A
   h_V_gated = g[:, 2] * h_V

4. 投影融合:
   h_importance = W_f * Concat(h_T_gated, h_A_gated, h_V_gated)
```

**公式 (论文 Eq. 5)**:
$$h_{\text{importance}} = W_f\left[\sigma(z)\odot h_T,~\sigma(z)\odot h_A,~\sigma(z)\odot h_V\right]$$

### 2.4 自适应融合

**数学公式 (论文 Eq. 6)**:
$$h_{\mathrm{fused}} = \alpha \cdot h_{\mathrm{entropy}} + (1-\alpha) \cdot h_{\mathrm{importance}}$$

其中 α ∈ [0, 1] 是可学习参数，用于平衡两个门控的贡献。

### 2.5 原论文消融实验结果

| 模型 | Acc-2 | F1 | Acc-7 | MAE |
|------|-------|-----|-------|-----|
| AGFN (完整) | **82.75** | **82.68** | **48.69** | **71.02** |
| AGFN w/ IEG only | 81.95 | 81.91 | 45.36 | 73.13 |
| AGFN w/ MIG only | 82.56 | 82.52 | 46.35 | 72.53 |
| AGFN w/o GFM | 82.46 | 82.45 | 46.65 | 72.31 |

**结论**: 双门控机制协同工作效果最佳。

---

## 3. 我们的适配实现

### 3.1 任务差异

| 方面 | 原论文 (AGFN) | 我们的任务 (DeMo) |
|------|---------------|-------------------|
| 任务类型 | 情感分析 (回归/分类) | 行人/车辆重识别 (度量学习) |
| 模态 | Text, Audio, Visual | RGB, NIR, TIR |
| 模态数量 | 3 | 3 |
| 特征来源 | BERT, COVAREP, FACET | ViT Backbone |
| 特征维度 | 不同维度 | 统一 512/768 |
| 融合位置 | 编码器后 | SDTPS 输出后 |
| 输出维度 | 单一融合特征 | 3C (保持与 concat 一致) |

### 3.2 我们的架构位置

```
DeMo 架构中 DGAF 的位置：

┌─────────────────────────────────────────────────────────────┐
│  RGB ──→ ViT Backbone ──→ RGB_cash (B, N, C), RGB_global    │
│  NIR ──→ ViT Backbone ──→ NI_cash (B, N, C), NI_global      │
│  TIR ──→ ViT Backbone ──→ TI_cash (B, N, C), TI_global      │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                    SDTPS (Token Selection)                   │
│                                                             │
│   输入: RGB_cash, NI_cash, TI_cash, RGB_global, ...         │
│   输出: RGB_enhanced, NI_enhanced, TI_enhanced              │
│         (B, K+1, C) - 选中的 K 个 token + CLS token         │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                     Mean Pooling                             │
│                                                             │
│   RGB_sdtps = RGB_enhanced.mean(dim=1)  # (B, C)            │
│   NI_sdtps = NI_enhanced.mean(dim=1)    # (B, C)            │
│   TI_sdtps = TI_enhanced.mean(dim=1)    # (B, C)            │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│    ★ DGAF (DualGatedPostFusion) - 替代简单 concat ★         │
│                                                             │
│   原始: sdtps_feat = concat([RGB, NI, TI])  # 简单拼接      │
│   新版: sdtps_feat = DGAF(RGB, NI, TI)      # 自适应融合    │
│                                                             │
│   输出: (B, 3C) - 保持与原 concat 维度一致                  │
└─────────────────────────────────────────────────────────────┘
                           ↓
                BN → Classifier → Score
```

### 3.3 DualGatedPostFusion 实现

```python
class DualGatedPostFusion(nn.Module):
    """
    双门控后融合模块 - 专门用于 SDTPS 输出的融合
    替代 SDTPS 后的简单 concat，使用双门控机制自适应融合三个模态
    """

    def __init__(
        self,
        feat_dim: int,           # 输入特征维度 (512 for CLIP, 768 for ViT)
        output_dim: int = None,  # 输出维度，默认 3 * feat_dim
        tau: float = 1.0,        # 熵门控温度
        init_alpha: float = 0.5, # α 初始值
        hidden_dim: int = None,  # MIG 隐藏层维度
    ):
        super().__init__()

        # ========== 信息熵门控 (IEG) ==========
        self.entropy_proj = nn.Linear(feat_dim, feat_dim)

        # ========== 模态重要性门控 (MIG) ==========
        self.gate_net = nn.Sequential(
            nn.Linear(3 * feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid()
        )

        # ========== 可学习的平衡参数 α ==========
        self._alpha = nn.Parameter(torch.tensor(init_alpha))

        # ========== 输出投影 ==========
        self.modal_enhance = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
        )

    def forward(self, h_rgb, h_nir, h_tir):
        # 1. IEG: 计算熵加权融合
        H_rgb = self.compute_entropy(h_rgb)
        H_nir = self.compute_entropy(h_nir)
        H_tir = self.compute_entropy(h_tir)

        z_rgb = self.entropy_proj(h_rgb).mean(dim=-1)
        z_nir = self.entropy_proj(h_nir).mean(dim=-1)
        z_tir = self.entropy_proj(h_tir).mean(dim=-1)

        score_rgb = z_rgb * torch.exp(-H_rgb / self.tau)
        score_nir = z_nir * torch.exp(-H_nir / self.tau)
        score_tir = z_tir * torch.exp(-H_tir / self.tau)

        entropy_weights = F.softmax(torch.stack([score_rgb, score_nir, score_tir], dim=-1), dim=-1)
        h_entropy = entropy_weights[:, 0:1] * h_rgb + entropy_weights[:, 1:2] * h_nir + entropy_weights[:, 2:3] * h_tir

        # 2. MIG: 计算门控加权融合
        h_concat = torch.cat([h_rgb, h_nir, h_tir], dim=-1)
        gates = self.gate_net(h_concat)
        h_importance = gates[:, 0:1] * h_rgb + gates[:, 1:2] * h_nir + gates[:, 2:3] * h_tir

        # 3. 自适应融合
        alpha = torch.sigmoid(self._alpha)
        h_fused = alpha * h_entropy + (1 - alpha) * h_importance

        # 4. 输出增强后拼接
        h_enhance = self.modal_enhance(h_fused)
        return torch.cat([h_rgb + h_enhance, h_nir + h_enhance, h_tir + h_enhance], dim=-1)
```

### 3.4 与原论文的差异

| 方面 | 原论文 | 我们的实现 | 原因 |
|------|--------|-----------|------|
| 输出维度 | (B, C) 单一特征 | (B, 3C) 增强拼接 | 保持与下游分类器兼容 |
| 融合方式 | 加权求和 | 增强后拼接 | 保留各模态信息用于重识别 |
| MIG 输出 | 缩放后投影 | 缩放后求和 | 简化计算 |
| 训练损失 | L1 + VAT | ID Loss + Triplet | 适应重识别任务 |

---

## 4. 参数设置说明

### 4.1 配置参数

```yaml
MODEL:
  USE_DGAF: True              # 启用 DGAF
  DGAF_TAU: 1.0               # 熵门控温度
  DGAF_INIT_ALPHA: 0.5        # α 初始值
```

### 4.2 参数含义

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| `DGAF_TAU` | 1.0 | (0, +∞) | 熵门控温度。越小权重越尖锐（近乎 one-hot），越大越平滑 |
| `DGAF_INIT_ALPHA` | 0.5 | [0, 1] | α 初始值。0=仅 MIG，1=仅 IEG，0.5=平衡 |

### 4.3 温度参数 τ 的影响

```
τ = 0.1: 权重非常尖锐，几乎只选择熵最低的模态
τ = 0.5: 权重较尖锐，低熵模态获得明显更高权重
τ = 1.0: 权重适中，默认推荐值
τ = 2.0: 权重较平滑，各模态权重差异较小
τ = 5.0: 权重非常平滑，接近均匀分布
```

### 4.4 参数量统计

```
DualGatedPostFusion (feat_dim=512, output_dim=1536):
- entropy_proj: 512 × 512 = 262,144
- gate_net: 1536 × 512 + 512 × 3 = 788,480 + 1,539 = 790,019
- modal_enhance: 512 × 512 + 512 = 262,656
- _alpha: 1

Total: ~1.31M parameters
```

---

## 5. 使用方法

### 5.1 训练命令

```bash
# 使用 DGAF + SDTPS
python train_net.py --config_file configs/RGBNT201/DeMo_DGAF.yml \
    --exp_name "sdtps_dgaf"

# 对比实验：不使用 DGAF (简单 concat)
python train_net.py --config_file configs/RGBNT201/DeMo_DGAF.yml \
    --exp_name "sdtps_concat" MODEL.USE_DGAF False

# 调整温度参数
python train_net.py --config_file configs/RGBNT201/DeMo_DGAF.yml \
    --exp_name "sdtps_dgaf_tau0.5" MODEL.DGAF_TAU 0.5

# 调整 α 初始值
python train_net.py --config_file configs/RGBNT201/DeMo_DGAF.yml \
    --exp_name "sdtps_dgaf_alpha0.7" MODEL.DGAF_INIT_ALPHA 0.7
```

### 5.2 消融实验脚本

```bash
bash scripts/dgaf_experiments.sh
```

### 5.3 预期效果

DGAF 相比简单 concat 的优势：
1. **抑制噪声模态**: 低质量/高熵模态获得更低权重
2. **自适应权重**: 样本级别动态调整模态贡献
3. **模态缺失鲁棒性**: 缺失模态特征熵高，自动降权
4. **跨模态冲突处理**: 通过双门控机制解决模态不一致问题

---

## 参考文献

```bibtex
@article{wu2025agfn,
  title={Beyond Simple Fusion: Adaptive Gated Fusion for Robust Multimodal Sentiment Analysis},
  author={Wu, Han and Sun, Yanming and Yang, Yunhe and Wong, Derek F.},
  journal={ICASSP},
  year={2025}
}
```
