# SDTPS + DGAF 流程图模板

本文件包含用于 Nano Banana Pro 绘制论文流程图的模板。

---

## Nano Banana Pro Prompt

```
You are an expert ML illustrator.
Draw a clean, NeurIPS/ICLR-style scientific figure using Nano Banana Pro.

GOAL:
Create a professional, publication-quality diagram that exactly follows the
structure and logic provided in the MODULE LIST below.
Do not invent components, do not reinterpret, do not add creativity.
Strictly follow the logical flow.

GLOBAL RULES:

- Flat, clean NeurIPS style (no gradients, no gloss, no shadows)
- Consistent thin line weights
- Professional pastel palette
- Rounded rectangles for blocks
- Arrows must clearly indicate data flow
- No long sentences, only short labels
- Keep spacing clean and balanced
- All modules must appear exactly once unless specified

LAYOUT:

- Horizontal left → right layout (recommended)
- Multi-branch flow: three parallel branches (RGB, NIR, TIR) that merge at SDTPS and DGAF
- Align components cleanly in straight lines
- Respect the module order exactly as listed

MODULE LIST:

1. Input(s):
   - RGB Image (256×128)
   - NIR Image (256×128)
   - TIR Image (256×128)

2. Preprocessing / Encoding / Embedding:
   - Shared ViT-B/16 Backbone (CLIP pretrained)
   - Output per modality: Patch Tokens (B, 128, 512) + Global Feature (B, 512)
   - Global-Local Fusion: concat([global, pool(patches)]) → MLP → enhanced global

3. Core Architecture / Stages / Blocks:
   - SDTPS Module (Sparse-Dense Token-aware Patch Selection):
     - Stage 1: TokenSparse
       - Self-Attention Score: s^{im} = sim(patch, global_self)
       - Cross-Attention Score: s^{m2} = sim(patch, global_other1), s^{m3} = sim(patch, global_other2)
       - MLP Predictor: s^p = σ(MLP(patch))
       - Combined Score: s = (1-2β)·s^p + β·(s^{m2} + s^{m3} + 2·s^{im})
       - Top-K Selection: N=128 → N_s=64 (sparse_ratio=0.5)
       - Redundant Fusion: extra_token = softmax(discarded_scores) × discarded_patches
     - Stage 2: TokenAggregation
       - Learnable Weight Matrix: W = softmax(MLP(selected_patches))
       - Aggregation: v̂_j = Σ W_{ji} × v_i
       - Compress: N_s=64 → N_c=25 (aggr_ratio=0.4)
     - Output: Enhanced Tokens (B, N_c+1, 512) per modality

4. Special Mechanisms:
   - DGAF Module (Dual-Gated Adaptive Fusion) V3:
     - Attention Pooling: learnable query × SDTPS tokens → pooled feature (B, 512) per modality
     - Information Entropy Gate (IEG):
       - Entropy: H(h) = -Σ p·log(p)
       - Reliability Score: exp(-H/τ)
       - Entropy-weighted Fusion: h_entropy = Σ softmax(z·exp(-H/τ)) × h_m
     - Modality Importance Gate (MIG):
       - Gate: g = σ(MLP(concat[h_rgb, h_nir, h_tir])) ∈ (0,1)³
       - Gated Features: h_m_gated = g_m × h_m
       - Importance Fusion: h_importance = Σ h_m_gated
     - Adaptive Fusion:
       - Learnable α (sigmoid constrained)
       - h_fused = α·h_entropy + (1-α)·h_importance
     - Output Enhancement: h_m_out = h_m + MLP(h_fused)
     - Final: concat([h_rgb_out, h_nir_out, h_tir_out]) → (B, 1536)

5. Output Head:
   - BatchNorm1d (1536)
   - Linear Classifier (1536 → num_classes)
   - Losses: ID Loss (Cross-Entropy + Label Smoothing) + Triplet Loss

NOTES:

- Three-branch parallel flow: RGB/NIR/TIR each go through shared backbone independently
- All three branches merge at SDTPS (cross-modal attention guides patch selection)
- SDTPS uses other modalities' global features to compute cross-attention scores
- DGAF further fuses the three enhanced modality features
- Keep SDTPS as a tall block with two internal stages (TokenSparse → TokenAggregation)
- Keep DGAF as a tall block with two parallel gates (IEG ‖ MIG) merging into α-fusion
- Color coding suggestion: RGB=pastel red, NIR=pastel green, TIR=pastel blue, shared=pastel gray
- Show the compression ratio: 128 → 64 → 25 → 26 (with extra token)

STYLE REQUIREMENTS:

- NeurIPS 2024 visual tone
- Very light background (#FAFAFA or white)
- Text left-aligned inside blocks
- Arrows short and clean
- Use consistent vertical spacing
- Mathematical formulas can be simplified to short labels

Generate the final diagram.
```

---

## 关键数据流总结 (ASCII Art)

```
RGB/NIR/TIR (256×128×3)
        ↓
   ViT-B/16 Backbone (shared)
        ↓
Patch Tokens (B,128,512) + Global (B,512)  × 3 modalities
        ↓
   Global-Local Fusion
        ↓
┌─────────────────────────────────────┐
│           SDTPS Module              │
│  ┌─────────────────────────────┐    │
│  │      TokenSparse            │    │
│  │  • Self-attn: s^{im}        │    │
│  │  • Cross-attn: s^{m2},s^{m3}│    │
│  │  • MLP pred: s^p            │    │
│  │  • Score fusion → Top-K     │    │
│  │  • 128 → 64 patches         │    │
│  └─────────────────────────────┘    │
│              ↓                      │
│  ┌─────────────────────────────┐    │
│  │    TokenAggregation         │    │
│  │  • W = softmax(MLP(x))      │    │
│  │  • v̂ = W × x               │    │
│  │  • 64 → 25 patches          │    │
│  │  • +extra → 26 tokens       │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
        ↓
   3 × (B, 26, 512)
        ↓
┌─────────────────────────────────────┐
│           DGAF Module (V3)          │
│  ┌──────────┐    ┌──────────────┐   │
│  │ Attn Pool│ →  │learnable Q×K │   │
│  └──────────┘    └──────────────┘   │
│        ↓ 3 × (B, 512)               │
│  ┌─────────────┬─────────────────┐  │
│  │    IEG      │       MIG       │  │
│  │ H=entropy   │  g=σ(MLP(cat))  │  │
│  │ exp(-H/τ)   │  h_gated=g×h    │  │
│  └──────┬──────┴────────┬────────┘  │
│         └──────┬────────┘           │
│                ↓                    │
│    α·h_entropy + (1-α)·h_importance │
│                ↓                    │
│    h_out = h + MLP(h_fused)         │
└─────────────────────────────────────┘
        ↓
   concat → (B, 1536)
        ↓
   BN → Classifier → ID Loss + Triplet Loss
```

---

## 模块详细说明

### 1. SDTPS (Sparse-Dense Token-aware Patch Selection)

**核心思想**: 利用跨模态信息引导每个模态的 patch 选择，保留最显著的 patches 并压缩冗余信息。

**数学公式**:

1. **MLP 预测得分**:
   $$s_i^p = \sigma(\text{MLP}(v_i))$$

2. **自注意力得分** (patch 与本模态全局特征的相似度):
   $$s_i^{im} = \text{sim}(v_i, g_{self})$$

3. **交叉注意力得分** (patch 与其他模态全局特征的相似度):
   $$s_i^{m2} = \text{sim}(v_i, g_{other1}), \quad s_i^{m3} = \text{sim}(v_i, g_{other2})$$

4. **综合得分**:
   $$s_i = (1-2\beta) \cdot s_i^p + \beta \cdot (s_i^{m2} + s_i^{m3} + 2 \cdot s_i^{im})$$

5. **Token 聚合**:
   $$\hat{v}_j = \sum_{i=1}^{N_s} W_{ji} \cdot v_i^s, \quad W = \text{softmax}(\text{MLP}(V_s))$$

**压缩比例**: 128 → 64 (sparse) → 25 (aggr) + 1 (extra) = 26 tokens

---

### 2. DGAF (Dual-Gated Adaptive Fusion)

**核心思想**: 使用双门控机制自适应融合三个模态的特征，兼顾可靠性（熵门控）和重要性（学习门控）。

**数学公式**:

1. **Attention Pooling** (V3 版本):
   $$h_m = \text{Attention}(Q_m, \text{tokens}_m, \text{tokens}_m)$$

2. **信息熵门控 (IEG)**:
   - 计算熵: $H(h) = -\sum p \cdot \log(p)$
   - 可靠性得分: $\text{score}_m = z_m \cdot \exp(-H_m / \tau)$
   - 熵加权融合: $h_{entropy} = \sum_m \text{softmax}(\text{scores})_m \cdot h_m$

3. **模态重要性门控 (MIG)**:
   - 门控因子: $g = \sigma(\text{MLP}(\text{concat}[h_{rgb}, h_{nir}, h_{tir}])) \in (0,1)^3$
   - 门控融合: $h_{importance} = \sum_m g_m \cdot h_m$

4. **自适应融合**:
   $$h_{fused} = \alpha \cdot h_{entropy} + (1-\alpha) \cdot h_{importance}$$

5. **输出增强**:
   $$h_m^{out} = h_m + \text{MLP}(h_{fused})$$

---

## 配置参数参考

来自 `DeMo_SDTPS_DGAF_ablation.yml`:

```yaml
# SDTPS 配置
USE_SDTPS: True
SDTPS_SPARSE_RATIO: 0.7      # 稀疏选择比例
SDTPS_AGGR_RATIO: 0.5        # 聚合比例
SDTPS_BETA: 0.25             # 得分融合权重
SDTPS_USE_GUMBEL: False      # 是否使用 Gumbel-Softmax
SDTPS_GUMBEL_TAU: 5.0        # Gumbel 温度
SDTPS_LOSS_WEIGHT: 2.0       # 损失权重

# DGAF 配置
USE_DGAF: True
DGAF_VERSION: 'v3'           # 使用 V3 版本（内置 Attention Pooling）
DGAF_TAU: 1.0                # 熵门控温度
DGAF_INIT_ALPHA: 0.5         # α 初始值
DGAF_NUM_HEADS: 8            # Attention Pooling 头数
```

---

## 代码文件参考

| 模块 | 文件路径 |
|------|----------|
| SDTPS | `modeling/sdtps_complete.py` |
| DGAF | `modeling/dual_gated_fusion.py` |
| 模型集成 | `modeling/make_model.py` |
| 配置 | `configs/RGBNT201/DeMo_SDTPS_DGAF_ablation.yml` |
