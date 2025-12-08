# SDTPS + DGAF 简化流程图模板

---

## Nano Banana Pro Prompt (简化版)

```
You are an expert ML illustrator.
Draw a clean, NeurIPS/ICLR-style scientific figure using Nano Banana Pro.

GOAL:
Create a professional, publication-quality diagram focusing on TWO novel modules:
SDTPS (Sparse-Dense Token-aware Patch Selection) and DGAF (Dual-Gated Adaptive Fusion).
Simplify the backbone part, highlight the innovation.

GLOBAL RULES:

- Flat, clean NeurIPS style (no gradients, no shadows)
- Professional pastel palette
- Rounded rectangles for blocks
- Arrows clearly indicate data flow
- Short labels only
- Focus on SDTPS and DGAF modules

LAYOUT:

- Vertical top → bottom layout
- Three parallel branches (RGB, NIR, TIR) shown as simplified inputs
- SDTPS and DGAF as detailed expanded blocks

MODULE LIST:

1. Input & Backbone (SIMPLIFIED - draw as single compact block):
   - Three inputs: RGB / NIR / TIR
   - Shared ViT Backbone
   - Output: Patch Tokens + Global Features (×3 modalities)

2. SDTPS Module (HIGHLIGHT - main innovation):
   - Cross-Modal Attention Scoring:
     * Each modality's patches attend to OTHER modalities' global features
     * s = (1-2β)·s^p + β·(s^{cross} + 2·s^{self})
   - Token Selection: Top-K salient patches
   - Token Aggregation: Learnable weighted pooling
   - Key insight: "Cross-modal guidance for patch selection"

3. DGAF Module (HIGHLIGHT - main innovation):
   - Dual Gates (parallel):
     * IEG: Entropy-based reliability weighting (low entropy = high weight)
     * MIG: Learned importance gating
   - Adaptive Fusion: h = α·h_IEG + (1-α)·h_MIG
   - Key insight: "Reliability + Importance dual-path fusion"

4. Output (SIMPLIFIED):
   - Classifier → ReID Loss

NOTES:

- Draw backbone as a SINGLE gray box labeled "Shared ViT Backbone"
- SDTPS: Show cross-modal arrows (NIR/TIR global → RGB patches, etc.)
- DGAF: Show two parallel gates merging with learnable α
- Use color: RGB=red, NIR=green, TIR=blue
- Emphasize the cross-modal information flow

STYLE:
- NeurIPS 2024 tone
- Light background
- Clean arrows
- Mathematical symbols simplified

Generate the final diagram.
```

---

## 简化数据流

```
        ┌─────┐ ┌─────┐ ┌─────┐
        │ RGB │ │ NIR │ │ TIR │
        └──┬──┘ └──┬──┘ └──┬──┘
           └───────┼───────┘
                   ↓
        ┌─────────────────────┐
        │  Shared ViT Backbone │
        │  (Patch + Global)    │
        └─────────────────────┘
                   ↓
    ╔══════════════════════════════════╗
    ║           SDTPS Module           ║
    ║                                  ║
    ║   Cross-Modal Attention:         ║
    ║   RGB_patch ← NIR_global         ║
    ║   RGB_patch ← TIR_global         ║
    ║              ↓                   ║
    ║   Score Fusion → Top-K Select    ║
    ║              ↓                   ║
    ║   Token Aggregation              ║
    ╚══════════════════════════════════╝
                   ↓
    ╔══════════════════════════════════╗
    ║           DGAF Module            ║
    ║                                  ║
    ║   ┌─────────┐   ┌─────────┐      ║
    ║   │   IEG   │   │   MIG   │      ║
    ║   │(Entropy)│   │ (Gate)  │      ║
    ║   └────┬────┘   └────┬────┘      ║
    ║        └──────┬──────┘           ║
    ║               ↓                  ║
    ║   α·h_IEG + (1-α)·h_MIG          ║
    ╚══════════════════════════════════╝
                   ↓
        ┌─────────────────────┐
        │   Classifier → Loss  │
        └─────────────────────┘
```

---

## 核心创新点公式

### SDTPS: 跨模态引导的 Patch 选择

$$s_i = (1-2\beta) \cdot s_i^{pred} + \beta \cdot (s_i^{cross\_m2} + s_i^{cross\_m3} + 2 \cdot s_i^{self})$$

**创新**: 用其他模态的全局特征引导当前模态的 patch 选择

---

### DGAF: 双门控自适应融合

$$h_{fused} = \alpha \cdot h_{entropy} + (1-\alpha) \cdot h_{importance}$$

其中:
- **IEG**: $h_{entropy} = \sum_m \text{softmax}(\exp(-H_m/\tau)) \cdot h_m$ (低熵=高可靠性)
- **MIG**: $h_{importance} = \sum_m g_m \cdot h_m$, where $g = \sigma(\text{MLP}(\cdot))$

**创新**: 熵门控关注可靠性，重要性门控关注显著性，α 自适应平衡

---

## 模块关系图

```
                    Backbone Output
                          │
            ┌─────────────┼─────────────┐
            ↓             ↓             ↓
      RGB_patches    NIR_patches   TIR_patches
      RGB_global     NIR_global    TIR_global
            │             │             │
            │    ┌────────┴────────┐    │
            │    │  Cross-Modal    │    │
            └───→│   Attention     │←───┘
                 │  (SDTPS core)   │
                 └────────┬────────┘
                          ↓
                 Selected & Aggregated
                   Tokens (×3)
                          │
            ┌─────────────┼─────────────┐
            ↓             ↓             ↓
         RGB_enh      NIR_enh       TIR_enh
            │             │             │
            └─────────────┼─────────────┘
                          ↓
                 ┌────────┴────────┐
                 │      DGAF       │
                 │  ┌────┐ ┌────┐  │
                 │  │IEG │ │MIG │  │
                 │  └──┬─┘ └─┬──┘  │
                 │     └──┬──┘     │
                 │        α        │
                 └────────┬────────┘
                          ↓
                    Fused Feature
```

---

## 绘图要点

1. **Backbone 部分**: 用一个灰色方块简单表示，标注 "Shared ViT"
2. **SDTPS 模块**:
   - 突出跨模态箭头（虚线或不同颜色）
   - 显示 "Cross-Modal Attention" 字样
3. **DGAF 模块**:
   - 两个并行的门控块 (IEG 和 MIG)
   - 用 α 符号表示自适应融合
4. **颜色方案**: RGB=淡红, NIR=淡绿, TIR=淡蓝, 共享部分=灰色
