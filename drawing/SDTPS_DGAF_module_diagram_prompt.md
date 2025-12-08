# SDTPS and DGAF Module Architecture Diagram Prompt

This file contains the prompt for generating a publication-quality diagram of the **SDTPS** and **DGAF** modules in the DeMo framework.

---

## Nano Banana Pro Prompt

```
You are an expert ML illustrator.
Draw a clean, NeurIPS/ICLR-style scientific figure using Nano Banana Pro.

GOAL:
Create TWO sub-figures (a) and (b) showing the internal architecture of:
(a) SDTPS: Sparse-Dense Token-aware Patch Selection
(b) DGAF: Dual-Gated Adaptive Fusion V3

GLOBAL RULES:

- Flat, clean NeurIPS style (no gradients, no gloss, no shadows)
- Thin consistent line weights
- Professional pastel palette
- Rounded rectangles for modules (only module names inside, NO detailed text)
- Arrows indicate data flow
- Dimension annotations ON the arrows/lines (e.g., "(B,128,512)")
- Operation labels ON the arrows (e.g., "TopK", "Softmax", "BMM")
- Keep spacing clean and balanced
- Two sub-figures side by side or stacked

COLOR SCHEME:

- RGB related: pastel coral (#FFB6C1)
- NIR related: pastel mint (#98FB98)
- TIR related: pastel sky blue (#87CEEB)
- Shared components: light gray (#E0E0E0)
- Fusion outputs: pastel purple (#DDA0DD)
- Gates/attention: pastel yellow (#FFFACD)

========================================
SUB-FIGURE (a): SDTPS Module
========================================

LAYOUT: Top-to-bottom flow, showing ONE modality branch (RGB as example)
with cross-modal guidance from NIR and TIR globals.

INPUT BLOCKS (top):
- [RGB_cash] - coral - "(B, 128, 512)"
- [RGB_global] - coral - "(B, 512)"
- [NI_global] - mint - "(B, 512)" (dashed border, cross-modal input)
- [TI_global] - sky blue - "(B, 512)" (dashed border, cross-modal input)

STAGE 1: TokenSparse
- Four parallel score computation paths:

  Path 1 (self-attention):
  [RGB_cash] + [RGB_global] --"Cross-Attn"--> [s^im]

  Path 2 (cross-modal 1):
  [RGB_cash] + [NI_global] --"Cross-Attn"--> [s^cross1]

  Path 3 (cross-modal 2):
  [RGB_cash] + [TI_global] --"Cross-Attn"--> [s^cross2]

  Path 4 (MLP):
  [RGB_cash] --"MLP + Sigmoid"--> [s^p]

- Score fusion:
  [s^im], [s^cross1], [s^cross2], [s^p] --"Weighted Sum"--> [Score]
  Arrow label: "s = (1-2B)s^p + B(s^cross1 + s^cross2 + 2s^im)"

- Selection:
  [Score] --"TopK (K=64)"--> [Selection Mask] "(B, 128)"

- Two output branches:
  [RGB_cash] + [Selection Mask] --"Gather Top-64"--> [Selected Tokens] "(B, 64, 512)"
  [RGB_cash] + [Selection Mask] --"Weighted Sum of Discarded"--> [Extra Token] "(B, 1, 512)"

STAGE 2: TokenAggregation
- [Selected Tokens] --"MLP"--> [Weight Logits] "(B, 64, 25)"
- [Weight Logits] --"Transpose"--> "(B, 25, 64)"
- --"Softmax"--> [Aggregation Weights] "(B, 25, 64)"
- [Aggregation Weights] + [Selected Tokens] --"BMM"--> [Aggregated Tokens] "(B, 25, 512)"

OUTPUT:
- [Aggregated Tokens] + [Extra Token] --"Concat"--> [RGB_enhanced] "(B, 26, 512)"

NOTES ON DIAGRAM:
- Show cross-modal globals (NI, TI) with dashed borders
- Label beta on the weighted sum arrow
- Show dimension changes clearly
- Indicate "x3 Modalities" note at bottom

========================================
SUB-FIGURE (b): DGAF Module (V3)
========================================

LAYOUT: Top-to-bottom flow with two parallel branches merging at the end.

INPUT BLOCKS (top):
- [RGB_enhanced] - coral - "(B, 26, 512)"
- [NI_enhanced] - mint - "(B, 26, 512)"
- [TI_enhanced] - sky blue - "(B, 26, 512)"

SECTION 1: Attention Pooling
- Three parallel paths:
  [RGB_enhanced] + [Q_rgb] --"Cross-Attn"--> [h_rgb] "(B, 512)"
  [NI_enhanced] + [Q_nir] --"Cross-Attn"--> [h_nir] "(B, 512)"
  [TI_enhanced] + [Q_tir] --"Cross-Attn"--> [h_tir] "(B, 512)"
- Note: Q_rgb, Q_nir, Q_tir are learnable query tokens "(1, 1, 512)"

SECTION 2: Dual Gates (two parallel branches)

BRANCH A - Information Entropy Gate (IEG):
- [h_rgb] --"Entropy"--> [H_rgb] "(B,)"
- [h_nir] --"Entropy"--> [H_nir] "(B,)"
- [h_tir] --"Entropy"--> [H_tir] "(B,)"
- [h_*] --"Linear"--> [z_*] "(B,)"
- [z_*, H_*] --"s = z * exp(-H/tau)"--> [entropy_scores] "(B, 3)"
- [entropy_scores] --"Softmax"--> [entropy_weights] "(B, 3)"
- [h_rgb, h_nir, h_tir] + [entropy_weights] --"Weighted Sum"--> [h_entropy] "(B, 512)"

BRANCH B - Modality Importance Gate (MIG):
- [h_rgb, h_nir, h_tir] --"Concat"--> [h_concat] "(B, 1536)"
- [h_concat] --"Linear + LayerNorm + ReLU"--> "(B, 512)"
- --"Linear + Sigmoid"--> [gates] "(B, 3)"
- [h_rgb, h_nir, h_tir] * [gates] --"Element-wise Multiply"--> [gated features]
- --"Sum"--> [h_importance] "(B, 512)"

SECTION 3: Adaptive Fusion
- [h_entropy] + [h_importance] --"alpha * h_e + (1-alpha) * h_i"--> [h_fused] "(B, 512)"
- Note: alpha is learnable parameter

SECTION 4: Output Projection
- [h_fused] --"Linear + LayerNorm"--> [h_enhance] "(B, 512)"
- [h_rgb] + [h_enhance] --"Residual Add"--> [h_rgb_out]
- [h_nir] + [h_enhance] --"Residual Add"--> [h_nir_out]
- [h_tir] + [h_enhance] --"Residual Add"--> [h_tir_out]
- [h_rgb_out, h_nir_out, h_tir_out] --"Concat"--> [sdtps_feat] "(B, 1536)"

OUTPUT:
- [sdtps_feat] - purple - "(B, 1536)"

STYLE REQUIREMENTS:

- NeurIPS 2024 visual tone
- White or very light background (#FAFAFA)
- Module boxes: only contain module/operation names
- All dimension info goes on arrows/connections
- All operation descriptions go on arrows
- Use consistent vertical/horizontal spacing
- Group related components with subtle dotted borders
- Show learnable parameters with small annotation boxes
- Color-code modality flows throughout

ARROW ANNOTATIONS FORMAT:
- Data flow arrows: thin solid black
- Dimension label: small text above/below arrow "(B, N, C)"
- Operation label: italic text on arrow "Softmax", "BMM", etc.
- Parameter labels: small boxes near operations "beta=0.25", "alpha"

Generate the final diagram with both sub-figures.
```

---

## Module Inventory

### SDTPS Module

| Component | Type | Input | Output | Description |
|-----------|------|-------|--------|-------------|
| CrossModalAttention | Attention | (B,N,C), (B,C) | (B,N) | Q-K cross-attention score |
| TokenSparse | Selection | (B,128,512) | (B,64,512), (B,1,512) | Top-K selection + extra token |
| TokenAggregation | Pooling | (B,64,512) | (B,25,512) | Learnable aggregation |

### DGAF Module (V3)

| Component | Type | Input | Output | Description |
|-----------|------|-------|--------|-------------|
| Attention Pooling | Attention | (B,26,512) | (B,512) | Query-based pooling |
| InformationEntropyGate | Gate | 3x(B,512) | (B,512) | Entropy-weighted fusion |
| ModalityImportanceGate | Gate | 3x(B,512) | (B,512) | Learned gating |
| Adaptive Fusion | Fusion | 2x(B,512) | (B,512) | Alpha-weighted combination |
| Output Projection | Linear | (B,512) | (B,1536) | Residual enhancement + concat |

---

## Connection Relationships

### SDTPS Connections

```
RGB_cash (B,128,512) ──┬──[Cross-Attn w/ RGB_global]──> s^im (B,128)
                       ├──[Cross-Attn w/ NI_global]───> s^cross1 (B,128)
                       ├──[Cross-Attn w/ TI_global]───> s^cross2 (B,128)
                       └──[MLP + Sigmoid]─────────────> s^p (B,128)
                                                            │
                                          [Weighted Sum]<───┴───
                                                │
                                          Score (B,128)
                                                │
                              ┌─────[TopK K=64]─┴─[Weighted Sum Discarded]─┐
                              │                                            │
                    Selected (B,64,512)                           Extra (B,1,512)
                              │                                            │
                        [TokenAggr]                                        │
                              │                                            │
                    Aggregated (B,25,512) ─────────[Concat]────────────────┘
                                                      │
                                            RGB_enhanced (B,26,512)
```

### DGAF Connections

```
RGB_enhanced ────[Attn Pool w/ Q_rgb]────> h_rgb ─┬─[IEG]──> h_entropy ──┐
(B,26,512)                                (B,512) │                      │
                                                  └─[MIG]──> h_importance┤
                                                                         │
NI_enhanced ─────[Attn Pool w/ Q_nir]────> h_nir ─┬─[IEG]                │
(B,26,512)                                (B,512) └─[MIG]                │
                                                                         │
TI_enhanced ─────[Attn Pool w/ Q_tir]────> h_tir ─┬─[IEG]     [Adaptive Fusion]
(B,26,512)                                (B,512) └─[MIG]          │
                                                               h_fused (B,512)
                                                                   │
                                                           [Output Proj]
                                                                   │
                                                     ┌─────────────┼─────────────┐
                                                     │             │             │
                                              h_rgb + h_e   h_nir + h_e   h_tir + h_e
                                                     │             │             │
                                                     └──────[Concat]─────────────┘
                                                                   │
                                                           sdtps_feat (B,1536)
```

---

## Key Formulas

### SDTPS Score Fusion
```
s = (1 - 2*beta) * s^p + beta * (s^cross1 + s^cross2 + 2 * s^im)
```
where beta = 0.25 (default)

### DGAF Information Entropy Gate
```
H(h_m) = -sum(p * log(p))           # Feature entropy
s_m = z_m * exp(-H_m / tau)          # Entropy-modulated score
weights = softmax([s_rgb, s_nir, s_tir])
h_entropy = sum(weights_m * h_m)
```

### DGAF Modality Importance Gate
```
gates = sigmoid(Linear(ReLU(LayerNorm(Linear(concat([h_rgb, h_nir, h_tir]))))))
h_importance = sum(gates_m * h_m)
```

### DGAF Adaptive Fusion
```
h_fused = alpha * h_entropy + (1 - alpha) * h_importance
```
where alpha is learnable (initialized to 0.5)

---

## Dimension Flow Summary

### SDTPS
```
Input:   RGB_cash (B, 128, 512), RGB_global (B, 512)
         NI_global (B, 512), TI_global (B, 512)  [cross-modal guidance]

Stage 1: TokenSparse
         Score computation: (B, 128)
         Selection: Top-64 -> Selected (B, 64, 512)
         Discarded weighted sum: Extra (B, 1, 512)

Stage 2: TokenAggregation
         MLP weights: (B, 64, 25) -> transpose -> (B, 25, 64)
         BMM: (B, 25, 64) @ (B, 64, 512) -> (B, 25, 512)

Output:  Concat: (B, 25, 512) + (B, 1, 512) -> (B, 26, 512)
```

### DGAF V3
```
Input:   RGB_enhanced, NI_enhanced, TI_enhanced: 3 x (B, 26, 512)

Attention Pooling:
         Queries: Q_rgb, Q_nir, Q_tir (1, 1, 512) learnable
         Output: h_rgb, h_nir, h_tir: 3 x (B, 512)

IEG Branch:
         Entropy: H_m (B,)
         Scores: s_m (B,) -> weights (B, 3)
         Output: h_entropy (B, 512)

MIG Branch:
         Concat: (B, 1536) -> Gates: (B, 3)
         Output: h_importance (B, 512)

Fusion:  h_fused = alpha * h_entropy + (1-alpha) * h_importance (B, 512)

Output:  Residual: h_m + enhance -> concat -> (B, 1536)
```

---

## Code Reference

**SDTPS Implementation**: `/home/maxingan/copyfromssd/workfromlocal/newdemo/DeMo2/modeling/sdtps_complete.py`
- `CrossModalAttention` (Line 21-118): Multi-head cross-attention for score computation
- `TokenSparse` (Line 121-245): Top-K selection with Gumbel-Softmax support
- `TokenAggregation` (Line 248-329): Learnable BMM-based aggregation
- `MultiModalSDTPS` (Line 332-605): Full multi-modal SDTPS module

**DGAF Implementation**: `/home/maxingan/copyfromssd/workfromlocal/newdemo/DeMo2/modeling/dual_gated_fusion.py`
- `InformationEntropyGate` (Line 34-116): Entropy-based reliability weighting
- `ModalityImportanceGate` (Line 119-182): Learned importance gating
- `DualGatedAdaptiveFusionV3` (Line 558-742): Full DGAF V3 with attention pooling

---

## Simplified ASCII Diagram

### SDTPS (Single Modality)

```
                        RGB_cash               NI_global    TI_global
                       (B,128,512)              (B,512)      (B,512)
                            │                      │             │
    RGB_global ─────────────┼──────────────────────┼─────────────┤
     (B,512)                │                      │             │
         │                  │                      │             │
         ▼                  ▼                      ▼             ▼
    ┌─────────┐        ┌─────────┐           ┌─────────┐   ┌─────────┐
    │  Self   │        │  MLP    │           │ Cross-1 │   │ Cross-2 │
    │  Attn   │        │ Sigmoid │           │  Attn   │   │  Attn   │
    └────┬────┘        └────┬────┘           └────┬────┘   └────┬────┘
         │                  │                      │             │
       s^im               s^p                  s^cross1     s^cross2
         │                  │                      │             │
         └──────────────────┴──────────────────────┴─────────────┘
                                    │
                            [Weighted Sum]
                                    │
                               Score (B,128)
                                    │
                              ┌─────┴─────┐
                              │   TopK    │
                              │   K=64    │
                              └─────┬─────┘
                        ┌───────────┴───────────┐
                        │                       │
               Selected Tokens         Weighted Sum Discarded
                (B,64,512)                  (B,1,512)
                        │                       │
               ┌────────┴────────┐              │
               │ TokenAggregation│              │
               │  MLP -> BMM     │              │
               └────────┬────────┘              │
                        │                       │
                  Aggregated                    │
                 (B,25,512)                     │
                        │                       │
                        └───────────┬───────────┘
                                    │
                               [Concat]
                                    │
                            RGB_enhanced
                             (B,26,512)
```

### DGAF V3

```
    RGB_enhanced         NI_enhanced         TI_enhanced
     (B,26,512)          (B,26,512)          (B,26,512)
         │                    │                    │
         ▼                    ▼                    ▼
    ┌─────────┐          ┌─────────┐          ┌─────────┐
    │  Attn   │          │  Attn   │          │  Attn   │
    │  Pool   │          │  Pool   │          │  Pool   │
    │(Q_rgb)  │          │(Q_nir)  │          │(Q_tir)  │
    └────┬────┘          └────┬────┘          └────┬────┘
         │                    │                    │
       h_rgb               h_nir                h_tir
      (B,512)             (B,512)              (B,512)
         │                    │                    │
         ├────────────────────┼────────────────────┤
         │                    │                    │
         ▼                    ▼                    ▼
    ┌─────────────────────────────────────────────────┐
    │                    IEG Branch                    │
    │  Entropy -> z*exp(-H/tau) -> Softmax -> WeightedSum │
    └────────────────────────┬────────────────────────┘
                             │
                        h_entropy (B,512)
                             │
         ├────────────────────┼────────────────────┤
         │                    │                    │
         ▼                    ▼                    ▼
    ┌─────────────────────────────────────────────────┐
    │                    MIG Branch                    │
    │    Concat -> Linear -> LN -> ReLU -> Sigmoid     │
    │                   -> Gated Sum                   │
    └────────────────────────┬────────────────────────┘
                             │
                      h_importance (B,512)
                             │
    h_entropy ───────────────┼───────────────────────────
                             │
                    ┌────────┴────────┐
                    │ Adaptive Fusion │
                    │ alpha*e+(1-a)*i │
                    └────────┬────────┘
                             │
                        h_fused (B,512)
                             │
                    ┌────────┴────────┐
                    │  Output Proj    │
                    │ Linear + LN     │
                    └────────┬────────┘
                             │
                        h_enhance (B,512)
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
    h_rgb + h_e        h_nir + h_e         h_tir + h_e
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                        [Concat]
                             │
                      sdtps_feat (B,1536)
```
