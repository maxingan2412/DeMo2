# DeMo Overall Architecture Diagram Prompt (V2)

This file contains the prompt for generating a publication-quality diagram of the **DeMo** model architecture, with emphasis on **SDTPS** and **DGAF** modules as the key innovations.

---

## Nano Banana Pro Prompt

```
You are an expert ML illustrator.
Draw a clean, NeurIPS/ICLR-style scientific figure using Nano Banana Pro.

GOAL:
Create a professional, publication-quality diagram showing the complete DeMo
architecture from input images to loss computation.
- NO dimension/shape annotations (no B, 128, 512, etc.)
- Only module names in boxes
- Only operation labels on arrows
- SDTPS and DGAF modules must be visually emphasized
- RGB, NIR, TIR are THREE PARALLEL BRANCHES with IDENTICAL operations (mirror structure)

CRITICAL RULE - THREE PARALLEL MODALITY STREAMS:
- RGB, NIR, TIR must be shown as THREE PARALLEL branches throughout the entire diagram
- Every operation applied to RGB must also show the same operation for NIR and TIR
- The three branches are visually symmetric/mirrored
- Use consistent colors: RGB (coral), NIR (mint), TIR (sky blue)

GLOBAL RULES:

- Flat, clean NeurIPS style (no gradients, no gloss, no shadows)
- Consistent thin line weights
- Professional pastel palette
- Rounded rectangles for blocks
- Arrows must clearly indicate data flow
- Only short labels on boxes and arrows
- Keep spacing clean and balanced
- NO DIMENSION ANNOTATIONS ANYWHERE

COLOR SCHEME:

- RGB stream: pastel coral (#FFB6C1)
- NIR stream: pastel mint (#98FB98)
- TIR stream: pastel sky blue (#87CEEB)
- Shared/General components: light gray (#D3D3D3)
- SDTPS module: pastel gold with dashed border (#FFE4B5)
- DGAF module: pastel lavender with dashed border (#E6E6FA)
- Loss functions: pastel red (#FFA07A)

LAYOUT:

- Main flow: top-to-bottom
- Three parallel modality streams (RGB, NIR, TIR) shown side by side
- Each modality goes through identical operations (mirrored structure)
- SDTPS and DGAF modules prominently displayed in the middle
- Two output branches (ORI and SDTPS) at the bottom
- Loss functions at the very bottom (training only, dashed border)

===== MODULE LIST (BOXES) =====

1. INPUT LAYER (Top) - Three Parallel Inputs:

   ┌───────────┐    ┌───────────┐    ┌───────────┐
   │ RGB Image │    │ NIR Image │    │ TIR Image │
   │  (coral)  │    │  (mint)   │    │ (sky blue)│
   └─────┬─────┘    └─────┬─────┘    └─────┬─────┘
         │                │                │
         └────────────────┼────────────────┘
                          │
                   ┌──────┴──────┐
                   │Camera Label │
                   └─────────────┘

2. SHARED BACKBONE (ViT-B-16) - Three Parallel Branches:

   ┌─────────────────────────────────────────────────────────────┐
   │                  Shared ViT-B-16 (CLIP)                     │
   │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐        │
   │  │ RGB Encoder │   │ NIR Encoder │   │ TIR Encoder │        │
   │  │   (coral)   │   │   (mint)    │   │ (sky blue)  │        │
   │  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘        │
   │         │                 │                 │                │
   │    Patch + SIE       Patch + SIE       Patch + SIE          │
   └─────────┼─────────────────┼─────────────────┼───────────────┘
             │                 │                 │
        ┌────┴────┐       ┌────┴────┐       ┌────┴────┐
        │RGB_patch│       │NIR_patch│       │TIR_patch│
        │RGB_glob │       │NIR_glob │       │TIR_glob │
        └────┬────┘       └────┬────┘       └────┬────┘

   Arrow label: "Patch Embed + SIE"

3. GLOBAL-LOCAL ENHANCEMENT - Three Parallel Reduce Operations:

   ┌─────────────────────────────────────────────────────────────┐
   │              Global-Local Enhancement                        │
   │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐        │
   │  │ rgb_reduce  │   │ nir_reduce  │   │ tir_reduce  │        │
   │  │   (coral)   │   │   (mint)    │   │ (sky blue)  │        │
   │  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘        │
   └─────────┼─────────────────┼─────────────────┼───────────────┘
             │                 │                 │

   Arrow labels: "Pool + Concat" -> each reduce module

4. **SDTPS MODULE** (Highlighted - Key Innovation) - Three Parallel Branches:

   ┌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┐
   ┊                         SDTPS                                ┊
   ┊        (Sparse-Dense Token-aware Patch Selection)            ┊
   ┊                   [pastel gold background]                   ┊
   ┊                                                              ┊
   ┊  ┌─────────────────────────────────────────────────────────┐ ┊
   ┊  │                    TokenSparse                          │ ┊
   ┊  │   ┌──────────┐    ┌──────────┐    ┌──────────┐          │ ┊
   ┊  │   │   RGB    │    │   NIR    │    │   TIR    │          │ ┊
   ┊  │   │ (coral)  │    │ (mint)   │    │(sky blue)│          │ ┊
   ┊  │   └────┬─────┘    └────┬─────┘    └────┬─────┘          │ ┊
   ┊  │        │               │               │                 │ ┊
   ┊  │        │<----Cross-Attn---->│<----Cross-Attn---->│       │ ┊
   ┊  │        │               │               │                 │ ┊
   ┊  └────────┼───────────────┼───────────────┼────────────────┘ ┊
   ┊           │ Score Fusion  │ Score Fusion  │ Score Fusion    ┊
   ┊           │ TopK          │ TopK          │ TopK            ┊
   ┊           v               v               v                  ┊
   ┊  ┌─────────────────────────────────────────────────────────┐ ┊
   ┊  │                  TokenAggregation                       │ ┊
   ┊  │   ┌──────────┐    ┌──────────┐    ┌──────────┐          │ ┊
   ┊  │   │   RGB    │    │   NIR    │    │   TIR    │          │ ┊
   ┊  │   │ (coral)  │    │ (mint)   │    │(sky blue)│          │ ┊
   ┊  │   └────┬─────┘    └────┬─────┘    └────┬─────┘          │ ┊
   ┊  └────────┼───────────────┼───────────────┼────────────────┘ ┊
   ┊           │ Softmax+BMM   │ Softmax+BMM   │ Softmax+BMM     ┊
   ┊           v               v               v                  ┊
   ┊     RGB_enhanced    NIR_enhanced    TIR_enhanced            ┊
   └╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┘
               │               │               │
               └───────────────┼───────────────┘
                               │

   Key arrow labels in SDTPS:
   - "Self-Attn" (within each modality)
   - "Cross-Attn" (between modalities, shown as horizontal arrows)
   - "Score Fusion"
   - "TopK"
   - "Softmax + BMM"

5. **DGAF MODULE** (Highlighted - Key Innovation) - Fuses Three Branches:

   ┌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┐
   ┊                          DGAF                                ┊
   ┊          (Dual-Gated Adaptive Fusion V3)                     ┊
   ┊                  [pastel lavender background]                ┊
   ┊                                                              ┊
   ┊  ┌─────────────────────────────────────────────────────────┐ ┊
   ┊  │                  Attention Pooling                      │ ┊
   ┊  │   ┌──────────┐    ┌──────────┐    ┌──────────┐          │ ┊
   ┊  │   │  Q_rgb   │    │  Q_nir   │    │  Q_tir   │          │ ┊
   ┊  │   │ (coral)  │    │ (mint)   │    │(sky blue)│          │ ┊
   ┊  │   └────┬─────┘    └────┬─────┘    └────┬─────┘          │ ┊
   ┊  │        │ Cross-Attn    │ Cross-Attn    │ Cross-Attn     │ ┊
   ┊  └────────┼───────────────┼───────────────┼────────────────┘ ┊
   ┊           v               v               v                  ┊
   ┊        h_rgb           h_nir           h_tir                ┊
   ┊           │               │               │                  ┊
   ┊           └───────────────┼───────────────┘                  ┊
   ┊                           │                                  ┊
   ┊              ┌────────────┴────────────┐                     ┊
   ┊              │                         │                     ┊
   ┊              v                         v                     ┊
   ┊  ┌───────────────────┐    ┌───────────────────┐              ┊
   ┊  │        IEG        │    │        MIG        │              ┊
   ┊  │  (Info Entropy    │    │   (Modality       │              ┊
   ┊  │      Gate)        │    │ Importance Gate)  │              ┊
   ┊  └─────────┬─────────┘    └─────────┬─────────┘              ┊
   ┊            │ Entropy                │ Sigmoid                ┊
   ┊            │ Weighted Sum           │ Gated Sum              ┊
   ┊            v                        v                        ┊
   ┊        h_entropy              h_importance                   ┊
   ┊            │                        │                        ┊
   ┊            └───────────┬────────────┘                        ┊
   ┊                        │ Adaptive Fusion                     ┊
   ┊                        v                                     ┊
   ┊  ┌─────────────────────────────────────────────────────────┐ ┊
   ┊  │              Adaptive Fusion (alpha)                    │ ┊
   ┊  └───────────────────────┬─────────────────────────────────┘ ┊
   ┊                          │ Residual + Concat                 ┊
   ┊                          v                                   ┊
   ┊                    [sdtps_feat]                              ┊
   └╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┘
                              │

   Key arrow labels in DGAF:
   - "Cross-Attn" (attention pooling)
   - "Entropy" (IEG computation)
   - "Sigmoid" (MIG gate)
   - "Weighted Sum" / "Gated Sum"
   - "Adaptive Fusion"
   - "Residual + Concat"

6. DUAL OUTPUT BRANCHES:

   From Global-Local (globals):         From DGAF:
             │                                │
        ┌────┴────┐                      ┌────┴────┐
        │RGB_glob │                      │         │
        │NIR_glob │                      │sdtps_feat│
        │TIR_glob │                      │         │
        └────┬────┘                      └────┬────┘
             │ Concat                         │
             v                                v
   ┌───────────────────┐          ┌───────────────────┐
   │    ORI Branch     │          │   SDTPS Branch    │
   │  ┌─────────────┐  │          │  ┌─────────────┐  │
   │  │  BatchNorm  │  │          │  │  BatchNorm  │  │
   │  └──────┬──────┘  │          │  └──────┬──────┘  │
   │         │ BN      │          │         │ BN      │
   │         v         │          │         v         │
   │  ┌─────────────┐  │          │  ┌─────────────┐  │
   │  │ Classifier  │  │          │  │ Classifier  │  │
   │  └──────┬──────┘  │          │  └──────┬──────┘  │
   └─────────┼─────────┘          └─────────┼─────────┘
             │ Linear                       │ Linear
             v                              v
        [ori_score]                   [sdtps_score]
        [ori_feat]                    [sdtps_feat]

7. LOSS FUNCTIONS (Training Only - Dashed Border):

   ┌╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴┐
   ┊                     Loss Functions                         ┊
   ┊                   [pastel red background]                  ┊
   ┊                                                            ┊
   ┊  ┌─────────────────┐              ┌─────────────────┐      ┊
   ┊  │    ORI Loss     │              │   SDTPS Loss    │      ┊
   ┊  │  ID + Triplet   │              │  ID + Triplet   │      ┊
   ┊  └────────┬────────┘              └────────┬────────┘      ┊
   ┊           │                                │               ┊
   ┊           └────────────┬───────────────────┘               ┊
   ┊                        │ Sum                               ┊
   ┊                        v                                   ┊
   ┊              ┌─────────────────┐                           ┊
   ┊              │   Total Loss    │                           ┊
   ┊              └─────────────────┘                           ┊
   └╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴┘

===== THREE PARALLEL BRANCHES EMPHASIS =====

IMPORTANT: Throughout the diagram, RGB/NIR/TIR must be shown as THREE PARALLEL PATHS:

1. Input: 3 image boxes side by side
2. Backbone: 3 encoder branches (same operations)
3. Global-Local: 3 reduce modules (rgb_reduce, nir_reduce, tir_reduce)
4. SDTPS TokenSparse: 3 branches with cross-modal connections between them
5. SDTPS TokenAggregation: 3 branches (same operations)
6. DGAF Attention Pooling: 3 query branches (Q_rgb, Q_nir, Q_tir)
7. After DGAF fusion: merged into single stream

Visual representation of parallel structure:

```
    RGB          NIR          TIR
     │            │            │
     ▼            ▼            ▼
  ┌─────┐     ┌─────┐     ┌─────┐
  │ Op  │     │ Op  │     │ Op  │   <- Same operation, 3 parallel boxes
  └──┬──┘     └──┬──┘     └──┬──┘
     │            │            │
     │<--- Cross-Modal --->│   <- Horizontal arrows show cross-modal interaction
     │            │            │
     ▼            ▼            ▼
```

===== VISUAL EMPHASIS =====

SDTPS Module Emphasis:
  - Dashed border (2px, gold color #FFE4B5)
  - Light gold background fill
  - Bold title "SDTPS"
  - Shows three parallel RGB/NIR/TIR branches clearly

DGAF Module Emphasis:
  - Dashed border (2px, lavender color #E6E6FA)
  - Light lavender background fill
  - Bold title "DGAF"
  - Shows convergence from 3 branches to 1

===== STYLE REQUIREMENTS =====

- NeurIPS 2024 / ICLR 2025 visual tone
- Very light background (#FAFAFA or white)
- Text center-aligned inside blocks
- Arrows short and clean with rounded corners
- Use consistent vertical spacing
- Three-column layout for RGB/NIR/TIR branches
- Show loss functions with dashed borders (training-only indicator)
- SDTPS and DGAF with dashed borders and highlight colors
- No dimension numbers anywhere
- No long sentences, only short labels

Generate the final diagram.
```

---

## Complete Flow Description (Text)

### Training Flow - Emphasizing Three Parallel Branches

```
                              INPUT
                                │
         ┌──────────────────────┼──────────────────────┐
         │                      │                      │
    RGB Image              NIR Image              TIR Image
    (coral)                (mint)                (sky blue)
         │                      │                      │
         └──────────────────────┼──────────────────────┘
                                │
                         Camera Label
                                │
         ╔══════════════════════╧══════════════════════╗
         ║         SHARED ViT-B-16 BACKBONE            ║
         ║  ┌─────────┐   ┌─────────┐   ┌─────────┐    ║
         ║  │   RGB   │   │   NIR   │   │   TIR   │    ║
         ║  │ Encoder │   │ Encoder │   │ Encoder │    ║
         ║  └────┬────┘   └────┬────┘   └────┬────┘    ║
         ╚═══════╪═════════════╪═════════════╪════════╝
                 │             │             │
            patch+glob    patch+glob    patch+glob
                 │             │             │
         ╔═══════╧═════════════╧═════════════╧════════╗
         ║         GLOBAL-LOCAL ENHANCEMENT           ║
         ║  ┌─────────┐   ┌─────────┐   ┌─────────┐   ║
         ║  │rgb_reduce│  │nir_reduce│  │tir_reduce│  ║
         ║  └────┬────┘   └────┬────┘   └────┬────┘   ║
         ╚═══════╪═════════════╪═════════════╪════════╝
                 │             │             │
    ╔════════════╧═════════════╧═════════════╧════════════╗
    ║    ╔═══════════════════════════════════════════╗    ║
    ║    ║              SDTPS (Innovation)           ║    ║
    ║    ║  ┌─────────────────────────────────────┐  ║    ║
    ║    ║  │           TokenSparse               │  ║    ║
    ║    ║  │  RGB ←─Cross─→ NIR ←─Cross─→ TIR    │  ║    ║
    ║    ║  │   │             │             │     │  ║    ║
    ║    ║  └───┼─────────────┼─────────────┼─────┘  ║    ║
    ║    ║      │ TopK        │ TopK        │ TopK   ║    ║
    ║    ║  ┌───┼─────────────┼─────────────┼─────┐  ║    ║
    ║    ║  │           TokenAggregation          │  ║    ║
    ║    ║  │  RGB           NIR           TIR    │  ║    ║
    ║    ║  │   │             │             │     │  ║    ║
    ║    ║  └───┼─────────────┼─────────────┼─────┘  ║    ║
    ║    ╚══════╪═════════════╪═════════════╪════════╝    ║
    ║           │             │             │             ║
    ║    ╔══════╧═════════════╧═════════════╧════════╗    ║
    ║    ║              DGAF (Innovation)            ║    ║
    ║    ║  ┌─────────────────────────────────────┐  ║    ║
    ║    ║  │         Attention Pooling           │  ║    ║
    ║    ║  │  Q_rgb        Q_nir        Q_tir    │  ║    ║
    ║    ║  │   │             │             │     │  ║    ║
    ║    ║  └───┼─────────────┼─────────────┼─────┘  ║    ║
    ║    ║      │             │             │        ║    ║
    ║    ║      └─────────────┼─────────────┘        ║    ║
    ║    ║                    │                      ║    ║
    ║    ║           ┌────────┴────────┐             ║    ║
    ║    ║           │                 │             ║    ║
    ║    ║        ┌──┴──┐           ┌──┴──┐          ║    ║
    ║    ║        │ IEG │           │ MIG │          ║    ║
    ║    ║        └──┬──┘           └──┬──┘          ║    ║
    ║    ║           └────────┬───────┘              ║    ║
    ║    ║                    │                      ║    ║
    ║    ║           ┌────────┴────────┐             ║    ║
    ║    ║           │ Adaptive Fusion │             ║    ║
    ║    ║           └────────┬────────┘             ║    ║
    ║    ╚════════════════════╪══════════════════════╝    ║
    ╚═════════════════════════╪═══════════════════════════╝
                              │
         ┌────────────────────┴────────────────────┐
         │                                         │
    ┌────┴─────┐                             ┌─────┴────┐
    │ORI Branch│                             │SDTPS Br. │
    │  BN+Cls  │                             │  BN+Cls  │
    └────┬─────┘                             └────┬─────┘
         │                                        │
         v                                        v
    [ori_score]                             [sdtps_score]
         │                                        │
         └────────────────┬───────────────────────┘
                          │
              ┌───────────┴───────────┐
              │    Loss Functions     │
              │  ORI Loss + SDTPS Loss│
              │      = Total Loss     │
              └───────────────────────┘
```

---

## Key Points for Three Parallel Branches

### Operations Applied to ALL THREE Modalities (Mirror Structure):

| Stage | RGB | NIR | TIR |
|-------|-----|-----|-----|
| Input | RGB Image | NIR Image | TIR Image |
| Backbone | RGB Encoder | NIR Encoder | TIR Encoder |
| Global-Local | rgb_reduce | nir_reduce | tir_reduce |
| TokenSparse | RGB sparse | NIR sparse | TIR sparse |
| TokenAggregation | RGB aggr | NIR aggr | TIR aggr |
| Attention Pooling | Q_rgb | Q_nir | Q_tir |
| After DGAF | → Merged into single sdtps_feat |

### Cross-Modal Interactions (Shown as Horizontal Arrows):

1. **TokenSparse**: Each modality's score computation uses attention to OTHER modalities' globals
   - RGB: attends to NIR_global, TIR_global
   - NIR: attends to RGB_global, TIR_global
   - TIR: attends to RGB_global, NIR_global

2. **DGAF IEG/MIG**: Considers all three modalities together for gating

---

## Arrow Label Summary

| Connection | Arrow Label |
|------------|-------------|
| Input → Backbone | Patch Embed + SIE |
| Backbone → Global-Local | Patch Tokens + Global |
| Global-Local → SDTPS | Pool + Concat |
| Within TokenSparse | Self-Attn, Cross-Attn, Score Fusion, TopK |
| TokenSparse → TokenAggregation | Selected + Extra |
| Within TokenAggregation | Softmax + BMM |
| SDTPS → DGAF | Enhanced Tokens |
| Within Attention Pooling | Cross-Attn |
| Pooling → IEG | Entropy |
| Pooling → MIG | Concat, Sigmoid |
| IEG + MIG → Fusion | Weighted Sum |
| Fusion → Output | Residual + Concat |
| Branches → Loss | ID Loss, Triplet Loss |

---

## Code Reference

**Main Model File**: `/home/maxingan/copyfromssd/workfromlocal/newdemo/DeMo2/modeling/make_model.py`

**SDTPS Implementation**: `/home/maxingan/copyfromssd/workfromlocal/newdemo/DeMo2/modeling/sdtps_complete.py`

**DGAF Implementation**: `/home/maxingan/copyfromssd/workfromlocal/newdemo/DeMo2/modeling/dual_gated_fusion.py`
