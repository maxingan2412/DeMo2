# DeMo Overall Model Architecture Diagram Prompt

This file contains the complete prompt for generating a publication-quality diagram of the **DeMo (Decoupled Feature-Based Mixture of Experts)** overall model architecture, including PIFE backbone, SDTPS, DGAF, and loss functions.

---

## Nano Banana Pro Prompt

```
You are an expert ML illustrator.
Draw a clean, NeurIPS/ICLR-style scientific figure using Nano Banana Pro.

GOAL:
Create a professional, publication-quality diagram showing the complete DeMo
architecture from input images to final loss computation.
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

- Main flow: left-to-right or top-to-bottom
- Three parallel modality streams (RGB, NIR, TIR) at the input
- Shared backbone for feature extraction
- SDTPS module processing tokens
- DGAF module for adaptive fusion
- Two output branches (ORI and SDTPS/DGAF)
- Loss functions at the bottom
- Clear separation between training and inference paths

MODULE LIST:

1. Input Layer:
   - RGB Image: (B, 3, 256, 128) - visible light
   - NIR Image: (B, 3, 256, 128) - near-infrared
   - TIR Image: (B, 3, 256, 128) - thermal infrared
   - Camera Label: (B,) - camera ID for SIE embedding

2. PIFE Backbone (Shared ViT-B-16):
   - Architecture: CLIP-pretrained ViT-B-16
   - Patch Embedding:
     - Patch size: 16 x 16
     - Image size: 256 x 128
     - Num patches: (256/16) x (128/16) = 16 x 8 = 128
   - Position Embedding: Learned positional encoding
   - Camera Embedding (SIE): cv_embed[cam_label] * sie_coe
   - Transformer Blocks: 12 layers
   - Output per modality:
     - Patch tokens (cash): (B, 128, 512) - local features
     - CLS token (global): (B, 512) - global feature

3. Global-Local Feature Enhancement (GLOBAL_LOCAL=True):
   - Local Pooling: AdaptiveAvgPool1d on patch tokens
     - RGB_local = pool(RGB_cash.permute(0,2,1)).squeeze(-1)  # (B, 512)
   - Feature Concatenation:
     - RGB_concat = concat([RGB_global, RGB_local], dim=-1)  # (B, 1024)
   - Feature Reduction:
     - rgb_reduce: LayerNorm(1024) -> Linear(1024->512) -> QuickGELU
     - RGB_global = rgb_reduce(RGB_concat)  # (B, 512)
   - Same for NIR and TIR modalities

4. SDTPS Module (Sparse-Dense Token-aware Patch Selection):
   - Input:
     - Patch tokens: RGB_cash, NI_cash, TI_cash (B, 128, 512)
     - Global features: RGB_global, NI_global, TI_global (B, 512)
   - Stage 1: TokenSparse (per modality)
     - Compute 4 scores:
       - s^{im}: Self-attention with own global (intra-modal)
       - s^{cross1}: Cross-attention with modality 2's global
       - s^{cross2}: Cross-attention with modality 3's global
       - s^p: MLP-predicted importance
     - Score Fusion:
       - s = (1-2*beta)*s^p + beta*(s^{cross1} + s^{cross2} + 2*s^{im})
     - Top-K Selection: Keep 64 patches (sparse_ratio=0.5)
     - Extra Token: Weighted sum of discarded patches
   - Stage 2: TokenAggregation
     - MLP weight generation: Linear(512->102->25)
     - Softmax normalization over source tokens
     - BMM aggregation: (B, 25, 64) @ (B, 64, 512) -> (B, 25, 512)
   - Output:
     - Enhanced tokens: (B, 26, 512) per modality (25 aggr + 1 extra)
     - Selection masks: (B, 128) per modality

5. DGAF Module (Dual-Gated Adaptive Fusion):
   - Input: SDTPS outputs (B, 26, 512) x 3 modalities
   - Attention Pooling (V3):
     - Learnable queries per modality
     - Cross-attention to aggregate tokens
     - Output: h_rgb, h_nir, h_tir (B, 512)
   - Information Entropy Gate (IEG):
     - Compute entropy H(h_m) per modality
     - Entropy-modulated scores: s_m = z_m * exp(-H_m/tau)
     - Softmax weights -> weighted sum
     - Output: h_entropy (B, 512)
   - Modality Importance Gate (MIG):
     - Concatenate: h_concat (B, 1536)
     - Gate network: Linear + LayerNorm + ReLU + Linear + Sigmoid
     - Gates: (B, 3) in range (0, 1)
     - Gated sum -> h_importance (B, 512)
   - Adaptive Fusion:
     - h_fused = alpha * h_entropy + (1-alpha) * h_importance
     - alpha: learnable parameter
   - Output Projection:
     - Residual enhancement per modality
     - Concatenate: sdtps_feat (B, 1536)

6. Original Feature Branch (ORI):
   - Simple Concatenation:
     - ori = concat([RGB_global, NI_global, TI_global], dim=-1)  # (B, 1536)
   - BN + Classifier:
     - bottleneck: BatchNorm1d(1536)
     - ori_global = bottleneck(ori)
     - classifier: Linear(1536 -> num_classes)
     - ori_score = classifier(ori_global)
   - Output:
     - ori_score: (B, num_classes) - logits
     - ori: (B, 1536) - features for triplet loss

7. SDTPS Feature Branch:
   - BN + Classifier:
     - bottleneck_sdtps: BatchNorm1d(1536)
     - classifier_sdtps: Linear(1536 -> num_classes)
     - sdtps_score = classifier_sdtps(bottleneck_sdtps(sdtps_feat))
   - Output:
     - sdtps_score: (B, num_classes) - logits
     - sdtps_feat: (B, 1536) - features for triplet loss

8. Loss Functions (Training Only):
   - ID Loss (Cross-Entropy with Label Smoothing):
     - For ORI branch: xent(ori_score, target)
     - For SDTPS branch: xent(sdtps_score, target)
     - Weight: MODEL.ID_LOSS_WEIGHT (default: 0.25)
   - Triplet Loss (Soft Triplet):
     - For ORI branch: triplet(ori, target)
     - For SDTPS branch: triplet(sdtps_feat, target)
     - Weight: MODEL.TRIPLET_LOSS_WEIGHT (default: 1.0)
   - Total Loss:
     - loss_ori = ID_LOSS_WEIGHT * id_loss_ori + TRIPLET_LOSS_WEIGHT * tri_loss_ori
     - loss_sdtps = SDTPS_LOSS_WEIGHT * (ID_LOSS_WEIGHT * id_loss_sdtps + TRIPLET_LOSS_WEIGHT * tri_loss_sdtps)
     - total_loss = loss_ori + loss_sdtps

9. Inference Output:
   - return_pattern=1: ori (B, 1536)
   - return_pattern=2: sdtps_feat (B, 1536)
   - return_pattern=3: concat([ori, sdtps_feat], dim=-1) (B, 3072)

NOTES:

- CRITICAL: Show three parallel streams for RGB/NIR/TIR throughout
- BACKBONE is SHARED across all three modalities (weight sharing)
- SDTPS processes each modality independently but with cross-modal guidance
- DGAF fuses the three modality streams into one representation
- TWO branches exist: ORI (baseline) and SDTPS (enhanced)
- During training, both branches contribute to loss
- During inference, features can be used separately or combined
- Color scheme:
  - RGB: pastel coral (#FFB6C1)
  - NIR: pastel mint (#98FB98)
  - TIR: pastel sky blue (#87CEEB)
  - Shared components: light gray
  - Loss: pastel red
  - Fusion outputs: pastel purple
- Show dimension annotations at key points
- Dashed lines for optional/conditional paths

DIMENSION ANNOTATIONS:

Input Images:
  RGB, NIR, TIR: (B, 3, 256, 128)

After Backbone:
  Patch tokens: (B, 128, 512) per modality
  Global tokens: (B, 512) per modality

After Global-Local:
  Enhanced globals: (B, 512) per modality

After SDTPS TokenSparse:
  Selected tokens: (B, 64, 512) per modality
  Extra token: (B, 1, 512) per modality

After SDTPS TokenAggregation:
  Aggregated: (B, 25, 512) per modality
  Final: (B, 26, 512) per modality

After DGAF:
  sdtps_feat: (B, 1536)

ORI Branch:
  ori: (B, 1536)
  ori_score: (B, num_classes)

SDTPS Branch:
  sdtps_feat: (B, 1536)
  sdtps_score: (B, num_classes)

Inference Output:
  Pattern 1: (B, 1536)
  Pattern 2: (B, 1536)
  Pattern 3: (B, 3072)

STYLE REQUIREMENTS:

- NeurIPS 2024 visual tone
- Very light background (#FAFAFA or white)
- Text left-aligned inside blocks
- Arrows short and clean
- Use consistent vertical spacing
- Group related components with subtle borders
- Show loss functions in a dedicated "Loss" box at bottom
- Indicate training-only components with dashed borders

Generate the final diagram.
```

---

## Complete Model Flow Description

### Training Flow

```
                              INPUT
                                |
         +----------------------+----------------------+
         |                      |                      |
    RGB Image              NIR Image              TIR Image
   (B,3,256,128)          (B,3,256,128)          (B,3,256,128)
         |                      |                      |
         +----------------------+----------------------+
                                |
                    [SHARED ViT-B-16 BACKBONE]
                    (CLIP-pretrained, SIE camera)
                                |
         +----------------------+----------------------+
         |                      |                      |
   RGB_cash, RGB_global   NI_cash, NI_global   TI_cash, TI_global
   (B,128,512) (B,512)   (B,128,512) (B,512)   (B,128,512) (B,512)
         |                      |                      |
         +----------------------+----------------------+
                                |
                    [GLOBAL-LOCAL Enhancement]
                    (Pool local + Concat + Reduce)
                                |
         +----------------------+----------------------+
         |                      |                      |
      RGB_global            NI_global             TI_global
       (B,512)               (B,512)               (B,512)
         |                      |                      |
    +----+----+            +----+----+            +----+----+
    |         |            |         |            |         |
 global     cash        global     cash        global     cash
    |         |            |         |            |         |
    |    +----+------------+----+----+------------+----+    |
    |    |                                             |    |
    |    +------------------SDTPS----------------------+    |
    |    |   TokenSparse: Cross-modal score fusion     |    |
    |    |   TokenAggregation: Learnable pooling       |    |
    |    +---------------------------------------------+    |
    |                          |                            |
    |           RGB_enhanced, NI_enhanced, TI_enhanced      |
    |                  (B, 26, 512) each                    |
    |                          |                            |
    |                       [DGAF]                          |
    |           (Dual-Gated Adaptive Fusion V3)             |
    |    Attention Pooling + IEG + MIG + Adaptive Fusion    |
    |                          |                            |
    |                    sdtps_feat                         |
    |                     (B, 1536)                         |
    |                          |                            |
    +--------------------------+----------------------------+
                               |
         +---------------------+---------------------+
         |                                           |
    [ORI Branch]                              [SDTPS Branch]
    concat globals                            DGAF output
         |                                           |
    bottleneck                               bottleneck_sdtps
    (BN1d, 1536)                             (BN1d, 1536)
         |                                           |
    classifier                               classifier_sdtps
    (1536->num_cls)                          (1536->num_cls)
         |                                           |
    ori_score                                sdtps_score
         |                                           |
         +---------------------+---------------------+
                               |
                    [LOSS COMPUTATION]
                               |
         +---------------------+---------------------+
         |                                           |
    ID_Loss(ori)                              ID_Loss(sdtps)
    Triplet_Loss(ori)                         Triplet_Loss(sdtps)
         |                                           |
    loss_ori = 0.25*ID + 1.0*Tri             loss_sdtps = 2.0 * (0.25*ID + 1.0*Tri)
         |                                           |
         +---------------------+---------------------+
                               |
                    total_loss = loss_ori + loss_sdtps
```

### Inference Flow

```
                    INPUT (same as training)
                              |
                      [BACKBONE]
                              |
                   [GLOBAL-LOCAL]
                              |
         +--------------------+--------------------+
         |                                         |
      globals                                   tokens
         |                                         |
         |                    +--------------------+
         |                    |
         |                 [SDTPS]
         |                    |
         |                 [DGAF]
         |                    |
    ori (B,1536)        sdtps_feat (B,1536)
         |                    |
         +--------------------+
                    |
           [return_pattern]
                    |
    ----------------+----------------
    |               |               |
 pattern=1      pattern=2      pattern=3
 ori (1536)   sdtps (1536)   concat (3072)
```

---

## Module Inventory Summary

| Component | Type | Parameters | Input Shape | Output Shape |
|-----------|------|------------|-------------|--------------|
| BACKBONE | ViT-B-16 | ~86M (shared) | (B, 3, 256, 128) | (B, 128, 512), (B, 512) |
| rgb_reduce | Sequential | 0.53M | (B, 1024) | (B, 512) |
| nir_reduce | Sequential | 0.53M | (B, 1024) | (B, 512) |
| tir_reduce | Sequential | 0.53M | (B, 1024) | (B, 512) |
| rgb_sparse | TokenSparse | 0.07M | (B, 128, 512) | (B, 64, 512) |
| rgb_aggr | TokenAggregation | 0.07M | (B, 64, 512) | (B, 25, 512) |
| nir_sparse | TokenSparse | 0.07M | (B, 128, 512) | (B, 64, 512) |
| nir_aggr | TokenAggregation | 0.07M | (B, 64, 512) | (B, 25, 512) |
| tir_sparse | TokenSparse | 0.07M | (B, 128, 512) | (B, 64, 512) |
| tir_aggr | TokenAggregation | 0.07M | (B, 64, 512) | (B, 25, 512) |
| dgaf (V3) | DGAF | 2.6M | 3x(B, 26, 512) | (B, 1536) |
| bottleneck | BN1d | 0.003M | (B, 1536) | (B, 1536) |
| classifier | Linear | 0.3M | (B, 1536) | (B, num_cls) |
| bottleneck_sdtps | BN1d | 0.003M | (B, 1536) | (B, 1536) |
| classifier_sdtps | Linear | 0.3M | (B, 1536) | (B, num_cls) |

**Total Trainable Parameters**: ~92M (including backbone fine-tuning)

---

## Loss Function Details

### ID Loss (Cross-Entropy with Label Smoothing)

```python
# Label Smoothing Cross-Entropy
xent = CrossEntropyLabelSmooth(num_classes=num_classes)

# For single score tensor:
ID_LOSS = xent(score, target)

# Loss weight from config:
weighted_id_loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS  # default: 0.25
```

### Triplet Loss (Soft Margin)

```python
# Soft Triplet Loss (no hard margin)
triplet = TripletLoss()  # when cfg.MODEL.NO_MARGIN = True

# For feature tensor:
TRI_LOSS = triplet(feat, target)[0]

# Loss weight from config:
weighted_tri_loss = cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS  # default: 1.0
```

### Combined Loss Computation

```python
# ORI branch loss
loss_ori = ID_LOSS_WEIGHT * id_loss_ori + TRIPLET_LOSS_WEIGHT * tri_loss_ori

# SDTPS branch loss (with higher weight)
loss_sdtps = SDTPS_LOSS_WEIGHT * (ID_LOSS_WEIGHT * id_loss_sdtps + TRIPLET_LOSS_WEIGHT * tri_loss_sdtps)
# SDTPS_LOSS_WEIGHT default: 2.0

# Total loss
total_loss = loss_ori + loss_sdtps
```

---

## Configuration Summary

### Model Configuration (from DeMo_SDTPS_DGAF_ablation.yml)

```yaml
MODEL:
  TRANSFORMER_TYPE: 'ViT-B-16'      # CLIP backbone
  STRIDE_SIZE: [16, 16]             # Patch size
  SIE_CAMERA: True                  # Camera embedding
  DIRECT: 1                         # Concatenate modalities
  ID_LOSS_WEIGHT: 0.25              # ID loss weight
  TRIPLET_LOSS_WEIGHT: 1.0          # Triplet loss weight
  GLOBAL_LOCAL: True                # Use local features

  # SDTPS Configuration
  USE_SDTPS: True
  SDTPS_SPARSE_RATIO: 0.7           # Keep 70% in sparse stage
  SDTPS_AGGR_RATIO: 0.5             # Aggregate to 50% of sparse
  SDTPS_BETA: 0.25                  # Score fusion weight
  SDTPS_USE_GUMBEL: False           # No Gumbel-Softmax
  SDTPS_LOSS_WEIGHT: 2.0            # SDTPS branch weight

  # DGAF Configuration
  USE_DGAF: True
  DGAF_VERSION: 'v3'                # Use V3 with attention pooling
  DGAF_TAU: 1.0                     # Entropy temperature
  DGAF_INIT_ALPHA: 0.5              # Initial balance
  DGAF_NUM_HEADS: 8                 # Attention heads
```

### Training Configuration

```yaml
SOLVER:
  BASE_LR: 0.00035                  # Learning rate
  MAX_EPOCHS: 50                    # Training epochs
  STEPS: [30, 40]                   # LR decay steps
  GAMMA: 0.1                        # LR decay factor
  WARMUP_ITERS: 10                  # Warmup epochs
  IMS_PER_BATCH: 64                 # Batch size
```

---

## Code Reference

**Main Model File**: `/home/maxingan/copyfromssd/workfromlocal/newdemo/DeMo2/modeling/make_model.py`

**Key Classes**:
- `DeMo` (Line 19-502): Main model class
- `build_transformer` in `meta_arch.py`: Backbone builder

**Module Files**:
- `/home/maxingan/copyfromssd/workfromlocal/newdemo/DeMo2/modeling/sdtps_complete.py`: SDTPS implementation
- `/home/maxingan/copyfromssd/workfromlocal/newdemo/DeMo2/modeling/dual_gated_fusion.py`: DGAF implementation
- `/home/maxingan/copyfromssd/workfromlocal/newdemo/DeMo2/modeling/meta_arch.py`: Backbone architecture
- `/home/maxingan/copyfromssd/workfromlocal/newdemo/DeMo2/modeling/clip/model.py`: CLIP ViT implementation

**Training/Inference**:
- `/home/maxingan/copyfromssd/workfromlocal/newdemo/DeMo2/engine/processor.py`: Training and inference loops
- `/home/maxingan/copyfromssd/workfromlocal/newdemo/DeMo2/layers/make_loss.py`: Loss function factory

**Configuration**:
- `/home/maxingan/copyfromssd/workfromlocal/newdemo/DeMo2/config/defaults.py`: All default parameters
- `/home/maxingan/copyfromssd/workfromlocal/newdemo/DeMo2/configs/RGBNT201/DeMo_SDTPS_DGAF_ablation.yml`: Ablation config

---

## Hierarchical Structure

```
Level 0: DeMo (Main Model)
    |
    +-- Level 1: BACKBONE (build_transformer)
    |       |
    |       +-- Level 2: ViT-B-16 (CLIP Visual Encoder)
    |       |       +-- Patch Embedding
    |       |       +-- Position Embedding
    |       |       +-- Camera Embedding (SIE)
    |       |       +-- 12 Transformer Blocks
    |       |       +-- Layer Norm
    |       |
    |       +-- Level 2: Output
    |               +-- CLS Token (global)
    |               +-- Patch Tokens (local)
    |
    +-- Level 1: Global-Local Enhancement
    |       |
    |       +-- Level 2: pool (AdaptiveAvgPool1d)
    |       +-- Level 2: rgb_reduce (LayerNorm + Linear + GELU)
    |       +-- Level 2: nir_reduce (LayerNorm + Linear + GELU)
    |       +-- Level 2: tir_reduce (LayerNorm + Linear + GELU)
    |
    +-- Level 1: SDTPS (MultiModalSDTPS)
    |       |
    |       +-- Level 2: RGB Branch
    |       |       +-- rgb_sparse (TokenSparse)
    |       |       +-- rgb_aggr (TokenAggregation)
    |       |
    |       +-- Level 2: NIR Branch
    |       |       +-- nir_sparse (TokenSparse)
    |       |       +-- nir_aggr (TokenAggregation)
    |       |
    |       +-- Level 2: TIR Branch
    |               +-- tir_sparse (TokenSparse)
    |               +-- tir_aggr (TokenAggregation)
    |
    +-- Level 1: DGAF (DualGatedAdaptiveFusionV3)
    |       |
    |       +-- Level 2: Attention Pooling
    |       +-- Level 2: Information Entropy Gate
    |       +-- Level 2: Modality Importance Gate
    |       +-- Level 2: Adaptive Fusion
    |       +-- Level 2: Output Projection
    |
    +-- Level 1: ORI Classification Head
    |       |
    |       +-- Level 2: bottleneck (BN1d)
    |       +-- Level 2: classifier (Linear)
    |
    +-- Level 1: SDTPS Classification Head
            |
            +-- Level 2: bottleneck_sdtps (BN1d)
            +-- Level 2: classifier_sdtps (Linear)
```

---

## SDTPS and DGAF Collaboration

### How They Work Together

1. **SDTPS Role**:
   - Selects important patches from 128 tokens
   - Uses cross-modal guidance (other modalities' globals)
   - Compresses tokens: 128 -> 64 -> 26 per modality
   - Produces enhanced token sequences

2. **DGAF Role**:
   - Receives SDTPS outputs (3 x (B, 26, 512))
   - Aggregates tokens into global features via attention
   - Assesses modality reliability (entropy) and importance (learned)
   - Produces single fused representation (B, 1536)

### Information Flow

```
SDTPS Output:
  RGB_enhanced (B, 26, 512) ----+
  NIR_enhanced (B, 26, 512) ----+----> DGAF ----> sdtps_feat (B, 1536)
  TIR_enhanced (B, 26, 512) ----+
```

### Why This Combination?

| Component | Strength | Limitation |
|-----------|----------|------------|
| SDTPS | Cross-modal token selection | Outputs separate per-modality |
| DGAF | Quality-aware fusion | Needs aggregated features |

Together: SDTPS provides cross-modal aware compressed tokens, DGAF adaptively fuses them based on quality and importance.
