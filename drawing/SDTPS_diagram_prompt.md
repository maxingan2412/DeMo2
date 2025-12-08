# SDTPS Module Architecture Diagram Prompt

This file contains the complete prompt for generating a publication-quality diagram of the **SDTPS (Sparse-Dense Token-aware Patch Selection)** module using Nano Banana Pro or similar visualization tools.

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

- Vertical top-to-bottom main flow for each modality
- Three parallel branches (RGB, NIR, TIR) side by side
- Cross-modal connections shown with dashed arrows
- Align components cleanly in straight lines
- Respect the module order exactly as listed

MODULE LIST:

1. Input(s):
   - RGB Patches: (B, 128, 512) - patch-level features from ViT backbone
   - NIR Patches: (B, 128, 512) - patch-level features from ViT backbone
   - TIR Patches: (B, 128, 512) - patch-level features from ViT backbone
   - RGB Global: (B, 512) - global CLS token feature
   - NIR Global: (B, 512) - global CLS token feature
   - TIR Global: (B, 512) - global CLS token feature

2. Score Computation (per modality, taking RGB as example):
   - Self-Attention Score s^{im}:
     - Input: RGB_patches (B, 128, 512), RGB_global (B, 512)
     - Operation: L2-normalize both, compute dot product
     - Output: s^{im} (B, 128) - intra-modal similarity
   - Cross-Attention Score s^{cross1}:
     - Input: RGB_patches (B, 128, 512), NIR_global (B, 512)
     - Operation: L2-normalize both, compute dot product
     - Output: s^{cross1} (B, 128) - cross-modal similarity with NIR
   - Cross-Attention Score s^{cross2}:
     - Input: RGB_patches (B, 128, 512), TIR_global (B, 512)
     - Operation: L2-normalize both, compute dot product
     - Output: s^{cross2} (B, 128) - cross-modal similarity with TIR
   - MLP Predictive Score s^p:
     - Input: RGB_patches (B, 128, 512)
     - Operation: MLP(512->128->1) + Sigmoid
     - Output: s^p (B, 128) - learned importance prediction

3. Core Architecture / Stages / Blocks:

   Stage 1: TokenSparse
   - Score Normalization:
     - Apply Min-Max normalization to s^{im}, s^{cross1}, s^{cross2}
     - Range: [0, 1]
   - Score Fusion:
     - Formula: s = (1-2*beta)*s^p + beta*(s^{cross1} + s^{cross2} + 2*s^{im})
     - Default beta = 0.25
     - Output: combined score s (B, 128)
   - Top-K Selection:
     - Sort scores descending
     - Keep top N_s = ceil(128 * sparse_ratio) = 64 patches
     - Output: select_tokens (B, 64, 512), keep_indices (B, 64)
   - Extra Token Generation:
     - Collect discarded patches: N - N_s = 64 patches
     - Compute softmax weights over discarded scores
     - Weighted sum: extra_token = sum(softmax(discarded_scores) * discarded_patches)
     - Output: extra_token (B, 1, 512)
   - (Optional) Gumbel-Softmax for differentiable selection during training:
     - Add Gumbel noise to scores
     - Straight-Through Estimator (STE) for hard selection with soft gradients

   Stage 2: TokenAggregation
   - MLP Weight Generation:
     - Input: select_tokens (B, 64, 512)
     - Network: LayerNorm -> Linear(512->102) -> GELU -> Linear(102->25)
     - Output: weight_logits (B, 64, 25)
   - Weight Transformation:
     - Transpose: (B, 64, 25) -> (B, 25, 64)
     - Scale: multiply by learnable scale parameter
   - Importance Integration (optional):
     - Add log(importance_weights) to weight logits
     - importance_weights from TokenSparse selection scores
   - Softmax Normalization:
     - Apply softmax over dim=2 (source tokens)
     - Ensures sum over source tokens = 1 for each target
     - Output: W (B, 25, 64), where W[b,j,:].sum() = 1
   - Matrix Multiplication:
     - aggr_tokens = bmm(W, select_tokens)
     - (B, 25, 64) @ (B, 64, 512) -> (B, 25, 512)

   Final Assembly:
   - Concatenate: [aggr_tokens, extra_token]
   - Output: enhanced_tokens (B, 26, 512) per modality

4. Special Mechanisms:

   Cross-Modal Guidance:
   - Each modality uses OTHER modalities' global features for cross-attention
   - RGB uses NIR_global and TIR_global
   - NIR uses RGB_global and TIR_global
   - TIR uses RGB_global and NIR_global
   - Purpose: select patches important across all modalities

   Three Parallel Branches:
   - Each modality (RGB, NIR, TIR) has independent:
     - TokenSparse module
     - TokenAggregation module
   - Shared: cross-modal global features for score computation

5. Output Head:
   - RGB_enhanced: (B, 26, 512)
   - NIR_enhanced: (B, 26, 512)
   - TIR_enhanced: (B, 26, 512)
   - rgb_mask: (B, 128) - binary selection mask
   - nir_mask: (B, 128) - binary selection mask
   - tir_mask: (B, 128) - binary selection mask

NOTES:

- CRITICAL: Show three parallel branches (RGB/NIR/TIR) processing simultaneously
- Show cross-modal connections as dashed lines from global features to other branches
- TokenSparse block should show 4 score sources converging to score fusion
- Score fusion formula: s = (1-2*beta)*s^p + beta*(s^{cross1} + s^{cross2} + 2*s^{im})
- Show dimension changes clearly: 128 -> 64 (TokenSparse) -> 25+1=26 (TokenAggregation)
- Extra token merges discarded information, shown as side branch
- TokenAggregation is a learned aggregation, not simple pooling
- Color suggestion: RGB=pastel coral, NIR=pastel mint, TIR=pastel sky blue
- Cross-modal arrows=dashed gray

DIMENSION ANNOTATIONS:

Input:
  Patches: (B, 128, 512)
  Global: (B, 512)

After TokenSparse:
  select_tokens: (B, 64, 512)
  extra_token: (B, 1, 512)
  Compression: 128 -> 64 (50% kept)

After TokenAggregation:
  aggr_tokens: (B, 25, 512)
  Compression: 64 -> 25 (39% of sparse, 19.5% of original)

Final Output:
  enhanced_tokens: (B, 26, 512) = [aggr_tokens; extra_token]
  Overall compression: 128 -> 26 (20.3% of original)

STYLE REQUIREMENTS:

- NeurIPS 2024 visual tone
- Very light background (#FAFAFA or white)
- Text left-aligned inside blocks
- Arrows short and clean
- Use consistent vertical spacing
- Mathematical formulas simplified to short labels
- Show tensor dimensions in small gray text near arrows

Generate the final diagram.
```

---

## Detailed Module Inventory

| Module Name | Type | Input Dimensions | Output Dimensions | Description |
|-------------|------|------------------|-------------------|-------------|
| Score Predictor MLP | nn.Sequential | (B, N, 512) | (B, N, 1) | Predicts patch importance |
| Min-Max Normalizer | Function | (B, N) | (B, N) | Normalizes scores to [0,1] |
| Score Fusion | Function | 4 x (B, N) | (B, N) | Combines all scores |
| Top-K Selector | Function | (B, N) | (B, N_s) indices | Selects top patches |
| Extra Token Aggregator | Function | (B, N-N_s, C) | (B, 1, C) | Weighted sum of discarded |
| Aggregation MLP | nn.Sequential | (B, N_s, 512) | (B, N_s, N_c) | Generates weight logits |
| Softmax Normalizer | Function | (B, N_c, N_s) | (B, N_c, N_s) | Row-wise normalization |
| BMM Aggregation | torch.bmm | (B, N_c, N_s), (B, N_s, C) | (B, N_c, C) | Weighted aggregation |

---

## Connection Relationships

```
RGB_patches -----> Self-Attention -----> s^{im}_rgb
    |                                        |
    +-------> Cross-Attention(NIR_global) -> s^{cross1}_rgb
    |                                        |
    +-------> Cross-Attention(TIR_global) -> s^{cross2}_rgb
    |                                        |
    +-------> MLP Predictor ---------------> s^p_rgb
                                             |
                                   [Score Fusion]
                                             |
                                    combined_score
                                             |
                              +-------+------+-------+
                              |              |       |
                          Top-K         Extra Token  |
                              |              |       |
                    select_tokens     extra_token    |
                              |              |       |
                    TokenAggregation         |       |
                              |              |       |
                       aggr_tokens           |       |
                              |              |       |
                              +------+-------+       |
                                     |               |
                             [Concatenate]           |
                                     |               |
                          RGB_enhanced (B,26,512)    |
                                                     |
                              (Repeat for NIR, TIR)--+
```

---

## Hierarchical Structure

```
Level 0: MultiModalSDTPS
    |
    +-- Level 1: RGB Branch
    |       |
    |       +-- Level 2: TokenSparse (rgb_sparse)
    |       |       +-- Score Predictor MLP
    |       |       +-- Score Normalization
    |       |       +-- Score Fusion
    |       |       +-- Top-K Selection
    |       |       +-- Extra Token Generation
    |       |
    |       +-- Level 2: TokenAggregation (rgb_aggr)
    |               +-- Weight MLP
    |               +-- Importance Integration
    |               +-- Softmax Normalization
    |               +-- BMM Aggregation
    |
    +-- Level 1: NIR Branch
    |       +-- (Same structure as RGB)
    |
    +-- Level 1: TIR Branch
            +-- (Same structure as RGB)
```

---

## Mathematical Formulas Summary

### TokenSparse Stage

1. **MLP Predictive Score**:
   $$s_i^p = \sigma(\text{MLP}(v_i)) = \sigma(W_2 \cdot \text{GELU}(W_1 \cdot v_i + b_1) + b_2)$$

2. **Self-Attention Score**:
   $$s_i^{im} = \frac{\langle v_i, g_{self} \rangle}{\|v_i\| \cdot \|g_{self}\|}$$

3. **Cross-Attention Scores**:
   $$s_i^{cross1} = \frac{\langle v_i, g_{m2} \rangle}{\|v_i\| \cdot \|g_{m2}\|}, \quad s_i^{cross2} = \frac{\langle v_i, g_{m3} \rangle}{\|v_i\| \cdot \|g_{m3}\|}$$

4. **Score Normalization** (Min-Max):
   $$\hat{s}_i = \frac{s_i - \min(s)}{\max(s) - \min(s) + \epsilon}$$

5. **Score Fusion**:
   $$s_i = (1-2\beta) \cdot s_i^p + \beta \cdot (s_i^{cross1} + s_i^{cross2} + 2 \cdot s_i^{im})$$

6. **Extra Token**:
   $$v_{extra} = \sum_{i \in \text{discarded}} \text{softmax}(s_i) \cdot v_i$$

### TokenAggregation Stage

7. **Weight Generation**:
   $$W_{logits} = \text{Linear}(\text{GELU}(\text{Linear}(\text{LayerNorm}(V_s))))$$

8. **Weight with Importance**:
   $$W' = W_{logits} + \log(\text{importance\_weights} + \epsilon)$$

9. **Normalized Weights**:
   $$W_{ij} = \frac{\exp(W'_{ij})}{\sum_k \exp(W'_{ik})}, \quad \text{s.t. } \sum_i W_{ji} = 1$$

10. **Aggregation**:
    $$\hat{v}_j = \sum_{i=1}^{N_s} W_{ji} \cdot v_i^s$$

---

## Data Flow Description

### Input Processing

1. **Receive Inputs**:
   - Three modality patch features: RGB_cash, NI_cash, TI_cash (B, 128, 512)
   - Three modality global features: RGB_global, NI_global, TI_global (B, 512)

### TokenSparse Processing (per modality)

2. **Compute Attention Scores**:
   - Self-attention: patch similarity to own global feature
   - Cross-attention 1: patch similarity to modality 2's global feature
   - Cross-attention 2: patch similarity to modality 3's global feature
   - MLP prediction: learned importance score

3. **Fuse Scores**:
   - Normalize each score to [0, 1]
   - Combine using weighted formula with beta parameter
   - Higher weight on self-attention (2x) emphasizes intra-modal consistency

4. **Select Top-K Patches**:
   - Sort combined scores descending
   - Keep top 64 patches (sparse_ratio = 0.5)
   - Generate binary selection mask

5. **Generate Extra Token**:
   - Collect 64 discarded patches
   - Apply softmax to their scores as weights
   - Compute weighted sum as summary of discarded info

### TokenAggregation Processing (per modality)

6. **Generate Aggregation Weights**:
   - Pass 64 selected tokens through MLP
   - Output: (B, 64, 25) weight logits

7. **Transform Weights**:
   - Transpose to (B, 25, 64)
   - Optionally add log(importance) for soft weighting

8. **Normalize and Aggregate**:
   - Softmax over source dimension (dim=2)
   - BMM: (B, 25, 64) @ (B, 64, 512) -> (B, 25, 512)

### Final Assembly

9. **Concatenate**:
   - Combine aggr_tokens (B, 25, 512) with extra_token (B, 1, 512)
   - Result: enhanced_tokens (B, 26, 512)

10. **Return**:
    - Three enhanced token sequences (one per modality)
    - Three selection masks (for visualization/analysis)

---

## Compression Ratio Summary

| Stage | Tokens Before | Tokens After | Compression Ratio |
|-------|---------------|--------------|-------------------|
| Input | 128 | 128 | 100% |
| TokenSparse | 128 | 64 + 1 (extra) | 50% |
| TokenAggregation | 64 | 25 | 39.1% |
| Final (with extra) | - | 26 | 20.3% of original |

**Overall Compression**: 128 tokens -> 26 tokens (79.7% reduction)

---

## Code Reference

**Source File**: `/home/maxingan/copyfromssd/workfromlocal/newdemo/DeMo2/modeling/sdtps_complete.py`

**Key Classes**:
- `TokenSparse` (Line 19-143): Handles score computation, fusion, and top-k selection
- `TokenAggregation` (Line 146-227): Handles learnable weight generation and aggregation
- `MultiModalSDTPS` (Line 230-452): Orchestrates all three modalities

**Configuration Parameters** (from `config/defaults.py`):
- `SDTPS_SPARSE_RATIO`: 0.5 (keep 50% in sparse stage)
- `SDTPS_AGGR_RATIO`: 0.4 (aggregate to 40% in aggregation stage)
- `SDTPS_BETA`: 0.25 (score fusion weight)
- `SDTPS_USE_GUMBEL`: False (disable Gumbel-Softmax by default)
- `SDTPS_GUMBEL_TAU`: 1.0 (temperature if Gumbel enabled)
