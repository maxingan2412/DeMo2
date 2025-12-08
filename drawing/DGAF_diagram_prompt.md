# DGAF Module Architecture Diagram Prompt

This file contains the complete prompt for generating a publication-quality diagram of the **DGAF (Dual-Gated Adaptive Fusion)** module using Nano Banana Pro or similar visualization tools.

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

- Main vertical flow from top to bottom
- Two parallel branches (IEG and MIG) in the middle section
- Three modality inputs (RGB, NIR, TIR) at the top
- Convergence point for adaptive fusion at bottom
- Align components cleanly in straight lines
- Respect the module order exactly as listed

MODULE LIST:

1. Input(s):
   - RGB_tokens: (B, K+1, 512) - SDTPS enhanced tokens (K aggregated + 1 extra)
   - NIR_tokens: (B, K+1, 512) - SDTPS enhanced tokens
   - TIR_tokens: (B, K+1, 512) - SDTPS enhanced tokens
   Note: K = 25 (typical setting), so input is (B, 26, 512)

2. Attention Pooling Stage (V3 only):
   - Purpose: Aggregate token sequences into global features
   - Components:
     - rgb_query: Learnable query parameter (1, 1, 512)
     - nir_query: Learnable query parameter (1, 1, 512)
     - tir_query: Learnable query parameter (1, 1, 512)
     - attn_pool: MultiheadAttention (embed_dim=512, num_heads=8)
     - attn_norm: LayerNorm(512)
   - Operation per modality:
     - query_expanded = query.expand(B, 1, 512)
     - pooled, _ = attn_pool(query_expanded, tokens, tokens)  # Cross-attention
     - h_m = attn_norm(pooled.squeeze(1))
   - Output: h_rgb, h_nir, h_tir each (B, 512)

3. Information Entropy Gate (IEG):
   - Purpose: Weight modalities by reliability (low entropy = high reliability)
   - Step 3.1 - Compute Entropy:
     - For each h_m (B, C):
       - feat_abs = |h_m| + 1e-8
       - prob = feat_abs / sum(feat_abs, dim=-1)
       - H_m = -sum(prob * log(prob + 1e-8), dim=-1)  # (B,)
   - Step 3.2 - Compute Attention Logits:
     - entropy_proj: Linear(512 -> 512)
     - z_m = entropy_proj(h_m).mean(dim=-1)  # (B,)
   - Step 3.3 - Entropy-Modulated Scores:
     - score_m = z_m * exp(-H_m / tau)  # Low entropy -> high score
     - tau = 1.0 (configurable temperature)
   - Step 3.4 - Softmax Weights:
     - scores = stack([score_rgb, score_nir, score_tir], dim=-1)  # (B, 3)
     - entropy_weights = softmax(scores, dim=-1)  # (B, 3)
   - Step 3.5 - Weighted Fusion:
     - h_entropy = w_rgb * h_rgb + w_nir * h_nir + w_tir * h_tir  # (B, 512)
   - Output: h_entropy (B, 512)

4. Modality Importance Gate (MIG):
   - Purpose: Learn sample-specific modality importance
   - Step 4.1 - Concatenate Features:
     - h_concat = concat([h_rgb, h_nir, h_tir], dim=-1)  # (B, 1536)
   - Step 4.2 - Gate Network:
     - gate_net: Sequential(
         Linear(1536 -> 512),
         LayerNorm(512),
         ReLU,
         Linear(512 -> 3),
         Sigmoid
       )
     - gates = gate_net(h_concat)  # (B, 3), values in (0, 1)
   - Step 4.3 - Gated Scaling:
     - h_rgb_gated = gates[:, 0:1] * h_rgb  # (B, 512)
     - h_nir_gated = gates[:, 1:2] * h_nir  # (B, 512)
     - h_tir_gated = gates[:, 2:3] * h_tir  # (B, 512)
   - Step 4.4 - Sum Fusion:
     - h_importance = h_rgb_gated + h_nir_gated + h_tir_gated  # (B, 512)
   - Output: h_importance (B, 512)

5. Adaptive Fusion:
   - Learnable Parameter:
     - _alpha: nn.Parameter (initialized to 0.5)
     - alpha = sigmoid(_alpha)  # Constrained to [0, 1]
   - Fusion Formula:
     - h_fused = alpha * h_entropy + (1 - alpha) * h_importance  # (B, 512)
   - Output: h_fused (B, 512)

6. Output Projection:
   - For output_dim = 3 * feat_dim = 1536:
     - modal_enhance: Sequential(Linear(512 -> 512), LayerNorm(512))
     - h_enhance = modal_enhance(h_fused)  # (B, 512)
     - Residual Enhancement:
       - h_rgb_out = h_rgb + h_enhance
       - h_nir_out = h_nir + h_enhance
       - h_tir_out = h_tir + h_enhance
     - Concatenation:
       - output = concat([h_rgb_out, h_nir_out, h_tir_out], dim=-1)  # (B, 1536)
   - Alternative for output_dim = feat_dim = 512:
     - output_proj: Sequential(Linear(512 -> 512), LayerNorm(512))
     - output = output_proj(h_fused)  # (B, 512)
   - Output: (B, 1536) for concat mode, (B, 512) for single mode

NOTES:

- CRITICAL: Show two parallel branches for IEG (entropy-based) and MIG (learned gates)
- IEG branch emphasizes reliability: use math symbol H for entropy
- MIG branch emphasizes importance: show gate values g_rgb, g_nir, g_tir
- Alpha parameter balances the two branches: show as learnable scalar
- Attention Pooling (V3): show learnable query attending to tokens
- Color suggestion:
  - IEG branch: pastel blue (reliability/entropy)
  - MIG branch: pastel orange (learned importance)
  - Fusion: pastel purple (combination)
  - RGB/NIR/TIR: coral/mint/sky blue (consistent with SDTPS)
- Show mathematical formulas compactly near relevant blocks

DIMENSION ANNOTATIONS:

Input (from SDTPS):
  RGB_tokens, NIR_tokens, TIR_tokens: (B, 26, 512)

After Attention Pooling:
  h_rgb, h_nir, h_tir: (B, 512)

IEG Branch:
  Entropy H_m: (B,) per modality
  Scores: (B, 3)
  Weights: (B, 3), sum = 1
  h_entropy: (B, 512)

MIG Branch:
  h_concat: (B, 1536)
  gates: (B, 3), values in (0, 1)
  h_importance: (B, 512)

After Fusion:
  alpha: scalar in [0, 1]
  h_fused: (B, 512)

Final Output:
  output: (B, 1536) = concat of 3 enhanced modalities

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
| rgb_query | nn.Parameter | - | (1, 1, 512) | Learnable query for RGB attention pooling |
| nir_query | nn.Parameter | - | (1, 1, 512) | Learnable query for NIR attention pooling |
| tir_query | nn.Parameter | - | (1, 1, 512) | Learnable query for TIR attention pooling |
| attn_pool | nn.MultiheadAttention | Q:(B,1,C), K/V:(B,K+1,C) | (B, 1, C) | Cross-attention pooling |
| attn_norm | nn.LayerNorm | (B, 512) | (B, 512) | Normalization after pooling |
| entropy_proj | nn.Linear | (B, 512) | (B, 512) | Project features for entropy calculation |
| gate_net | nn.Sequential | (B, 1536) | (B, 3) | Generate modality gates |
| _alpha | nn.Parameter | - | scalar | Learnable balance parameter |
| modal_enhance | nn.Sequential | (B, 512) | (B, 512) | Enhancement projection |

---

## Connection Relationships

```
RGB_tokens (B, 26, 512)  NIR_tokens (B, 26, 512)  TIR_tokens (B, 26, 512)
         |                        |                        |
    [rgb_query]              [nir_query]              [tir_query]
         |                        |                        |
    [Attn Pool]              [Attn Pool]              [Attn Pool]
         |                        |                        |
    h_rgb (B,512)            h_nir (B,512)            h_tir (B,512)
         |                        |                        |
         +------------------------+------------------------+
                                  |
         +------------------------+------------------------+
         |                                                 |
         v                                                 v
  [Information Entropy Gate]                    [Modality Importance Gate]
         |                                                 |
    Compute Entropy H_m                            Concatenate h_concat
    for each modality                                    (B, 1536)
         |                                                 |
    Entropy-Modulated Scores                        Gate Network
    score_m = z_m * exp(-H/tau)                     gates = sigmoid(MLP)
         |                                                 |
    Softmax Weights                                 Gated Scaling
    entropy_weights (B, 3)                          h_m * g_m
         |                                                 |
    Weighted Sum                                    Sum Fusion
    h_entropy (B, 512)                              h_importance (B, 512)
         |                                                 |
         +-------------------------+------------------------+
                                   |
                          [Adaptive Fusion]
                    h_fused = alpha * h_entropy +
                              (1-alpha) * h_importance
                                   |
                          [Output Projection]
                                   |
                    +-----------------------------+
                    |             |               |
              h_rgb_out      h_nir_out       h_tir_out
                    |             |               |
                    +-----[Concatenate]-------+
                                  |
                          output (B, 1536)
```

---

## Hierarchical Structure

```
Level 0: DualGatedAdaptiveFusionV3
    |
    +-- Level 1: Attention Pooling (per modality)
    |       |
    |       +-- rgb_query (learnable parameter)
    |       +-- nir_query (learnable parameter)
    |       +-- tir_query (learnable parameter)
    |       +-- attn_pool (shared MultiheadAttention)
    |       +-- attn_norm (shared LayerNorm)
    |
    +-- Level 1: Information Entropy Gate (IEG)
    |       |
    |       +-- Level 2: Entropy Computation
    |       |       +-- |h_m| + epsilon
    |       |       +-- prob = normalize(feat_abs)
    |       |       +-- H_m = -sum(p * log(p))
    |       |
    |       +-- Level 2: Score Computation
    |       |       +-- entropy_proj (Linear)
    |       |       +-- z_m = proj(h_m).mean()
    |       |       +-- score_m = z_m * exp(-H_m / tau)
    |       |
    |       +-- Level 2: Weighted Fusion
    |               +-- softmax(scores)
    |               +-- weighted sum
    |
    +-- Level 1: Modality Importance Gate (MIG)
    |       |
    |       +-- Level 2: Concatenation
    |       |       +-- h_concat = [h_rgb; h_nir; h_tir]
    |       |
    |       +-- Level 2: Gate Network
    |       |       +-- Linear(1536 -> 512)
    |       |       +-- LayerNorm(512)
    |       |       +-- ReLU
    |       |       +-- Linear(512 -> 3)
    |       |       +-- Sigmoid
    |       |
    |       +-- Level 2: Gated Fusion
    |               +-- g_m * h_m
    |               +-- sum
    |
    +-- Level 1: Adaptive Fusion
    |       |
    |       +-- _alpha (learnable parameter)
    |       +-- h_fused = alpha * h_entropy + (1-alpha) * h_importance
    |
    +-- Level 1: Output Projection
            |
            +-- modal_enhance (Linear + LayerNorm)
            +-- Residual enhancement: h_m + h_enhance
            +-- Concatenation
```

---

## Mathematical Formulas Summary

### Attention Pooling (V3)

1. **Query-based Pooling**:
   $$h_m = \text{LayerNorm}(\text{MultiheadAttn}(Q_m, V_m, V_m)[:,0,:])$$
   where $Q_m$ is the learnable query, $V_m$ are the tokens from SDTPS.

### Information Entropy Gate (IEG)

2. **Feature Entropy**:
   $$H(h_m) = -\sum_{i=1}^{C} p_i \log(p_i), \quad p_i = \frac{|h_{m,i}| + \epsilon}{\sum_j |h_{m,j}| + \epsilon}$$

3. **Entropy-Modulated Score**:
   $$s_m = z_m \cdot \exp(-H(h_m) / \tau)$$
   where $z_m = \text{mean}(\text{Linear}(h_m))$

4. **Entropy Weights**:
   $$w_m = \frac{\exp(s_m)}{\sum_{m'} \exp(s_{m'})}$$

5. **Entropy-Weighted Fusion**:
   $$h_{\text{entropy}} = \sum_{m} w_m \cdot h_m$$

### Modality Importance Gate (MIG)

6. **Gate Values**:
   $$g = \sigma(\text{MLP}([h_{\text{rgb}}; h_{\text{nir}}; h_{\text{tir}}]))$$
   where $g \in (0,1)^3$

7. **Gated Fusion**:
   $$h_{\text{importance}} = g_{\text{rgb}} \cdot h_{\text{rgb}} + g_{\text{nir}} \cdot h_{\text{nir}} + g_{\text{tir}} \cdot h_{\text{tir}}$$

### Adaptive Fusion

8. **Learnable Balance**:
   $$\alpha = \sigma(\alpha_{\text{raw}}), \quad \alpha_{\text{raw}} \sim \mathcal{N}(0, 1)$$

9. **Final Fusion**:
   $$h_{\text{fused}} = \alpha \cdot h_{\text{entropy}} + (1 - \alpha) \cdot h_{\text{importance}}$$

### Output Projection

10. **Enhanced Output**:
    $$h_m^{\text{out}} = h_m + \text{LayerNorm}(\text{Linear}(h_{\text{fused}}))$$
    $$\text{output} = [h_{\text{rgb}}^{\text{out}}; h_{\text{nir}}^{\text{out}}; h_{\text{tir}}^{\text{out}}]$$

---

## Data Flow Description

### Input Processing (V3 Attention Pooling)

1. **Receive SDTPS Outputs**:
   - Three token sequences from SDTPS: (B, 26, 512) each
   - Each sequence contains 25 aggregated tokens + 1 extra token

2. **Attention Pooling**:
   - Use learnable query to attend to tokens via cross-attention
   - Query (1, 1, 512) expanded to (B, 1, 512)
   - MultiheadAttention: Q attends to K=V (tokens)
   - Output normalized with LayerNorm
   - Result: h_rgb, h_nir, h_tir each (B, 512)

### Information Entropy Gate (Reliability-Based)

3. **Entropy Computation**:
   - Convert features to probability distribution
   - Compute Shannon entropy H(h_m)
   - Low entropy = high certainty = more reliable

4. **Score Computation**:
   - Project features with linear layer
   - Modulate with entropy: score = z * exp(-H/tau)
   - Lower entropy leads to higher score

5. **Weighted Fusion**:
   - Apply softmax to get weights summing to 1
   - Compute weighted average of modality features
   - Result: h_entropy (B, 512)

### Modality Importance Gate (Learned)

6. **Feature Concatenation**:
   - Concatenate all modality features: (B, 1536)

7. **Gate Generation**:
   - MLP maps concatenated features to 3 gate values
   - Sigmoid ensures gates in (0, 1)
   - Each gate controls one modality's contribution

8. **Gated Sum**:
   - Scale each modality by its gate
   - Sum to get h_importance (B, 512)

### Adaptive Fusion

9. **Balance Two Branches**:
   - Learnable alpha (initialized to 0.5)
   - h_fused = alpha * h_entropy + (1-alpha) * h_importance
   - Allows model to learn optimal balance

### Output Projection

10. **Residual Enhancement**:
    - Project fused feature through Linear + LayerNorm
    - Add to each original modality feature
    - Concatenate enhanced features: (B, 1536)

---

## Design Rationale

### Why Two Gates?

| Gate Type | Purpose | Signal Source | Strength |
|-----------|---------|---------------|----------|
| IEG | Reliability weighting | Feature entropy | Good for noisy/uncertain modalities |
| MIG | Importance weighting | Learned gates | Good for sample-specific fusion |

### Why Adaptive Fusion?

- Different samples may benefit from different fusion strategies
- Learnable alpha allows end-to-end optimization
- Initial value 0.5 provides balanced starting point

### V3 vs V1 Differences

| Aspect | V1 | V3 |
|--------|----|----|
| Input | (B, C) global features | (B, K+1, C) token sequences |
| Pooling | External mean pooling | Internal attention pooling |
| Query | None | Learnable per-modality |
| Flexibility | Fixed pooling | Learned aggregation |

---

## Code Reference

**Source File**: `/home/maxingan/copyfromssd/workfromlocal/newdemo/DeMo2/modeling/dual_gated_fusion.py`

**Key Classes**:
- `InformationEntropyGate` (Line 34-116): Computes entropy-based weights
- `ModalityImportanceGate` (Line 119-182): Learns sample-specific gates
- `DualGatedAdaptiveFusion` (Line 185-287): V1 - requires pre-pooled features
- `DualGatedPostFusion` (Line 406-555): For SDTPS output fusion
- `DualGatedAdaptiveFusionV3` (Line 558-742): V3 - with built-in attention pooling

**Configuration Parameters** (from `config/defaults.py`):
- `USE_DGAF`: False (enable DGAF)
- `DGAF_VERSION`: 'v3' (use V3 with attention pooling)
- `DGAF_TAU`: 1.0 (entropy temperature)
- `DGAF_INIT_ALPHA`: 0.5 (initial balance parameter)
- `DGAF_NUM_HEADS`: 8 (attention heads for V3)

**Parameter Count**:
- V1 (DualGatedPostFusion): ~2.1M parameters
- V3 (DualGatedAdaptiveFusionV3): ~2.6M parameters (includes attention pooling)
