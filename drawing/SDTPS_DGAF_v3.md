# SDTPS + DGAF 流程图模板 (精简版)

---

## Nano Banana Pro Prompt

```
You are an expert ML illustrator.
Draw a clean, NeurIPS/ICLR-style scientific figure using Nano Banana Pro.

GOAL:
Create a publication-quality diagram for a multi-modal ReID method.
Focus on TWO novel modules: SDTPS and DGAF.
The backbone is NOT our contribution - draw it as ONE unified block.

GLOBAL RULES:
- Flat, clean NeurIPS style
- Pastel colors, no gradients
- Short labels only
- SDTPS and DGAF are the focus

LAYOUT:
- Vertical flow: Backbone → SDTPS → DGAF → Output

MODULE LIST:

1. Backbone (ONE unified block, not our contribution):
   - Label: "Shared ViT Backbone"
   - Input: "RGB / NIR / TIR"
   - Output: "Patches + Global Features (×3)"
   - Draw as single gray rounded rectangle
   - Small note: "Each modality processed identically"

2. SDTPS Module (MAIN INNOVATION - detailed):
   - Title: "SDTPS: Cross-Modal Token Selection"
   - Key operation: Cross-modal attention scoring
     * Formula: s = (1-2β)·s^p + β·(s^{cross} + 2·s^{self})
   - Show: Patches from one modality guided by other modalities' global features
   - Output: Selected & aggregated tokens per modality

3. DGAF Module (MAIN INNOVATION - detailed):
   - Title: "DGAF: Dual-Gated Fusion"
   - Two parallel gates:
     * IEG (Information Entropy Gate): reliability-based
     * MIG (Modality Importance Gate): learned gating
   - Fusion: h = α·h_IEG + (1-α)·h_MIG
   - α is learnable

4. Output (simplified):
   - "Classifier → Loss"

DRAWING INSTRUCTIONS:
- Backbone: ONE gray box, input "×3 modalities", output "×3 features"
- SDTPS: Show cross-modal arrows inside the block
- DGAF: Show IEG and MIG as two parallel sub-blocks merging with α
- Color: innovation blocks in light blue, backbone in gray

Generate the diagram.
```

---

## 简化流程图

```
     ┌─────────────────────────────────┐
     │      RGB / NIR / TIR Images     │
     └───────────────┬─────────────────┘
                     ↓
     ┌─────────────────────────────────┐
     │                                 │
     │     Shared ViT Backbone         │
     │     (×3 modalities, identical)  │
     │                                 │
     │     Output: Patches + Global    │
     └───────────────┬─────────────────┘
                     ↓
     ╔═════════════════════════════════╗
     ║          SDTPS Module           ║
     ║                                 ║
     ║  ┌─────────────────────────┐    ║
     ║  │  Cross-Modal Attention  │    ║
     ║  │                         │    ║
     ║  │  patch_m ← global_other │    ║
     ║  └────────────┬────────────┘    ║
     ║               ↓                 ║
     ║  Score → Top-K → Aggregation    ║
     ╚═══════════════╤═════════════════╝
                     ↓
     ╔═════════════════════════════════╗
     ║          DGAF Module            ║
     ║                                 ║
     ║     ┌───────┐   ┌───────┐       ║
     ║     │  IEG  │   │  MIG  │       ║
     ║     │(熵门控)│   │(重要性)│       ║
     ║     └───┬───┘   └───┬───┘       ║
     ║         └─────┬─────┘           ║
     ║               ↓                 ║
     ║      α·h_IEG + (1-α)·h_MIG      ║
     ╚═══════════════╤═════════════════╝
                     ↓
     ┌─────────────────────────────────┐
     │        Classifier → Loss        │
     └─────────────────────────────────┘
```

---

## 核心创新

### SDTPS: 跨模态引导的 Token 选择

```
s = (1-2β)·s^pred + β·(s^cross + 2·s^self)
```

- 用**其他模态**的全局特征引导**当前模态**的 patch 选择
- 选择显著 patches，聚合冗余 patches

### DGAF: 双门控自适应融合

```
h_fused = α·h_IEG + (1-α)·h_MIG
```

- **IEG**: 低熵 = 高可靠性 → 高权重
- **MIG**: 学习每个模态的重要性
- **α**: 可学习，平衡两种门控

---

## 绘图要点

| 部分 | 处理方式 |
|------|----------|
| Backbone | **一个**灰色方块，标注"×3 modalities" |
| SDTPS | 蓝色方块，内部显示跨模态箭头 |
| DGAF | 蓝色方块，内部显示 IEG \|\| MIG → α融合 |
| Output | 简单方块 |
