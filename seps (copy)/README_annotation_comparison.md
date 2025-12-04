# SEPS 代码注释版本对比

## 📁 文件概览

| 文件名 | 行数 | 注释风格 | 适合人群 | 代码逻辑 |
|--------|------|---------|---------|---------|
| `seps_modules_reviewed.py` | 744 | 简洁注释 | 有经验的开发者 | ✅ 完全相同 |
| `seps_modules_reviewed_v2.py` | 1569 | 详细注释+论文对应 | 研究者/论文复现 | ✅ 完全相同 |
| `seps_modules_ultra_detailed.py` | 1100 | 逐行注释+Tensor变化 | PyTorch初学者 | ✅ 完全相同 |

## 🎯 三个版本的差异

### 1️⃣ seps_modules_reviewed.py (简洁版)
```python
class TokenSparse(nn.Module):
    """Patch 选择器（可选论文打分 + 直通 Gumbel-topk）。"""

    def __init__(self, embed_dim: int = 512, ...):
        super().__init__()
        self.embed_dim = embed_dim
        if use_paper_version:
            self.score_predictor = nn.Sequential(...)
```

**特点：**
- ✅ 简洁的中文注释
- ✅ 关键逻辑说明
- ❌ 无详细公式对应
- ❌ 无Tensor形状说明

---

### 2️⃣ seps_modules_reviewed_v2.py (论文对应版)
```python
class TokenSparse(nn.Module):
    """
    Token稀疏选择模块 - SDTPS的第一阶段

    实现论文 Section 3.2.1 "Semantic Scoring" 中描述的语义评分机制。

    ==================== 论文公式对应 ====================

    公式(1) - Score-aware Prediction Network:
        s_i^p = σ(MLP(v_i)), i ∈ {1, ..., N}

        其中:
        - s_i^p ∈ [0,1]: 第i个patch的预测显著性得分
        - v_i: 第i个视觉patch的特征向量
        - σ: sigmoid激活函数
        - MLP: 两层全连接网络

    公式(2) - 多源注意力得分:
        s_i^{st} = Norm(v_i^T · E_{st} / d)  # 稀疏文本相关性
        ...
    """
```

**特点：**
- ✅ 完整的文档字符串
- ✅ 论文公式一一对应
- ✅ 数学符号详细说明
- ✅ 参数含义解释
- ❌ 代码逐行注释较少

---

### 3️⃣ seps_modules_ultra_detailed.py (逐行注释版) ⭐ 新文件
```python
class TokenSparse(nn.Module):
    """
    Token稀疏选择模块

    功能: 从N个patch中选择K个显著patch (K = N * sparse_ratio)
    方法: 综合评分 = MLP预测 + 图像自注意力 + 文本交叉注意力
    """

    def __init__(self, embed_dim: int = 512, ...):
        super().__init__()
        self.embed_dim = embed_dim  # 特征维度
        self.sparse_ratio = sparse_ratio  # 保留比例

        # 论文公式(1): MLP预测器 (仅论文版本)
        if use_paper_version:
            # 输入: (*, C) → 输出: (*, 1)
            self.score_predictor = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 4),  # (*, C) → (*, C//4)
                nn.GELU(),                              # 激活函数
                nn.Linear(embed_dim // 4, 1),          # (*, C//4) → (*, 1)
                nn.Sigmoid(),                           # 输出范围[0,1]
            )

    def forward(self, tokens, ...):
        """
        前向传播

        输入:
            tokens: (B, N, C) - patch特征
            attention_x: (B, N) - 图像自注意力得分
            ...

        输出:
            select_tokens: (B, N_keep, C) - 选中的patch
            extra_token: (B, 1, C) - 融合的丢弃patch
            score_mask: (B, N) - 决策矩阵 (1=选中, 0=丢弃)
        """
        B_v, L_v, C = tokens.size()  # 获取形状: batch, patch数, 特征维度

        # =========================================================
        # 计算综合得分 score
        # =========================================================
        if self.use_paper_version:
            # 论文版本: 公式(1)-(3)

            # 公式(1): MLP预测得分
            s_pred = self.score_predictor(tokens)  # (B, N, C) → (B, N, 1)
            s_pred = s_pred.squeeze(-1)            # (B, N, 1) → (B, N)

            # Min-Max归一化函数
            def normalize_score(s: torch.Tensor) -> torch.Tensor:
                # 输入: (B, N) → 输出: (B, N), 范围[0,1]
                s_min = s.min(dim=-1, keepdim=True)[0]  # (B, 1)
                s_max = s.max(dim=-1, keepdim=True)[0]  # (B, 1)
                return (s - s_min) / (s_max - s_min + 1e-8)  # (B, N)

            # 公式(2): 归一化各注意力得分
            s_im = normalize_score(attention_x)     # (B, N) - 图像自注意力
            ...
```

**特点：**
- ✅ **每行代码都有注释**
- ✅ **Tensor形状变化清晰标注** `(B, N, C) → (B, N, 1) → (B, N)`
- ✅ **函数功能简明说明**
- ✅ **参数输入输出格式明确**
- ✅ **关键步骤分块注释**
- ✅ **适合PyTorch初学者学习**

---

## 📊 代码量对比

```
seps_modules_reviewed.py:      744 行 (基准)
seps_modules_reviewed_v2.py:  1569 行 (+111%)
seps_modules_ultra_detailed.py: 1100 行 (+48%)
```

**ultra_detailed 版本行数更少的原因：**
- 去除了超长的文档字符串（v2的详细论文解释）
- 采用行内注释而非大段注释块
- 重点关注代码逻辑和Tensor变化，而非论文理论

---

## 🎓 使用建议

### 选择 `seps_modules_reviewed.py` 如果你：
- ✅ 熟悉PyTorch和Transformer
- ✅ 只需要快速理解代码逻辑
- ✅ 不需要论文对应

### 选择 `seps_modules_reviewed_v2.py` 如果你：
- ✅ 需要复现论文实验
- ✅ 想深入理解论文公式
- ✅ 需要详细的文档字符串

### 选择 `seps_modules_ultra_detailed.py` 如果你：⭐ **推荐新手**
- ✅ 正在学习PyTorch
- ✅ 需要理解每行代码的作用
- ✅ 想追踪Tensor形状变化
- ✅ 需要快速定位问题

---

## 🔍 关键代码片段对比

### 场景1: TokenSparse 的 forward 函数

#### reviewed.py (简洁版)
```python
B_v, L_v, C = tokens.size()

if self.use_paper_version:
    s_pred = self.score_predictor(tokens).squeeze(-1)
    ...
```

#### reviewed_v2.py (论文对应版)
```python
        # 获取输入形状
        B_v, L_v, C = tokens.size()  # B_v=batch, L_v=N(patch数), C=d(特征维度)

        if self.use_paper_version:
            # =====================================================
            # 论文版本: 完整实现公式(1)-(3)
            # =====================================================

            # -----------------------------------------------------
            # 公式(1): Score-aware Prediction Network
            # s_i^p = σ(MLP(v_i)), i ∈ {1, ..., N}
            # -----------------------------------------------------
            s_pred = self.score_predictor(tokens).squeeze(-1)  # (B, N, 1) -> (B, N)
            ...
```

#### ultra_detailed.py (逐行注释版) ⭐
```python
        B_v, L_v, C = tokens.size()  # 获取形状: batch, patch数, 特征维度

        # =========================================================
        # 计算综合得分 score
        # =========================================================
        if self.use_paper_version:
            # 论文版本: 公式(1)-(3)

            # 公式(1): MLP预测得分
            s_pred = self.score_predictor(tokens)  # (B, N, C) → (B, N, 1)
            s_pred = s_pred.squeeze(-1)            # (B, N, 1) → (B, N)
            ...
```

**对比总结：**
- `reviewed.py`: 最简洁，适合快速阅读
- `reviewed_v2.py`: 最详细，论文对应完整
- `ultra_detailed.py`: 平衡版，**Tensor形状追踪最清晰**

---

## 🧪 测试验证

所有三个版本的代码逻辑**完全相同**，测试结果一致：

```bash
$ python seps_modules_ultra_detailed.py
Using device: cuda
======================================================================
测试开源代码版本 (use_paper_version=False)
======================================================================
Similarity shape: torch.Size([2, 2])
Mask shape: torch.Size([2, 2, 16])
Parameters: 291

======================================================================
测试论文版本 (use_paper_version=True)
======================================================================
Similarity shape: torch.Size([2, 2])
Mask shape: (torch.Size([2, 2, 16]), torch.Size([2, 2, 16]))
Parameters: 1,198

✓ 所有测试通过!
```

---

## 💡 学习路径推荐

1. **初学者路径：**
   ```
   seps_modules_ultra_detailed.py (理解代码逻辑)
   → seps_modules_reviewed_v2.py (理解论文对应)
   → 论文原文 (理解理论)
   ```

2. **研究者路径：**
   ```
   论文原文 (理解理论)
   → seps_modules_reviewed_v2.py (代码实现)
   → seps_modules_reviewed.py (简洁参考)
   ```

3. **工程师路径：**
   ```
   seps_modules_reviewed.py (快速理解)
   → seps_modules_ultra_detailed.py (调试时查看)
   ```

---

## 📚 相关资源

- 论文: [arXiv:2511.01390](https://arxiv.org)
- 会议: ICLR 2026
- GitHub: [seps-repo](https://github.com/...)

---

## 🙏 致谢

本注释对比文档由 Claude Code 生成，帮助不同背景的开发者更好地理解 SEPS 框架。
