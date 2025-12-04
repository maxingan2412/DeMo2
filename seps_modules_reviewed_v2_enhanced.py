"""
SEPS: Semantic-Enhanced Patch Slimming Framework (增强版 - 完整注释+Tensor变化)
====================================================================================

基于 seps_modules_reviewed_v2.py，在原有详细注释基础上增加：
1. 每个操作的 Tensor 形状变化
2. 函数功能简述
3. 关键步骤的分块标注

论文: SEPS: Semantic-enhanced Patch Slimming Framework for fine-grained cross-modal alignment
发表: ICLR 2026
arXiv: 2511.01390

使用方式:
    >>> from seps_modules_reviewed_v2_enhanced import CrossSparseAggrNet, SEPSLoss
    >>>
    >>> # 论文完整版
    >>> model = CrossSparseAggrNet(
    ...     use_paper_version=True,
    ...     use_dual_aggr=True,
    ...     use_gumbel_softmax=True,
    ... )
    >>>
    >>> # 开源代码兼容版
    >>> model = CrossSparseAggrNet(use_paper_version=False)
"""

# =============================================================================
# 依赖导入
# =============================================================================
import math
from typing import Iterable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# 全局配置
# =============================================================================
USE_PAPER_VERSION_DEFAULT = False  # 默认使用开源代码版本


# =============================================================================
# 辅助函数
# =============================================================================

def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """
    L2归一化函数

    功能: 将向量归一化为单位向量（模长=1）

    数学原理:
        x_norm = x / ||x||_2
        其中 ||x||_2 = sqrt(x_1² + x_2² + ... + x_n²)

    应用: 归一化后，点积 = 余弦相似度
        a·b = ||a|| ||b|| cos(θ) = 1×1×cos(θ) = cos(θ)

    论文中用于: 计算 patch-word 相似度矩阵 A
        A_{ij} = (v̂_i)^T t_j / (||v̂_i|| ||t_j||)

    Args:
        x: 输入张量 - 任意形状
        dim: 归一化的维度，默认-1（最后一个维度）
        eps: 防止除零的小常数

    Returns:
        归一化后的张量，形状不变，指定维度上的模长为1

    Tensor变化:
        输入: x - (*, d)
        输出: x_norm - (*, d)，其中 ||x_norm||_2 = 1

    示例:
        >>> x = torch.tensor([[3.0, 4.0]])  # 长度 = sqrt(9+16) = 5
        >>> x_norm = l2_normalize(x, dim=-1)
        >>> print(x_norm)  # tensor([[0.6, 0.8]])
        >>> print(x_norm.norm())  # tensor(1.0000)
    """
    return F.normalize(x, p=2, dim=dim, eps=eps)  # 形状不变


def is_sqr(n: int) -> bool:
    """
    检查n是否为完全平方数

    功能: 判断ViT输出是否包含[CLS] token

    原理:
        - ViT输出形状: (B, N+1, C) 或 (B, N, C)
        - 如果 N 是完全平方数(如196=14²)，说明没有[CLS]
        - 如果 N 不是完全平方数(如197=196+1)，说明有[CLS]

    Args:
        n: 待检查的整数（patch序列长度）

    Returns:
        True: n是完全平方数（没有[CLS] token）
        False: n不是完全平方数（有[CLS] token）

    示例:
        >>> is_sqr(196)  # 14² = 196
        True
        >>> is_sqr(197)  # 196 + 1 (含[CLS])
        False
    """
    a = int(math.sqrt(n))  # 计算平方根并取整
    return a * a == n      # 检查平方后是否等于原数


# =============================================================================
# SDTPS Stage 1: TokenSparse (Semantic Scoring + Patch Selection)
# 对应论文 Section 3.2 - Sparse and Dense Text-Aware Patch Selection Module
# =============================================================================

class TokenSparse(nn.Module):
    """
    Token稀疏选择模块 - SDTPS的第一阶段

    功能: 从N个patch中选择K个最显著的patch
          K = ceil(N × sparse_ratio)

    方法: 综合评分 = MLP预测 + 多源注意力（图像、稀疏文本、稠密文本）

    ==================== 论文公式对应 ====================

    公式(1) - Score-aware Prediction Network:
        s_i^p = σ(MLP(v_i)), i ∈ {1, ..., N}

    公式(2) - 多源注意力得分:
        s_i^{st} = Norm(v_i^T · E_{st} / d)  # 稀疏文本相关性
        s_i^{dt} = Norm(v_i^T · E_{dt} / d)  # 稠密文本相关性
        s_i^{im} = Norm(v_i^T · E_{im} / d)  # 图像自注意力

    公式(3) - 综合显著性得分:
        s_i = (1-2β)·s_i^p + β·(s_i^{st} + s_i^{dt} + 2·s_i^{im})

    ==================== 网络结构 ====================

    论文版本 (use_paper_version=True):
        Input: v_i (*, C)
          ↓ Linear(C → C//4)
          ↓ GELU()
          ↓ Linear(C//4 → 1)
          ↓ Sigmoid()
        Output: s_i^p (*, 1) ∈ [0,1]

    Args:
        embed_dim: 特征维度 d (如512)
        sparse_ratio: 保留比例 ρ (如0.6表示保留60%)
        use_paper_version: 是否使用论文完整机制

    Tensor流程:
        输入:
            tokens: (B, N, C)
            attention_x: (B, N)
            attention_y: (B, N)
            attention_y_dense: (B, N) or None
        输出:
            select_tokens: (B, N_keep, C)  where N_keep = ceil(N × ρ)
            extra_token: (B, 1, C)
            score_mask: (B, N)
    """

    def __init__(
        self,
        embed_dim: int = 512,           # 特征维度 d
        sparse_ratio: float = 0.6,      # 选择比例 ρ
        use_paper_version: bool = USE_PAPER_VERSION_DEFAULT,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.sparse_ratio = sparse_ratio
        self.use_paper_version = use_paper_version

        # 论文公式(1): Score-aware Prediction Network
        if use_paper_version:
            self.score_predictor = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 4),  # (*, C) → (*, C//4)
                nn.GELU(),                              # 激活函数
                nn.Linear(embed_dim // 4, 1),          # (*, C//4) → (*, 1)
                nn.Sigmoid(),                           # 输出∈[0,1]
            )

    def forward(
        self,
        tokens: torch.Tensor,                           # (B, N, C)
        attention_x: torch.Tensor,                      # (B, N)
        attention_y: torch.Tensor,                      # (B, N)
        attention_y_dense: Optional[torch.Tensor] = None,  # (B, N) or None
        beta: float = 0.25,                             # β参数
        use_gumbel: bool = False,                       # Gumbel-Softmax开关
        gumbel_tau: float = 1.0,                        # Gumbel温度τ
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        执行语义评分和patch选择

        流程:
            1. 计算综合得分 score ← 公式(1)-(3)
            2. Top-K选择: 选择得分最高的K个patch
            3. 生成决策矩阵 D
            4. 提取选中的patch
            5. 融合被丢弃的patch为extra token

        Args:
            tokens: (B, N, C) - patch特征 V = {v_1, ..., v_N}
            attention_x: (B, N) - 图像自注意力 s^{im}
            attention_y: (B, N) - 稀疏文本注意力 s^{st}
            attention_y_dense: (B, N) or None - 稠密文本注意力 s^{dt}
            beta: β权重参数，默认0.25
            use_gumbel: 是否使用Gumbel-Softmax可微采样
            gumbel_tau: Gumbel温度参数τ

        Returns:
            select_tokens: (B, N_keep, C) - 选中的显著patch
            extra_token: (B, 1, C) - 融合的冗余patch
            score_mask: (B, N) - 决策矩阵D，1=选中，0=丢弃

        Tensor变化:
            tokens: (B, N, C)
            ↓ [计算得分]
            score: (B, N)
            ↓ [Top-K选择]
            keep_policy: (B, N_keep)
            score_mask: (B, N)
            ↓ [gather操作]
            select_tokens: (B, N_keep, C)
            extra_token: (B, 1, C)
        """
        # 获取输入形状
        B_v, L_v, C = tokens.size()  # B=batch, L_v=N(patch数), C=d(特征维度)

        # =========================================================
        # Step 1: 计算综合得分 score
        # =========================================================
        if self.use_paper_version:
            # 论文版本: 公式(1)-(3)

            # 公式(1): s_i^p = σ(MLP(v_i))
            s_pred = self.score_predictor(tokens)  # (B, N, C) → (B, N, 1)
            s_pred = s_pred.squeeze(-1)            # (B, N, 1) → (B, N)

            # Min-Max归一化函数
            def normalize_score(s: torch.Tensor) -> torch.Tensor:
                """将得分归一化到[0,1]范围"""
                s_min = s.min(dim=-1, keepdim=True)[0]  # (B, N) → (B, 1)
                s_max = s.max(dim=-1, keepdim=True)[0]  # (B, N) → (B, 1)
                return (s - s_min) / (s_max - s_min + 1e-8)  # (B, N)

            # 公式(2): 归一化各注意力得分
            s_im = normalize_score(attention_x)     # (B, N) - 图像自注意力
            s_st = normalize_score(attention_y)     # (B, N) - 稀疏文本
            s_dt = (
                normalize_score(attention_y_dense)  # (B, N) - 稠密文本
                if attention_y_dense is not None
                else torch.zeros_like(s_st)         # (B, N) - 全0
            )

            # 公式(3): 综合得分
            # score = (1-2β)·s_pred + β·(s_st + s_dt + 2·s_im)
            score = (1 - 2 * beta) * s_pred + beta * (s_st + s_dt + 2 * s_im)  # (B, N)
        else:
            # 开源代码版本: 简单相加
            score = attention_x + attention_y  # (B, N)

        # =========================================================
        # Step 2: Top-K选择
        # =========================================================
        num_keep_token = max(1, math.ceil(L_v * self.sparse_ratio))  # K = ceil(N×ρ)

        # 降序排序
        score_sort, score_index = torch.sort(score, dim=1, descending=True)
        # score_sort: (B, N) - 排序后的得分
        # score_index: (B, N) - 原始索引

        keep_policy = score_index[:, :num_keep_token]  # (B, K) - 保留的索引

        # =========================================================
        # Step 3: 生成决策矩阵 D
        # =========================================================
        if use_gumbel:
            # Gumbel-Softmax可微采样 + Straight-Through Estimator
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(score) + 1e-9) + 1e-9)  # (B, N)
            soft_mask = F.softmax((score + gumbel_noise) / gumbel_tau, dim=1)  # (B, N) - 软决策
            hard_mask = torch.zeros_like(score).scatter(1, keep_policy, 1.0)   # (B, N) - 硬决策
            score_mask = hard_mask + (soft_mask - soft_mask.detach())  # (B, N) - STE
        else:
            # 标准Top-K (不可微)
            score_mask = torch.zeros_like(score).scatter(1, keep_policy, 1.0)  # (B, N)

        # =========================================================
        # Step 4: 提取选中的patch
        # =========================================================
        select_tokens = torch.gather(
            tokens, dim=1,
            index=keep_policy.unsqueeze(-1).expand(-1, -1, C)  # (B, K) → (B, K, C)
        )  # (B, N, C) → (B, K, C)

        # =========================================================
        # Step 5: 融合被丢弃的patch
        # =========================================================
        non_keep_policy = score_index[:, num_keep_token:]  # (B, N-K) - 丢弃的索引
        non_tokens = torch.gather(
            tokens, dim=1,
            index=non_keep_policy.unsqueeze(-1).expand(-1, -1, C)
        )  # (B, N, C) → (B, N-K, C)

        non_keep_score = score_sort[:, num_keep_token:]  # (B, N-K) - 丢弃patch的得分
        non_keep_score = F.softmax(non_keep_score, dim=1).unsqueeze(-1)  # (B, N-K) → (B, N-K, 1)
        extra_token = torch.sum(non_tokens * non_keep_score, dim=1, keepdim=True)  # (B, N-K, C) → (B, 1, C)

        return select_tokens, extra_token, score_mask


# =============================================================================
# SDTPS Stage 2a: TokenAggregation (单分支聚合)
# 对应论文公式(4)简化版
# =============================================================================

class TokenAggregation(nn.Module):
    """
    单分支Token聚合模块

    功能: 将N个patch聚合为N_c个patch (N_c < N)
          通过学习权重矩阵W进行加权聚合

    ==================== 论文公式(4)简化版 ====================

    原公式(4):
        v̂_j = Σ_{i=1}^{N_s} (W_s)_{ij} · v_i^s + Σ_{i=1}^{N_d} (W_d)_{ij} · v_i^d

    简化版（本类实现）:
        v̂_j = Σ_{i=1}^{N} W_{ij} · v_i, j ∈ {1, ..., N_c}

    其中:
        - W ∈ R^{N × N_c}: 聚合权重矩阵
        - Σ_i W_{ij} = 1: 权重归一化（softmax保证）

    ==================== 网络结构 ====================

    Input: v_i (B, N, C)
      ↓ LayerNorm
      ↓ Linear(C → hidden)
      ↓ GELU
      ↓ Linear(hidden → N_c)
    Output: logits (B, N, N_c)
      ↓ Transpose
    Weight: (B, N_c, N)
      ↓ Softmax
    Output: v̂_j (B, N_c, C)

    Args:
        dim: 特征维度 d
        keeped_patches: 聚合后的patch数量 N_c
        dim_ratio: 隐藏层维度比例

    Tensor流程:
        输入: x (B, N, C)
        ↓ weight network
        logits: (B, N, N_c)
        ↓ transpose + scale
        weight: (B, N_c, N)
        ↓ softmax
        weight: (B, N_c, N)，Σ_i W[b,j,i]=1
        ↓ bmm with x
        输出: (B, N_c, C)
    """

    def __init__(
        self,
        dim: int = 512,             # 特征维度 d
        keeped_patches: int = 64,   # 聚合后patch数量 N_c
        dim_ratio: float = 0.2,     # 隐藏层维度比例
    ):
        super().__init__()
        hidden_dim = int(dim * dim_ratio)  # 隐藏层维度

        # 权重生成网络
        self.weight = nn.Sequential(
            nn.LayerNorm(dim),                      # (*, C) → (*, C)
            nn.Linear(dim, hidden_dim),             # (*, C) → (*, hidden)
            nn.GELU(),                               # 激活
            nn.Linear(hidden_dim, keeped_patches),  # (*, hidden) → (*, N_c)
        )
        self.scale = nn.Parameter(torch.ones(1, 1, 1))  # 可学习缩放因子

    def forward(
        self,
        x: torch.Tensor,                              # (B, N, C)
        keep_policy: Optional[torch.Tensor] = None,   # (B, N) or None
    ) -> torch.Tensor:
        """
        聚合patches

        公式: v̂_j = Σ_i W_{ij} * v_i

        Args:
            x: (B, N, C) - 输入patch特征
            keep_policy: (B, N) or None - 可选mask

        Returns:
            (B, N_c, C) - 聚合后的patch特征

        Tensor变化:
            x: (B, N, C)
            ↓ self.weight()
            logits: (B, N, N_c)
            ↓ transpose
            weight: (B, N_c, N)
            ↓ softmax
            weight: (B, N_c, N)
            ↓ bmm
            output: (B, N_c, C)
        """
        # 生成权重
        weight = self.weight(x)                 # (B, N, C) → (B, N, N_c)
        weight = weight.transpose(2, 1)         # (B, N, N_c) → (B, N_c, N)
        weight = weight * self.scale            # (B, N_c, N) - 缩放

        # 应用mask（如果有）
        if keep_policy is not None:
            keep_policy = keep_policy.unsqueeze(1)  # (B, N) → (B, 1, N)
            weight = weight - (1 - keep_policy) * 1e10  # 无效位置设为极小值

        # Softmax归一化
        weight = F.softmax(weight, dim=2)       # (B, N_c, N)，Σ_i W[b,j,i]=1

        # 批量矩阵乘法: W @ x
        return torch.bmm(weight, x)             # (B, N_c, N) @ (B, N, C) → (B, N_c, C)


# =============================================================================
# SDTPS Stage 2b: DualTokenAggregation (双分支联合聚合)
# 对应论文公式(4)完整版
# =============================================================================

class DualTokenAggregation(nn.Module):
    """
    双分支Token聚合模块 - 论文公式(4)完整实现

    功能: 联合聚合稀疏文本和稠密文本选择的patch
          v̂ = W_s·V_s + W_d·V_d

    ==================== 论文公式(4) ====================

    v̂_j = Σ_{i=1}^{N_s} (W_s)_{ij} · v_i^s + Σ_{i=1}^{N_d} (W_d)_{ij} · v_i^d

    其中:
        - V_s = {v_1^s, ..., v_{N_s}^s}: 稀疏文本选择的显著patch
        - V_d = {v_1^d, ..., v_{N_d}^d}: 稠密文本选择的显著patch
        - W_s ∈ R^{N_s × N_c}: 稀疏文本的聚合权重矩阵
        - W_d ∈ R^{N_d × N_c}: 稠密文本的聚合权重矩阵
        - N_c: 聚合后的patch数量

    权重学习:
        W_s = Softmax(MLP(V_s))
        W_d = Softmax(MLP(V_d))

    Args:
        dim: 特征维度 d
        keeped_patches: 聚合后patch数量 N_c
        dim_ratio: 隐藏层维度比例

    Tensor流程:
        输入:
            tokens_sparse: (B, N_s, C)
            tokens_dense: (B, N_d, C) or None
        处理:
            W_s: (B, N_c, N_s)
            W_d: (B, N_c, N_d)
        输出:
            out: (B, N_c, C)
    """

    def __init__(
        self,
        dim: int = 512,             # 特征维度 d
        keeped_patches: int = 64,   # N_c: 聚合后patch数量
        dim_ratio: float = 0.2,     # 隐藏层维度比例
    ):
        super().__init__()
        hidden_dim = int(dim * dim_ratio)

        # 稀疏文本分支权重网络
        self.weight_sparse = nn.Sequential(
            nn.LayerNorm(dim),                      # (*, C) → (*, C)
            nn.Linear(dim, hidden_dim),             # (*, C) → (*, hidden)
            nn.GELU(),
            nn.Linear(hidden_dim, keeped_patches),  # (*, hidden) → (*, N_c)
        )

        # 稠密文本分支权重网络
        self.weight_dense = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, keeped_patches),
        )

        self.scale = nn.Parameter(torch.ones(1, 1, 1))

    def _aggregate(
        self,
        x: torch.Tensor,               # (B, N, C)
        mask: Optional[torch.Tensor],  # (B, N) or None
        weight_net: nn.Module,
    ) -> torch.Tensor:
        """
        单分支聚合辅助函数

        公式: v̂ = W·x

        Args:
            x: (B, N, C) - 输入patch特征
            mask: (B, N) or None - 决策矩阵D
            weight_net: 权重生成网络

        Returns:
            (B, N_c, C) - 聚合后特征

        Tensor变化:
            x: (B, N, C)
            ↓ weight_net
            weight: (B, N_c, N)
            ↓ softmax
            weight: (B, N_c, N)
            ↓ bmm
            output: (B, N_c, C)
        """
        weight = weight_net(x).transpose(2, 1) * self.scale  # (B, N, C) → (B, N_c, N)

        if mask is not None:
            weight = weight - (1 - mask.unsqueeze(1)) * 1e10  # 应用mask

        weight = F.softmax(weight, dim=2)  # (B, N_c, N) - 归一化
        return torch.bmm(weight, x)        # (B, N_c, N) @ (B, N, C) → (B, N_c, C)

    def forward(
        self,
        tokens_sparse: torch.Tensor,                    # (B, N_s, C)
        tokens_dense: Optional[torch.Tensor] = None,    # (B, N_d, C) or None
        mask_sparse: Optional[torch.Tensor] = None,     # (B, N_s) or None
        mask_dense: Optional[torch.Tensor] = None,      # (B, N_d) or None
    ) -> torch.Tensor:
        """
        双分支联合聚合

        公式: v̂_j = Σ (W_s)_{ij}·v_i^s + Σ (W_d)_{ij}·v_i^d

        Args:
            tokens_sparse: (B, N_s, C) - 稀疏文本选择的patch V_s
            tokens_dense: (B, N_d, C) or None - 稠密文本选择的patch V_d
            mask_sparse: (B, N_s) or None - 稀疏分支mask
            mask_dense: (B, N_d) or None - 稠密分支mask

        Returns:
            (B, N_c, C) - 聚合后的patch特征 V̂

        Tensor变化:
            tokens_sparse: (B, N_s, C)
            ↓ _aggregate (sparse)
            out: (B, N_c, C)
            ↓ (如果有dense)
            tokens_dense: (B, N_d, C)
            ↓ _aggregate (dense)
            out: (B, N_c, C) + (B, N_c, C)
            输出: (B, N_c, C)
        """
        # 稀疏分支聚合: Σ (W_s)_{ij}·v_i^s
        out = self._aggregate(tokens_sparse, mask_sparse, self.weight_sparse)  # (B, N_c, C)

        # 稠密分支聚合: + Σ (W_d)_{ij}·v_i^d
        if tokens_dense is not None:
            out = out + self._aggregate(tokens_dense, mask_dense, self.weight_dense)  # (B, N_c, C)

        return out  # (B, N_c, C)


# =============================================================================
# HRPA: Highly-Relevant Patch-Word Alignment
# 对应论文 Section 3.3 及公式(5)
# =============================================================================

def mask_xattn_one_text(
    img_embs: torch.Tensor,                           # (B_v, N_c, C)
    cap_i_expand: torch.Tensor,                       # (B_v, M, C)
    img_mask: Optional[torch.Tensor] = None,          # (B_v, N_c) or None
    i2t: bool = True,
    scan: bool = True,
    use_paper_version: bool = USE_PAPER_VERSION_DEFAULT,
    top_k: int = 5,
    relevance_mlp: Optional[nn.Module] = None,
) -> torch.Tensor:
    """
    HRPA: 高相关性Patch-Word对齐函数

    功能: 计算图像-文本相似度 S(I, T)
          通过双向对齐机制增强细粒度匹配

    ==================== 论文公式(5) ====================

    S(I, T) = [patch-to-word] + [word-to-patch]

    patch-to-word:
        (1/N_c) Σ_i max_j(A)_{ij} + MLP(TOPK(max_j(A)_{ij}))

    word-to-patch:
        (1/M) Σ_j max_i(A)_{ij} + MLP(TOPK(max_i(A)_{ij}))

    其中:
        - A ∈ R^{N_c × M}: patch-word相似度矩阵
        - A_{ij} = (v̂_i)^T t_j: 已归一化的余弦相似度
        - TOPK: 选择前K个最大值
        - MLP: 相关性学习网络

    Args:
        img_embs: (B_v, N_c, C) - 已归一化的patch特征 V̂
        cap_i_expand: (B_v, M, C) - 已归一化的word特征 T
        img_mask: (B_v, N_c) or None - patch mask
        i2t: 是否计算双向对齐
        scan: 是否使用LeakyReLU（SCAN技巧）
        use_paper_version: 是否使用论文的TopK+MLP机制
        top_k: TopK的K值
        relevance_mlp: 相关性学习网络

    Returns:
        (B_v, 1) - 图像-文本相似度 S(I, T)

    Tensor变化:
        img_embs: (B_v, N_c, C)
        cap_i_expand: (B_v, M, C)
        ↓ bmm
        cap2img_sim (A): (B_v, M, N_c)
        ↓ max (word→patch)
        row_sim: (B_v, M)
        ↓ mean
        row_sim_mean: (B_v, 1)
        ↓ (如果双向)
        column_sim: (B_v, N_c)
        ↓ mean
        column_sim_mean: (B_v, 1)
        ↓ 相加
        sim_one_text: (B_v, 1)
    """
    # 计算相似度矩阵 A = V̂·T^T
    cap2img_sim = torch.bmm(cap_i_expand, img_embs.transpose(1, 2))  # (B_v, M, C) @ (B_v, C, N_c) → (B_v, M, N_c)

    if scan:
        cap2img_sim = F.leaky_relu(cap2img_sim, negative_slope=0.1)  # SCAN技巧: 抑制负相似度

    # =========================================================
    # word-to-patch对齐: 对每个word找最相关的patch
    # =========================================================
    if img_mask is None:
        row_sim = cap2img_sim.max(dim=2)[0]  # (B_v, M, N_c) → (B_v, M)
    else:
        row_sim = (cap2img_sim - 1000 * (1 - img_mask).unsqueeze(1)).max(dim=2)[0]  # 屏蔽无效patch

    row_sim_mean = row_sim.mean(dim=1, keepdim=True)  # (B_v, M) → (B_v, 1)

    # 论文版本: 添加TopK+MLP
    if use_paper_version and relevance_mlp is not None:
        B_v, M = row_sim.shape
        k = min(top_k, M)
        row_topk, _ = row_sim.topk(k, dim=1)  # (B_v, M) → (B_v, k)

        if k < top_k:
            padding = torch.zeros(B_v, top_k - k, device=row_topk.device)
            row_topk = torch.cat([row_topk, padding], dim=1)  # (B_v, k) → (B_v, top_k)

        row_sim_mean = row_sim_mean + relevance_mlp(row_topk)  # (B_v, 1) + (B_v, 1) → (B_v, 1)

    # =========================================================
    # patch-to-word对齐: 对每个patch找最相关的word
    # =========================================================
    if i2t:
        column_sim = cap2img_sim.max(dim=1)[0]  # (B_v, M, N_c) → (B_v, N_c)

        if img_mask is None:
            column_sim_mean = column_sim.mean(dim=1, keepdim=True)  # (B_v, N_c) → (B_v, 1)
        else:
            column_sim_mean = (column_sim * img_mask).sum(dim=-1, keepdim=True) / (
                img_mask.sum(dim=-1, keepdim=True) + 1e-8
            )  # 只对有效patch求平均

        # 论文版本: 添加TopK+MLP
        if use_paper_version and relevance_mlp is not None:
            B_v, N = column_sim.shape
            k = min(top_k, N)
            col_topk, _ = column_sim.topk(k, dim=1)  # (B_v, N) → (B_v, k)

            if k < top_k:
                padding = torch.zeros(B_v, top_k - k, device=col_topk.device)
                col_topk = torch.cat([col_topk, padding], dim=1)

            column_sim_mean = column_sim_mean + relevance_mlp(col_topk)

        # 双向对齐相加
        sim_one_text = row_sim_mean + column_sim_mean  # (B_v, 1) + (B_v, 1) → (B_v, 1)
    else:
        sim_one_text = row_sim_mean  # (B_v, 1)

    return sim_one_text  # (B_v, 1)


class HRPA(nn.Module):
    """
    HRPA模块的类封装版本

    功能: 计算高相关性的patch-word对齐相似度
          封装 mask_xattn_one_text 函数，方便管理参数

    论文原文 (Section 3.3):
        "The HRPA module introduces relevance-aware selection with mean value
        computation to facilitate nuanced fine-grained interactions."

    Args:
        embed_dim: 特征维度 d（接口预留）
        top_k: TopK的K值
        use_paper_version: 是否使用论文机制
        bidirectional: 是否使用双向对齐
        scan: 是否使用LeakyReLU

    Tensor流程:
        输入:
            patch_features: (B_v, N_c, C)
            word_features: (B_v, M, C)
        输出:
            similarity: (B_v, 1)
    """

    def __init__(
        self,
        embed_dim: int = 512,
        top_k: int = 5,
        use_paper_version: bool = USE_PAPER_VERSION_DEFAULT,
        bidirectional: bool = True,
        scan: bool = True,
    ):
        super().__init__()

        self.use_paper_version = use_paper_version
        self.bidirectional = bidirectional
        self.top_k = top_k
        self.scan = scan

        # 论文公式(5)中的MLP: 相关性学习网络
        if use_paper_version:
            self.relevance_mlp = nn.Sequential(
                nn.Linear(top_k, top_k * 2),  # (*, top_k) → (*, top_k*2)
                nn.GELU(),
                nn.Linear(top_k * 2, 1),      # (*, top_k*2) → (*, 1)
            )
        else:
            self.relevance_mlp = None

    def forward(
        self,
        patch_features: torch.Tensor,                  # (B_v, N_c, C)
        word_features: torch.Tensor,                   # (B_v, M, C)
        patch_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算图像-文本相似度 S(I, T)

        Args:
            patch_features: (B_v, N_c, C) - 已归一化的聚合patch特征 V̂
            word_features: (B_v, M, C) - 已归一化的文本word特征 T
            patch_mask: (B_v, N_c) or None - 可选的patch mask

        Returns:
            (B_v, 1) - 图像-文本相似度 S(I, T)

        Tensor变化:
            patch_features: (B_v, N_c, C)
            word_features: (B_v, M, C)
            ↓ mask_xattn_one_text
            similarity: (B_v, 1)
        """
        return mask_xattn_one_text(
            img_embs=patch_features,
            cap_i_expand=word_features,
            img_mask=patch_mask,
            i2t=self.bidirectional,
            scan=self.scan,
            use_paper_version=self.use_paper_version,
            top_k=self.top_k,
            relevance_mlp=self.relevance_mlp,
        )  # (B_v, 1)


# =============================================================================
# CrossSparseAggrNet: 完整的SDTPS + HRPA流程
# =============================================================================

class CrossSparseAggrNet(nn.Module):
    """
    完整的SEPS模型: SDTPS + HRPA

    功能: 实现完整的跨模态对齐流程
          1. SDTPS: 选择+聚合显著patch
          2. HRPA: 计算细粒度对齐相似度

    ==================== 流程概览 ====================

    1. 特征归一化 (L2 normalize)
    2. 分离[CLS] token
    3. 计算图像自注意力 s^{im}
    4. 对每个文本:
        a. 计算交叉注意力 s^{st}, s^{dt}
        b. TokenSparse: 选择显著patch
        c. TokenAggregation: 聚合patch
        d. HRPA: 计算相似度
    5. 返回相似度矩阵和决策矩阵

    ==================== 论文 vs 开源代码 ====================

    | 特性 | 论文 | 开源代码 |
    |------|------|----------|
    | s_dt | 使用 | 不使用 |
    | 聚合 | W_s+W_d双权重 | 单一权重 |
    | 稠密文本 | 只选择 | 也计算相似度 |
    | HRPA | 纯余弦 | LeakyReLU |
    | 决策 | Gumbel | Hard Top-K |

    Args:
        embed_size: 特征维度 d
        num_patches: 输入patch数量 N（不含CLS）
        sparse_ratio: 选择比例 ρ
        aggr_ratio: 聚合比例
        use_paper_version: 是否使用论文完整机制
        top_k: HRPA的TopK参数
        use_gumbel_softmax: 是否使用Gumbel-Softmax
        gumbel_tau: Gumbel温度
        use_dual_aggr: 是否使用双分支聚合
        beta: 公式(3)中的权重参数β

    Tensor流程:
        输入:
            img_embs: (B_v, N+1, C) or (B_v, N, C)
            cap_embs: (B_t, L_s, C)
            long_cap_embs: (B_t, L_d, C) or None

        处理:
            1. 归一化: (B_v, N, C)
            2. 选择: (B_v, N_keep, C)
            3. 聚合: (B_v, N_c, C)
            4. 对齐: (B_v, 1) for each text

        输出:
            训练: (similarity_matrix, score_mask)
                similarity_matrix: (B_v, B_t)
                score_mask: tuple or tensor
            推理: similarity_matrix (B_v, B_t)
    """

    def __init__(
        self,
        embed_size: int = 512,
        num_patches: int = 196,
        sparse_ratio: float = 0.5,
        aggr_ratio: float = 0.4,
        use_paper_version: bool = USE_PAPER_VERSION_DEFAULT,
        top_k: int = 5,
        use_gumbel_softmax: bool = False,
        gumbel_tau: float = 1.0,
        use_dual_aggr: bool = True,
        beta: float = 0.25,
    ):
        super().__init__()

        # 保存配置
        self.hidden_dim = embed_size
        self.num_patches = num_patches
        self.sparse_ratio = sparse_ratio
        self.aggr_ratio = aggr_ratio
        self.use_paper_version = use_paper_version
        self.top_k = top_k
        self.use_gumbel_softmax = use_gumbel_softmax
        self.gumbel_tau = gumbel_tau
        self.use_dual_aggr = use_dual_aggr
        self.beta = beta

        # N_c: 聚合后的patch数量
        self.keeped_patches = int(num_patches * aggr_ratio * sparse_ratio)

        # SDTPS Stage 1: TokenSparse (稀疏文本分支)
        self.sparse_net_cap = TokenSparse(
            embed_dim=self.hidden_dim,
            sparse_ratio=self.sparse_ratio,
            use_paper_version=use_paper_version,
        )

        # SDTPS Stage 1: TokenSparse (稠密文本分支)
        self.sparse_net_long = TokenSparse(
            embed_dim=self.hidden_dim,
            sparse_ratio=self.sparse_ratio,
            use_paper_version=use_paper_version,
        )

        # SDTPS Stage 2: Aggregation
        if use_paper_version and use_dual_aggr:
            # 论文版本: 双分支聚合
            self.aggr_net = DualTokenAggregation(
                dim=self.hidden_dim,
                keeped_patches=self.keeped_patches,
            )
        else:
            # 开源代码版本: 单分支聚合
            self.aggr_net = TokenAggregation(
                dim=self.hidden_dim,
                keeped_patches=self.keeped_patches,
            )

        # HRPA模块
        self.hrpa = HRPA(
            embed_dim=self.hidden_dim,
            top_k=self.top_k,
            use_paper_version=self.use_paper_version,
            bidirectional=True,
            scan=not self.use_paper_version,  # 论文: 纯余弦, 代码: LeakyReLU
        )

    def forward(
        self,
        img_embs: torch.Tensor,                         # (B_v, N+1, C) or (B_v, N, C)
        cap_embs: torch.Tensor,                         # (B_t, L_s, C)
        cap_lens: torch.Tensor,                         # (B_t,)
        long_cap_embs: Optional[torch.Tensor] = None,   # (B_t, L_d, C) or None
        long_cap_lens: Optional[torch.Tensor] = None,   # (B_t,) or None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, ...]]]]:
        """
        前向传播: 计算图像-文本相似度矩阵

        完整流程:
            1. L2归一化: 所有特征归一化为单位向量
            2. 分离[CLS]: 分离CLS token和空间patch
            3. 计算 s^{im}: 图像自注意力
            4. 对每个文本:
                a. 计算 s^{st}, s^{dt}: 交叉注意力
                b. TokenSparse: 选择显著patch
                c. TokenAggregation: 聚合patch
                d. HRPA: 计算S(I,T)
            5. 整合结果并返回

        Args:
            img_embs: (B_v, N+1, C) or (B_v, N, C) - 图像patch特征（可能含[CLS]）
            cap_embs: (B_t, L_s, C) - 稀疏文本特征
            cap_lens: (B_t,) - 稀疏文本实际长度
            long_cap_embs: (B_t, L_d, C) or None - 稠密文本特征
            long_cap_lens: (B_t,) or None - 稠密文本长度

        Returns:
            训练模式 (self.training=True):
                (similarity_matrix, score_mask)
                - similarity_matrix: (B_v, B_t)
                - score_mask: 决策矩阵
            推理模式 (self.training=False):
                similarity_matrix: (B_v, B_t)

        Tensor变化详解:
            # Step 1: 归一化
            img_embs: (B_v, N+1, C) → (B_v, N+1, C) [normalized]

            # Step 2: 分离[CLS]
            img_cls_emb: (B_v, 1, C)
            img_spatial_embs: (B_v, N, C)

            # Step 3: 图像自注意力
            img_spatial_glo: (B_v, 1, C)
            img_spatial_self_attention: (B_v, N)

            # Step 4: 对每个文本
            for i in range(B_t):
                # 4a: 交叉注意力
                cap_i_glo: (1, 1, C)
                attn_cap: (B_v, N)

                # 4b: TokenSparse
                select_tokens_cap: (B_v, N_keep, C)
                extra_token_cap: (B_v, 1, C)
                score_mask_cap: (B_v, N)

                # 4c: TokenAggregation
                aggr_tokens: (B_v, N_c, C)
                keep_spatial_tokens: (B_v, N_c+1, C)
                select_tokens: (B_v, N_c+2, C) or (B_v, N_c+1, C)

                # 4d: HRPA
                sim_one_text: (B_v, 1)

            # Step 5: 整合
            improve_sims: (B_v, B_t)
        """
        B_v, L_v, C = img_embs.shape  # batch, patch数+[CLS], 特征维度

        # =========================================================
        # Step 1: L2归一化
        # =========================================================
        img_embs_norm = F.normalize(img_embs, dim=-1)      # (B_v, L_v, C) → (B_v, L_v, C)
        cap_embs_norm = F.normalize(cap_embs, dim=-1)      # (B_t, L_s, C) → (B_t, L_s, C)
        long_cap_embs_norm = (
            F.normalize(long_cap_embs, dim=-1)              # (B_t, L_d, C) → (B_t, L_d, C)
            if long_cap_embs is not None
            else None
        )

        # =========================================================
        # Step 2: 分离[CLS] token
        # =========================================================
        self.has_cls_token = not is_sqr(img_embs.shape[1])  # 判断是否有[CLS]

        if self.has_cls_token:
            img_cls_emb = img_embs[:, 0:1, :]           # (B_v, L_v, C) → (B_v, 1, C)
            img_spatial_embs = img_embs[:, 1:, :]       # (B_v, L_v, C) → (B_v, N, C)
            img_spatial_embs_norm = img_embs_norm[:, 1:, :]  # (B_v, N, C)
        else:
            img_cls_emb = None
            img_spatial_embs = img_embs                 # (B_v, N, C)
            img_spatial_embs_norm = img_embs_norm       # (B_v, N, C)

        # =========================================================
        # Step 3: 计算图像自注意力 s^{im}
        # =========================================================
        with torch.no_grad():
            # 图像全局表示: 所有patch的平均
            img_spatial_glo_norm = F.normalize(
                img_spatial_embs.mean(dim=1, keepdim=True), dim=-1
            )  # (B_v, N, C) → (B_v, 1, C) → (B_v, 1, C)

            # 每个patch与全局表示的相似度
            img_spatial_self_attention = (
                img_spatial_glo_norm * img_spatial_embs_norm  # (B_v, 1, C) * (B_v, N, C)
            ).sum(dim=-1)  # → (B_v, N)

        # 结果收集列表
        improve_sims = []          # 相似度列表
        long_sims = []             # 稠密文本相似度（开源代码版本）
        score_mask_all = []        # 决策矩阵列表
        score_mask_long_all = []

        # =========================================================
        # Step 4: 对每个文本进行处理
        # =========================================================
        for i in range(len(cap_lens)):
            # 获取第i个文本
            n_word = int(cap_lens[i])                       # M: word数量
            cap_i = cap_embs[i, :n_word, :]                # (L_s, C) → (M, C)
            cap_i_expand = cap_embs_norm[i, :n_word, :].unsqueeze(0).repeat(B_v, 1, 1)  # (M, C) → (B_v, M, C)

            # 初始化稠密文本相关变量
            dense_attn = None
            long_cap_i_expand = None
            select_tokens_long = extra_token_long = score_mask_long = None

            # ---------------------------------------------------------
            # Step 4a: 计算文本-图像交叉注意力
            # ---------------------------------------------------------
            with torch.no_grad():
                # 稀疏文本全局表示
                cap_i_glo = F.normalize(
                    cap_i.mean(0, keepdim=True).unsqueeze(0), dim=-1
                )  # (M, C) → (1, C) → (1, 1, C)

                # s_i^{st}: 稀疏文本交叉注意力
                attn_cap = (cap_i_glo * img_spatial_embs_norm).sum(dim=-1)  # (1, 1, C) * (B_v, N, C) → (B_v, N)

                # s_i^{dt}: 稠密文本交叉注意力
                if long_cap_embs_norm is not None and long_cap_lens is not None:
                    n_word_long = int(long_cap_lens[i])
                    long_cap_i = long_cap_embs[i, :n_word_long, :]  # (L_d, C) → (M_d, C)
                    long_cap_i_expand = (
                        long_cap_embs_norm[i, :n_word_long, :]
                        .unsqueeze(0).repeat(B_v, 1, 1)
                    )  # (M_d, C) → (B_v, M_d, C)

                    long_cap_i_glo = F.normalize(
                        long_cap_i.mean(0, keepdim=True).unsqueeze(0), dim=-1
                    )  # (M_d, C) → (1, 1, C)

                    dense_attn = (long_cap_i_glo * img_spatial_embs_norm).sum(dim=-1)  # (B_v, N)

            # ---------------------------------------------------------
            # Step 4b: SDTPS Stage 1 - TokenSparse
            # ---------------------------------------------------------
            # 稀疏文本分支
            select_tokens_cap, extra_token_cap, score_mask_cap = self.sparse_net_cap(
                tokens=img_spatial_embs,                    # (B_v, N, C)
                attention_x=img_spatial_self_attention,     # (B_v, N)
                attention_y=attn_cap,                        # (B_v, N)
                attention_y_dense=dense_attn if self.use_paper_version else None,
                beta=self.beta,
                use_gumbel=self.use_gumbel_softmax,
                gumbel_tau=self.gumbel_tau,
            )
            # select_tokens_cap: (B_v, N, C) → (B_v, N_keep, C)
            # extra_token_cap: (B_v, 1, C)
            # score_mask_cap: (B_v, N)

            # 稠密文本分支
            if dense_attn is not None:
                select_tokens_long, extra_token_long, score_mask_long = self.sparse_net_long(
                    tokens=img_spatial_embs,                # (B_v, N, C)
                    attention_x=img_spatial_self_attention,
                    attention_y=dense_attn,
                    attention_y_dense=attn_cap if self.use_paper_version else None,
                    beta=self.beta,
                    use_gumbel=self.use_gumbel_softmax,
                    gumbel_tau=self.gumbel_tau,
                )
                # select_tokens_long: (B_v, N_keep, C)
                # extra_token_long: (B_v, 1, C)
                # score_mask_long: (B_v, N)

            # ---------------------------------------------------------
            # Step 4c: SDTPS Stage 2 - Aggregation
            # ---------------------------------------------------------
            if self.use_paper_version:
                # 论文版本: 双分支联合聚合
                if (self.use_dual_aggr and select_tokens_long is not None
                    and extra_token_long is not None):
                    # 双分支聚合: v̂ = W_s·v^s + W_d·v^d
                    aggr_tokens = self.aggr_net(
                        select_tokens_cap,      # (B_v, N_keep, C)
                        select_tokens_long,     # (B_v, N_keep, C)
                        None, None,             # mask不需要
                    )  # → (B_v, N_c, C)

                    # 融合extra token
                    extra_token = torch.stack(
                        [extra_token_cap, extra_token_long], dim=0
                    ).mean(dim=0)  # (2, B_v, 1, C) → (B_v, 1, C)

                    mask_pack: Union[torch.Tensor, Tuple[torch.Tensor, ...]] = (
                        score_mask_cap, score_mask_long,
                    )
                else:
                    aggr_tokens = self.aggr_net(select_tokens_cap)  # (B_v, N_keep, C) → (B_v, N_c, C)
                    extra_token = extra_token_cap  # (B_v, 1, C)
                    mask_pack = (score_mask_cap,)

                # 拼接聚合tokens和extra token
                keep_spatial_tokens = torch.cat([aggr_tokens, extra_token], dim=1)  # (B_v, N_c, C) + (B_v, 1, C) → (B_v, N_c+1, C)

                # 添加[CLS] token
                if self.has_cls_token:
                    select_tokens = torch.cat((img_cls_emb, keep_spatial_tokens), dim=1)  # (B_v, 1, C) + (B_v, N_c+1, C) → (B_v, N_c+2, C)
                else:
                    select_tokens = keep_spatial_tokens  # (B_v, N_c+1, C)

                select_tokens = F.normalize(select_tokens, dim=-1)  # 归一化

                # ---------------------------------------------------------
                # Step 4d: HRPA - 计算相似度
                # ---------------------------------------------------------
                sim_one_text = self.hrpa(
                    patch_features=select_tokens,   # (B_v, N_c+2, C) or (B_v, N_c+1, C)
                    word_features=cap_i_expand,     # (B_v, M, C)
                )  # → (B_v, 1)

                improve_sims.append(sim_one_text)  # 收集相似度
                score_mask_all.append(mask_pack)   # 收集决策矩阵

            else:
                # 开源代码版本: 两分支分别处理
                aggr_tokens = self.aggr_net(select_tokens_cap)  # (B_v, N_keep, C) → (B_v, N_c, C)
                keep_spatial_tokens = torch.cat([aggr_tokens, extra_token_cap], dim=1)  # (B_v, N_c+1, C)

                if self.has_cls_token:
                    select_tokens = torch.cat((img_cls_emb, keep_spatial_tokens), dim=1)  # (B_v, N_c+2, C)
                else:
                    select_tokens = keep_spatial_tokens

                select_tokens = F.normalize(select_tokens, dim=-1)

                # 稀疏文本相似度
                sim_one_text = self.hrpa(
                    patch_features=select_tokens,
                    word_features=cap_i_expand,
                )  # (B_v, 1)
                improve_sims.append(sim_one_text)
                score_mask_all.append(score_mask_cap)

                # 稠密文本相似度（开源代码也计算）
                if select_tokens_long is not None and long_cap_i_expand is not None:
                    aggr_tokens_long = self.aggr_net(select_tokens_long)  # (B_v, N_c, C)
                    keep_spatial_tokens = torch.cat([aggr_tokens_long, extra_token_long], dim=1)  # (B_v, N_c+1, C)

                    if self.has_cls_token:
                        select_tokens_long_final = torch.cat((img_cls_emb, keep_spatial_tokens), dim=1)  # (B_v, N_c+2, C)
                    else:
                        select_tokens_long_final = keep_spatial_tokens

                    select_tokens_long_final = F.normalize(select_tokens_long_final, dim=-1)

                    sim_one_text = self.hrpa(
                        patch_features=select_tokens_long_final,
                        word_features=long_cap_i_expand,  # (B_v, M_d, C)
                    )  # (B_v, 1)
                    long_sims.append(sim_one_text)
                    score_mask_long_all.append(score_mask_long)

        # =========================================================
        # Step 5: 整合结果
        # =========================================================
        improve_sims = torch.cat(improve_sims, dim=1)  # [(B_v, 1), ...] → (B_v, B_t)

        # 开源代码版本: 相似度相加
        if not self.use_paper_version and long_sims:
            improve_sims = improve_sims + torch.cat(long_sims, dim=1)  # (B_v, B_t) + (B_v, B_t) → (B_v, B_t)

        # 整理决策矩阵
        if self.use_paper_version:
            sparse_masks_list = [m[0] if isinstance(m, tuple) else m for m in score_mask_all]
            dense_masks_list = [m[1] for m in score_mask_all if isinstance(m, tuple) and len(m) > 1]

            sparse_stack = torch.stack(sparse_masks_list, dim=0)  # [(B_v, N), ...] → (B_t, B_v, N)
            dense_stack = (
                torch.stack(dense_masks_list, dim=0) if len(dense_masks_list) > 0 else None
            )

            score_mask_out: Union[torch.Tensor, Tuple[torch.Tensor, ...]]
            if dense_stack is not None:
                score_mask_out = (sparse_stack, dense_stack)  # (D_s, D_d)
            else:
                score_mask_out = (sparse_stack,)
        else:
            score_mask_out = torch.stack(score_mask_all, dim=0)  # (B_t, B_v, N)
            if score_mask_long_all:
                score_mask_out = score_mask_out + torch.stack(score_mask_long_all, dim=0)

        # 返回结果
        if self.training:
            return improve_sims, score_mask_out  # 训练模式: 返回(相似度, 决策矩阵)
        return improve_sims  # 推理模式: 只返回相似度


# =============================================================================
# 损失函数
# 对应论文 Section 3.3 末尾
# =============================================================================

class ContrastiveLoss(nn.Module):
    """
    对比损失 (Triplet Loss with Hard Negative Mining)

    功能: 拉近正样本对的相似度，推开负样本对的相似度

    ==================== 论文公式(6) ====================

    L_align = Σ_{(I,T)} ([α - S(I,T) + S(I,T̂)]_+ + [α - S(I,T) + S(Î,T)]_+)

    其中:
        - α: margin（边界值）
        - (I, T): 正样本图像-文本对
        - T̂ = argmax_{j≠T} S(I,j): 最难负样本文本
        - Î = argmax_{i≠I} S(i,T): 最难负样本图像
        - [x]_+ = max(x, 0): hinge函数

    Args:
        margin: 边界值 α，默认0.2
        max_violation: 是否使用hard negative mining

    Tensor流程:
        输入: scores (B, B)
        ↓ 提取对角线
        diagonal: (B, 1)
        ↓ 计算损失
        cost_s: (B, B) - 文本检索损失
        cost_im: (B, B) - 图像检索损失
        ↓ (如果max_violation)
        cost_s: (B,) - 只保留最难负样本
        cost_im: (B,)
        ↓ 求和
        输出: 标量
    """

    def __init__(self, margin: float = 0.2, max_violation: bool = False):
        super().__init__()
        self.margin = margin            # α
        self.max_violation = max_violation
        self.mask_repeat = True         # 处理一图多文情况

    def max_violation_on(self):
        """开启hard negative mining"""
        self.max_violation = True

    def max_violation_off(self):
        """关闭hard negative mining"""
        self.max_violation = False

    def forward(
        self,
        scores: torch.Tensor,                      # (B, B)
        img_ids: Optional[torch.Tensor] = None,    # (B,) or None
    ) -> torch.Tensor:
        """
        计算对比损失

        Args:
            scores: (B, B) - 相似度矩阵，scores[i,j] = S(I_i, T_j)
            img_ids: (B,) or None - 图像ID（处理一图多文）

        Returns:
            标量损失

        Tensor变化:
            scores: (B, B)
            ↓ diag
            diagonal: (B,) → (B, 1)
            ↓ expand
            d1, d2: (B, B)
            ↓ compute cost
            cost_s, cost_im: (B, B)
            ↓ masked_fill
            cost_s, cost_im: (B, B)
            ↓ (如果max_violation) max
            cost_s, cost_im: (B,)
            ↓ sum
            输出: 标量
        """
        # 对角线是正样本对的相似度
        diagonal = scores.diag().view(scores.size(0), 1)  # (B,) → (B, 1)
        d1 = diagonal.expand_as(scores)      # (B, 1) → (B, B) 按行扩展
        d2 = diagonal.t().expand_as(scores)  # (B, 1)^T → (B, B) 按列扩展

        # 文本检索损失: [α - S(I,T) + S(I,T̂)]_+
        cost_s = (self.margin + scores - d1).clamp(min=0)  # (B, B)

        # 图像检索损失: [α - S(I,T) + S(Î,T)]_+
        cost_im = (self.margin + scores - d2).clamp(min=0)  # (B, B)

        # 构建mask屏蔽正样本
        if not self.mask_repeat:
            mask = torch.eye(scores.size(0), dtype=torch.bool, device=scores.device)  # (B, B)
        else:
            if img_ids is not None:
                # 同一图像的多个caption都视为正样本
                mask = img_ids.unsqueeze(0) == img_ids.unsqueeze(1)  # (B,) → (B, B)
            else:
                mask = torch.eye(scores.size(0), dtype=torch.bool, device=scores.device)

        # 正样本位置损失设为0
        cost_s = cost_s.masked_fill_(mask, 0)    # (B, B)
        cost_im = cost_im.masked_fill_(mask, 0)

        # Hard Negative Mining
        if self.max_violation:
            # 只使用最难负样本（得分最高的负样本）
            cost_s = cost_s.max(dim=1)[0]    # (B, B) → (B,)
            cost_im = cost_im.max(dim=0)[0]  # (B, B) → (B,)

        return cost_s.sum() + cost_im.sum()  # 标量


class RatioLoss(nn.Module):
    """
    比例约束损失

    功能: 约束选择的patch数量接近目标比例ρ
          增强训练稳定性

    ==================== 论文公式(7) ====================

    L_ratio = (ρ - λ_1 · (1/N_s) Σ (D_s)_i - λ_2 · (1/N_d) Σ (D_d)_i)²

    其中:
        - ρ: 目标选择比例（超参数）
        - λ_1, λ_2: 稀疏/稠密文本的权重系数
        - D_s, D_d: 决策矩阵
        - (D_s)_i ∈ {0, 1}: 第i个patch是否被选中

    Args:
        target_ratio: 目标选择比例 ρ
        lambda_sparse: 稀疏文本权重 λ_1
        lambda_dense: 稠密文本权重 λ_2

    Tensor流程:
        输入: score_mask
            - 简化版: (B_t, B_v, N)
            - 论文版: tuple of (D_s, D_d)
        ↓ mean
        sparse_ratio: 标量
        dense_ratio: 标量
        ↓ compute loss
        输出: 标量
    """

    def __init__(
        self,
        target_ratio: float = 0.5,     # ρ
        lambda_sparse: float = 1.0,    # λ_1
        lambda_dense: float = 1.0,     # λ_2
    ):
        super().__init__()
        self.target_ratio = target_ratio
        self.lambda_sparse = lambda_sparse
        self.lambda_dense = lambda_dense

    def forward(
        self,
        score_mask: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    ) -> torch.Tensor:
        """
        计算比例约束损失

        Args:
            score_mask: 决策矩阵
                - Tensor: 简化版本
                - Tuple[D_s, D_d]: 论文版本

        Returns:
            标量损失

        Tensor变化:
            score_mask: (B_t, B_v, N) or tuple
            ↓ mean
            ratio: 标量
            ↓ compute MSE
            输出: 标量
        """
        if isinstance(score_mask, tuple) or isinstance(score_mask, list):
            # 论文版本: 分别计算稀疏和稠密的比例损失
            sparse_mask = score_mask[0]  # D_s: (B_t, B_v, N)
            dense_mask = score_mask[1] if len(score_mask) > 1 else None  # D_d

            # (ρ - (1/N_s) Σ (D_s)_i)²
            sparse_loss = (sparse_mask.float().mean() - self.target_ratio) ** 2  # 标量

            # (ρ - (1/N_d) Σ (D_d)_i)²
            dense_loss = (
                (dense_mask.float().mean() - self.target_ratio) ** 2
                if dense_mask is not None
                else torch.tensor(0.0, device=sparse_mask.device)
            )  # 标量

            # λ_1 · sparse_loss + λ_2 · dense_loss
            return self.lambda_sparse * sparse_loss + self.lambda_dense * dense_loss  # 标量

        # 简化版本
        return (score_mask.float().mean() - self.target_ratio) ** 2  # 标量


class SEPSLoss(nn.Module):
    """
    SEPS完整损失函数

    功能: 结合对比损失和比例约束损失

    ==================== 论文公式(7)末尾 ====================

    L = L_align + L_ratio

    其中:
        - L_align: 对比损失（公式6）
        - L_ratio: 比例约束损失（公式7）

    Args:
        margin: 对比损失的margin α
        target_ratio: 目标选择比例 ρ
        ratio_weight: L_ratio的权重
        max_violation: 是否使用hard negative mining
        lambda_sparse: 稀疏文本权重 λ_1
        lambda_dense: 稠密文本权重 λ_2

    Tensor流程:
        输入:
            similarity_matrix: (B_v, B_t)
            score_mask: 决策矩阵
        ↓ ContrastiveLoss
        align_loss: 标量
        ↓ RatioLoss
        ratio_loss: 标量
        ↓ 加权求和
        total_loss: 标量
        输出: (total_loss, align_loss, ratio_loss)
    """

    def __init__(
        self,
        margin: float = 0.2,           # α
        target_ratio: float = 0.5,     # ρ
        ratio_weight: float = 2.0,     # L_ratio权重
        max_violation: bool = False,
        lambda_sparse: float = 1.0,    # λ_1
        lambda_dense: float = 1.0,     # λ_2
    ):
        super().__init__()

        self.contrastive_loss = ContrastiveLoss(
            margin=margin,
            max_violation=max_violation,
        )
        self.ratio_loss = RatioLoss(
            target_ratio=target_ratio,
            lambda_sparse=lambda_sparse,
            lambda_dense=lambda_dense,
        )
        self.ratio_weight = ratio_weight

    def set_max_violation(self, max_violation: bool = True):
        """动态设置hard negative mining"""
        if max_violation:
            self.contrastive_loss.max_violation_on()
        else:
            self.contrastive_loss.max_violation_off()

    def forward(
        self,
        similarity_matrix: torch.Tensor,                           # (B_v, B_t)
        score_mask: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        img_ids: Optional[torch.Tensor] = None,                    # (B,) or None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算总损失

        公式: L = L_align + ratio_weight * L_ratio

        Args:
            similarity_matrix: (B_v, B_t) - 相似度矩阵
            score_mask: 决策矩阵
            img_ids: (B,) or None - 图像ID

        Returns:
            (total_loss, align_loss, ratio_loss)
            - total_loss: 总损失
            - align_loss: 对比损失（用于监控）
            - ratio_loss: 比例损失（用于监控）

        Tensor变化:
            similarity_matrix: (B_v, B_t)
            ↓ contrastive_loss
            align_loss: 标量

            score_mask: 决策矩阵
            ↓ ratio_loss
            ratio_loss: 标量

            ↓ 加权求和
            total_loss: 标量
        """
        align_loss = self.contrastive_loss(similarity_matrix, img_ids)  # 标量
        r_loss = self.ratio_loss(score_mask)                            # 标量
        total_loss = align_loss + self.ratio_weight * r_loss            # 标量

        return total_loss, align_loss, r_loss


# =============================================================================
# 便捷别名
# =============================================================================

SDTPS = CrossSparseAggrNet           # SDTPS模块别名
SDTPS_TokenSparse = TokenSparse      # TokenSparse别名
SDTPS_TokenAggregation = TokenAggregation  # TokenAggregation别名
HRPA_function = mask_xattn_one_text  # HRPA函数别名


def create_seps_model(
    embed_size: int = 512,
    num_patches: int = 196,
    sparse_ratio: float = 0.5,
    aggr_ratio: float = 0.4,
    use_paper_version: bool = USE_PAPER_VERSION_DEFAULT,
    use_gumbel_softmax: bool = False,
    gumbel_tau: float = 1.0,
) -> CrossSparseAggrNet:
    """
    创建SEPS模型的便捷工厂函数

    Args:
        embed_size: 特征维度 d
        num_patches: patch数量 N
        sparse_ratio: 选择比例 ρ
        aggr_ratio: 聚合比例
        use_paper_version: 是否使用论文完整机制
        use_gumbel_softmax: 是否使用Gumbel-Softmax
        gumbel_tau: Gumbel温度

    Returns:
        配置好的CrossSparseAggrNet实例
    """
    return CrossSparseAggrNet(
        embed_size=embed_size,
        num_patches=num_patches,
        sparse_ratio=sparse_ratio,
        aggr_ratio=aggr_ratio,
        use_paper_version=use_paper_version,
        use_gumbel_softmax=use_gumbel_softmax,
        gumbel_tau=gumbel_tau,
    )


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    """简单形状自检"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 测试参数
    B = 6
    num_patches = 196
    embed_size = 512
    sparse_len = 5
    dense_len = 20

    # 构造测试数据
    img = torch.randn(B, num_patches + 1, embed_size, device=device)
    cap = torch.randn(B, sparse_len, embed_size, device=device)
    cap_len = torch.full((B,), sparse_len, device=device)
    long_cap = torch.randn(B, dense_len, embed_size, device=device)
    long_len = torch.full((B,), dense_len, device=device)

    print("=" * 70)
    print("测试开源代码版本 (use_paper_version=False)")
    print("=" * 70)

    seps_code = create_seps_model(
        embed_size=embed_size,
        num_patches=num_patches,
        sparse_ratio=0.5,
        aggr_ratio=0.5,
        use_paper_version=False,
    ).to(device)
    seps_code.train()

    sims, mask = seps_code(img, cap, cap_len, long_cap, long_len)
    print(f"Similarity shape: {sims.shape}")
    print(f"Mask shape: {mask[0].shape if isinstance(mask, tuple) else mask.shape}")
    print(f"Parameters: {sum(p.numel() for p in seps_code.parameters()):,}")

    print("\n" + "=" * 70)
    print("测试论文版本 (use_paper_version=True)")
    print("=" * 70)

    seps_paper = create_seps_model(
        embed_size=embed_size,
        num_patches=num_patches,
        sparse_ratio=0.5,
        aggr_ratio=0.5,
        use_paper_version=True,
    ).to(device)
    seps_paper.train()

    sims, mask = seps_paper(img, cap, cap_len, long_cap, long_len)
    print(f"Similarity shape: {sims.shape}")
    print(f"Mask shape: {tuple(m.shape for m in mask) if isinstance(mask, tuple) else mask.shape}")
    print(f"Parameters: {sum(p.numel() for p in seps_paper.parameters()):,}")

    print("\n✓ 所有测试通过!")
