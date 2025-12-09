"""
SDTPS: Sparse and Dense Token-Aware Patch Selection - 完整修复版

严格按照以下三个来源实现：
1. 论文 tex: iclr2026_conference.tex
2. 论文版本代码: seps_modules_reviewed_v2_enhanced.py
3. 开源代码: seps(copy)/lib/cross_net.py

改动：将原始的"图像-文本"对齐改为"RGB-NIR-TIR"多模态对齐

v2 更新: 支持真正的 Cross-Attention 替代简单余弦相似度
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CrossModalAttention(nn.Module):
    """
    真正的 Cross-Attention 模块

    与简单余弦相似度的区别：
    - 余弦相似度：直接计算 patches 和 global 的相似度，无可学习参数
    - Cross-Attention：通过 Q, K, V 投影学习"用什么视角去看 patch"

    计算流程：
    Q = W_q @ patches     # (B, N, C) -> (B, N, C)
    K = W_k @ global      # (B, 1, C) -> (B, 1, C)
    V = W_v @ global      # (B, 1, C) -> (B, 1, C)

    Attention = softmax(Q @ K^T / sqrt(d))  # (B, N, 1)
    Output = Attention @ V                   # (B, N, C)

    但我们只需要 attention score 作为重要性分数，不需要 output
    所以简化为：score = mean(Q @ K^T / sqrt(d), dim=-1)  # (B, N)
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 4,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Q, K 投影（我们只需要 attention score，不需要 V）
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)

        # 输出投影：将多头得分合并为单一得分
        self.score_proj = nn.Linear(num_heads, 1, bias=False)

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        # score_proj 初始化为均匀权重
        nn.init.constant_(self.score_proj.weight, 1.0 / self.num_heads)
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
        if self.k_proj.bias is not None:
            nn.init.zeros_(self.k_proj.bias)

    def forward(
        self,
        patches: torch.Tensor,       # (B, N, C) - 查询
        global_feat: torch.Tensor,   # (B, C) or (B, 1, C) - 键
    ) -> torch.Tensor:
        """
        计算 Cross-Attention score

        Args:
            patches: (B, N, C) - patch tokens 作为 Query
            global_feat: (B, C) or (B, 1, C) - global feature 作为 Key

        Returns:
            score: (B, N) - 每个 patch 的重要性得分
        """
        B, N, C = patches.shape

        if global_feat.dim() == 2:
            global_feat = global_feat.unsqueeze(1)  # (B, C) -> (B, 1, C)

        # Q 投影: patches 作为查询   q k 反过来。
        q = self.q_proj(global_feat)  # (B, N, C)
        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # (B, num_heads, N, head_dim)

        # K 投影: global 作为键
        k = self.k_proj(patches)  # (B, 1, C)
        k = k.reshape(B, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # (B, num_heads, 1, head_dim)

        # Attention score: Q @ K^T / sqrt(d),
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, 1)
        attn = attn.softmax(dim=-1) # 再N的维度做softmax
        attn = attn.squeeze(-1)  # (B, num_heads, N)

        # 可选：应用 dropout（训练时）
        attn = self.attn_drop(attn)

        # 合并多头得分为单一得分
        attn = attn.permute(0, 2, 1)  # (B, N, num_heads)
       # score = self.score_proj(attn).squeeze(-1)  # (B, N)
        attn = attn.mean(-1) # 利用余弦相似度分数，乘以可学习矩阵 对注意力分数做门控

        return attn


class TokenSparse(nn.Module):
    """
    Token 稀疏选择模块

    对应论文公式 1-3:
    - 公式1: s_i^p = σ(MLP(v_i))
    - 公式2: s_i^{st}, s_i^{dt}, s_i^{im}
    - 公式3: s_i = (1-2β)·s_i^p + β·(s_i^{st} + s_i^{dt} + 2·s_i^{im})

    输出: N → N_s (选择sparse_ratio比例的patches)
    """

    def __init__(
        self,
        embed_dim: int = 512,
        sparse_ratio: float = 0.5,  # 和原论文一致
        use_gumbel: bool = False,
        gumbel_tau: float = 1.0,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.sparse_ratio = sparse_ratio
        self.use_gumbel = use_gumbel
        self.gumbel_tau = gumbel_tau

        # 公式1: MLP Score Predictor
        self.score_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        tokens: torch.Tensor,                # (B, N, C)
        self_attention: torch.Tensor,        # (B, N) - s^{im}
        cross_attention_m2: torch.Tensor,    # (B, N) - 模态2
        cross_attention_m3: torch.Tensor,    # (B, N) - 模态3
        beta: float = 0.25,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Token 选择（严格按照论文）

        Args:
            tokens: (B, N, C)
            self_attention: (B, N) - 对应 s^{im}
            cross_attention_m2: (B, N) - 对应 s^{st}
            cross_attention_m3: (B, N) - 对应 s^{dt}
            beta: 权重参数

        Returns:
            select_tokens: (B, N_s, C) - 选中的patches
            extra_token: (B, 1, C) - 融合的冗余patches
            score_mask: (B, N) - 完整的决策矩阵 D
            selected_importance: (B, N_s) - 选中patches的重要性得分（softmax归一化）
            keep_indices: (B, N_s) - 选中patches的索引
        """
        B, N, C = tokens.size()

        # ========== 公式1: MLP 预测 ==========
        s_pred = self.score_predictor(tokens).squeeze(-1)  # (B, N)  这个丢掉

        # ========== 公式2: 归一化注意力得分 ==========
        def normalize_score(s: torch.Tensor) -> torch.Tensor:
            """Min-Max归一化到[0,1]"""
            s_min = s.min(dim=-1, keepdim=True)[0]
            s_max = s.max(dim=-1, keepdim=True)[0]
            return (s - s_min) / (s_max - s_min + 1e-8)

        s_im = normalize_score(self_attention)
        s_m2 = normalize_score(cross_attention_m2)
        s_m3 = normalize_score(cross_attention_m3)

        # ========== 公式3: 综合得分 ==========
       # score =  beta * (s_m2 + s_m3 + 2 * s_im)
        score = (s_m2 + s_m3 + s_im) / 3

        # ========== Decision: 生成决策矩阵 D ==========
        num_keep = max(1, math.ceil(N * self.sparse_ratio))

        # 排序获取Top-K
        score_sorted, score_indices = torch.sort(score, dim=1, descending=True)
        keep_policy = score_indices[:, :num_keep]  # (B, N_s)

        # Gumbel-Softmax（可选）
        if self.training and self.use_gumbel:
            # Gumbel噪声（添加裁剪避免极端值）
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(score) + 1e-9) + 1e-9)
            gumbel_noise = torch.clamp(gumbel_noise, min=-5.0, max=5.0)  # 裁剪到[-5, 5]
            soft_mask = F.softmax((score + gumbel_noise) / self.gumbel_tau, dim=1)

            # Hard mask
            hard_mask = torch.zeros_like(score).scatter(1, keep_policy, 1.0)

            # Straight-Through Estimator
            score_mask = hard_mask + (soft_mask - soft_mask.detach())
        else:
            # 标准 Top-K  这里直接点成而不是选出来
            score_mask = torch.zeros_like(score).scatter(1, keep_policy, 1.0)

        # ========== Selection: 提取选中的patches ==========   selection buyao  直接点乘得到返回带0 的token
        select_tokens = torch.gather(
            tokens, dim=1,
            index=keep_policy.unsqueeze(-1).expand(-1, -1, C)
        )  # (B, N_s, C)

        # ========== 融合被丢弃的patches ==========
        non_keep_policy = score_indices[:, num_keep:]
        non_tokens = torch.gather(
            tokens, dim=1,
            index=non_keep_policy.unsqueeze(-1).expand(-1, -1, C)
        )  # (B, N-N_s, C)

        non_keep_score = score_sorted[:, num_keep:]
        non_keep_score = F.softmax(non_keep_score, dim=1).unsqueeze(-1)
        extra_token = torch.sum(non_tokens * non_keep_score, dim=1, keepdim=True)  # (B, 1, C)

        # ========== 修复：传递真正的软重要性得分 ==========
        # 使用排序后的 Top-K 得分（softmax归一化），而非二值 mask
        # 这样 Gumbel 的软权重可以影响聚合过程
        selected_scores = score_sorted[:, :num_keep]  # (B, N_s) - 排序后的 Top-K 得分
        selected_importance = F.softmax(selected_scores, dim=1)  # (B, N_s) - 归一化为权重

        return select_tokens, extra_token, score_mask, selected_importance, keep_policy


class TokenAggregation(nn.Module):  # 去掉这部分
    """
    Token 聚合模块

    对应论文公式4（单分支版本）:
    v̂_j = Σ_{i=1}^{N_s} W_{ij} · v_i^s

    其中:
    - W ∈ R^{N_c × N_s}: 聚合权重矩阵
    - W = Softmax(MLP(V_s))
    - Σ_i W_{ji} = 1 (每行归一化)

    修复：支持通过 importance_weights 引入软权重
    聚合时考虑每个 token 的重要性得分

    输出: N_s → N_c (进一步压缩)
    """

    def __init__(
        self,
        dim: int = 512,
        keeped_patches: int = 26,  # N_c
        dim_ratio: float = 0.2,
    ):
        super().__init__()

        hidden_dim = int(dim * dim_ratio)  # 512 × 0.2 = 102

        # MLP 生成聚合权重
        self.weight = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),      # 512 → 102
            nn.GELU(),
            nn.Linear(hidden_dim, keeped_patches),  # 102 → N_c
        )

        self.scale = nn.Parameter(torch.ones(1, 1, 1))

    def forward(
        self,
        x: torch.Tensor,                           # (B, N_s, C)
        importance_weights: Optional[torch.Tensor] = None,  # (B, N_s) - 重要性权重
    ) -> torch.Tensor:
        """
        聚合tokens

        Args:
            x: (B, N_s, C) - 选中的tokens
            importance_weights: (B, N_s) or None - 每个token的重要性权重（softmax归一化）

        Returns:
            (B, N_c, C) - 聚合后的tokens

        流程:
            x (B, N_s, C)
              ↓ MLP
            weight (B, N_s, N_c)
              ↓ transpose
            weight (B, N_c, N_s)
              ↓ (可选) 乘以 importance_weights
              ↓ softmax
            weight (B, N_c, N_s), Σ_i W[b,j,i]=1
              ↓ bmm
            output (B, N_c, C)
        """
        # 生成聚合权重 logits
        weight = self.weight(x)               # (B, N_s, C) → (B, N_s, N_c)
        weight = weight.transpose(2, 1)       # (B, N_s, N_c) → (B, N_c, N_s)
        weight = weight * self.scale          # 可学习缩放

        # 如果有 importance_weights，用它来调整聚合权重
        # 重要性高的 token 在聚合时获得更大的权重
        if importance_weights is not None:
            importance_weights = importance_weights.unsqueeze(1)  # (B, N_s) → (B, 1, N_s)
            # 将重要性权重加到 logits 上（log 空间相加 = 概率空间相乘）
            weight = weight + torch.log(importance_weights + 1e-8)

        # Softmax归一化（保证每个聚合token的权重和为1）
        weight = F.softmax(weight, dim=2)     # (B, N_c, N_s)

        # 批量矩阵乘法: v̂_j = Σ_i W_{ji} × v_i
        return torch.bmm(weight, x)           # (B, N_c, N_s) @ (B, N_s, C) → (B, N_c, C)


class MultiModalSDTPS(nn.Module):
    """
    多模态 SDTPS 模块 - 完整修复版

    完整流程（以RGB为例）:
        RGB_cash (B, 128, 512)
          ↓ TokenSparse
        select_tokens (B, 64, 512)  ← sparse_ratio=0.5
          ↓ TokenAggregation
        aggr_tokens (B, 26, 512)    ← aggr_ratio=0.4
          ↓ +extra_token
        enhanced (B, 27, 512)

    参数设置（和原论文一致）:
    - sparse_ratio = 0.5
    - aggr_ratio = 0.4
    - 最终比例 = 0.5 × 0.4 = 0.2 (20%)

    原论文: 196 → 98 → 39 (比例0.199)
    我们: 128 → 64 → 26 (比例0.203)

    v2 更新: 支持两种 Cross-Attention 类型
    - 'cosine': 原始余弦相似度（无可学习参数）
    - 'attention': 真正的多头 Cross-Attention（有可学习参数）
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_patches: int = 128,      # 输入patch数量
        sparse_ratio: float = 0.5,   # 和原论文一致
        aggr_ratio: float = 0.4,     # 和原论文一致
        use_gumbel: bool = False,
        gumbel_tau: float = 1.0,
        beta: float = 0.25,
        dim_ratio: float = 0.2,      # aggregation MLP的隐藏层比例
        cross_attn_type: str = 'cosine',  # 'cosine' or 'attention'
        cross_attn_heads: int = 4,   # Cross-Attention 的头数
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.sparse_ratio = sparse_ratio
        self.aggr_ratio = aggr_ratio
        self.beta = beta
        self.cross_attn_type = cross_attn_type

        # 计算聚合后的patch数量 N_c
        self.keeped_patches = int(num_patches * aggr_ratio * sparse_ratio)
        # = int(128 × 0.4 × 0.5) = 25

        print(f"[SDTPS] 参数设置:")
        print(f"  输入patches: {num_patches}")
        print(f"  sparse_ratio: {sparse_ratio} → 选中 {math.ceil(num_patches * sparse_ratio)} patches")
        print(f"  aggr_ratio: {aggr_ratio} → 聚合到 {self.keeped_patches} patches")
        print(f"  最终比例: {self.keeped_patches / num_patches:.3f}")
        print(f"  cross_attn_type: {cross_attn_type}")
        if cross_attn_type == 'attention':
            print(f"  cross_attn_heads: {cross_attn_heads}")

        # ========== Cross-Attention 模块（如果使用 'attention' 类型） ==========
        if cross_attn_type == 'attention':
            # RGB 模态的 cross-attention
            self.rgb_cross_nir = CrossModalAttention(embed_dim, cross_attn_heads)
            self.rgb_cross_tir = CrossModalAttention(embed_dim, cross_attn_heads)
            self.rgb_self_attn = CrossModalAttention(embed_dim, cross_attn_heads)

            # NIR 模态的 cross-attention
            self.nir_cross_rgb = CrossModalAttention(embed_dim, cross_attn_heads)
            self.nir_cross_tir = CrossModalAttention(embed_dim, cross_attn_heads)
            self.nir_self_attn = CrossModalAttention(embed_dim, cross_attn_heads)

            # TIR 模态的 cross-attention
            self.tir_cross_rgb = CrossModalAttention(embed_dim, cross_attn_heads)
            self.tir_cross_nir = CrossModalAttention(embed_dim, cross_attn_heads)
            self.tir_self_attn = CrossModalAttention(embed_dim, cross_attn_heads)

        # ========== RGB 模态 ==========
        # Stage 1: TokenSparse
        self.rgb_sparse = TokenSparse(
            embed_dim=embed_dim,
            sparse_ratio=sparse_ratio,
            use_gumbel=use_gumbel,
            gumbel_tau=gumbel_tau,
        )
        # Stage 2: TokenAggregation
        # self.rgb_aggr = TokenAggregation(
        #     dim=embed_dim,
        #     keeped_patches=self.keeped_patches,
        #     dim_ratio=dim_ratio,
        # )

        # ========== NIR 模态 ==========
        self.nir_sparse = TokenSparse(
            embed_dim=embed_dim,
            sparse_ratio=sparse_ratio,
            use_gumbel=use_gumbel,
            gumbel_tau=gumbel_tau,
        )
        # self.nir_aggr = TokenAggregation(
        #     dim=embed_dim,
        #     keeped_patches=self.keeped_patches,
        #     dim_ratio=dim_ratio,
        # )

        # ========== TIR 模态 ==========
        self.tir_sparse = TokenSparse(
            embed_dim=embed_dim,
            sparse_ratio=sparse_ratio,
            use_gumbel=use_gumbel,
            gumbel_tau=gumbel_tau,
        )
        # self.tir_aggr = TokenAggregation(
        #     dim=embed_dim,
        #     keeped_patches=self.keeped_patches,
        #     dim_ratio=dim_ratio,
        # )

    def _compute_self_attention_cosine(
        self,
        patches: torch.Tensor,       # (B, N, C)
        global_feat: torch.Tensor,   # (B, C)
    ) -> torch.Tensor:
        """
        计算自注意力 s^{im} - 余弦相似度版本（无可学习参数）

        cosine_sim(a, b) = (a · b) / (||a|| * ||b||)
        由于先做 L2 归一化，||a|| = ||b|| = 1，所以点积 = 余弦相似度

        注意：移除了 no_grad 以允许 Backbone finetune
        原 SEPS 论文使用 no_grad 因为 Backbone 冻结
        但重识别任务需要 Backbone 学习判别性特征
        """
        if global_feat.dim() == 2:
            global_feat = global_feat.unsqueeze(1)

        # L2 归一化后，点积 = 余弦相似度
        patches_norm = F.normalize(patches, dim=-1)  # ||patches_norm|| = 1
        global_norm = F.normalize(global_feat, dim=-1)  # ||global_norm|| = 1

        # 归一化向量的点积 = 余弦相似度
        self_attn = (patches_norm * global_norm).sum(dim=-1)

        return self_attn  # (B, N)

    def _compute_cross_attention_cosine(
        self,
        patches: torch.Tensor,       # (B, N, C)
        cross_global: torch.Tensor,  # (B, C)
    ) -> torch.Tensor:
        """
        计算交叉注意力 s^{cross} - 余弦相似度版本（无可学习参数）

        cosine_sim(a, b) = (a · b) / (||a|| * ||b||)
        由于先做 L2 归一化，||a|| = ||b|| = 1，所以点积 = 余弦相似度

        注意：移除了 no_grad 以允许跨模态学习
        允许梯度在不同模态之间传播，实现跨模态引导的特征学习
        """
        if cross_global.dim() == 2:
            cross_global = cross_global.unsqueeze(1)

        # L2 归一化后，点积 = 余弦相似度
        patches_norm = F.normalize(patches, dim=-1)  # ||patches_norm|| = 1
        cross_norm = F.normalize(cross_global, dim=-1)  # ||cross_norm|| = 1

        # 归一化向量的点积 = 余弦相似度
        cross_attn = (patches_norm * cross_norm).sum(dim=-1)

        return cross_attn  # (B, N)

    def forward(
        self,
        RGB_cash: torch.Tensor,      # (B, N, C)
        NI_cash: torch.Tensor,       # (B, N, C)
        TI_cash: torch.Tensor,       # (B, N, C)
        RGB_global: torch.Tensor,    # (B, C)
        NI_global: torch.Tensor,     # (B, C)
        TI_global: torch.Tensor,     # (B, C)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        多模态 SDTPS 完整流程

        Returns:
            RGB_enhanced: (B, N_c+1, C)
            NI_enhanced: (B, N_c+1, C)
            TI_enhanced: (B, N_c+1, C)
            rgb_mask: (B, N)
            nir_mask: (B, N)
            tir_mask: (B, N)
        """

        # ==================== RGB 模态 ====================
        # Step 1: 计算attention scores
        if self.cross_attn_type == 'attention':
            # 使用真正的 Cross-Attention  都要加起来，normaliz
            rgb_self_attn = self.rgb_self_attn(RGB_cash, RGB_global)
            rgb_nir_cross = self.rgb_cross_nir(RGB_cash, NI_global)
            rgb_tir_cross = self.rgb_cross_tir(RGB_cash, TI_global)
        else:
            # 使用余弦相似度
            rgb_self_attn = self._compute_self_attention_cosine(RGB_cash, RGB_global)
            rgb_nir_cross = self._compute_cross_attention_cosine(RGB_cash, NI_global)
            rgb_tir_cross = self._compute_cross_attention_cosine(RGB_cash, TI_global)

        # Step 2: TokenSparse - 选择显著patches
        rgb_select, rgb_extra, rgb_mask, rgb_importance, rgb_indices = self.rgb_sparse(
            tokens=RGB_cash,
            self_attention=rgb_self_attn,
            cross_attention_m2=rgb_nir_cross,
            cross_attention_m3=rgb_tir_cross,
            beta=self.beta,
        )  # (B, N_s, C), (B, 1, C), (B, N), (B, N_s), (B, N_s)

        # Step 3: TokenAggregation - 聚合patches（传递重要性权重！）
        rgb_aggr = self.rgb_aggr(
            x=rgb_select,
            importance_weights=rgb_importance  # ← 使用软重要性权重
        )  # (B, N_s, C) → (B, N_c, C)

        # Step 4: 拼接聚合tokens和extra token
        RGB_enhanced = torch.cat([rgb_aggr, rgb_extra], dim=1)  # (B, N_c+1, C)

        # ==================== NIR 模态 ====================
        if self.cross_attn_type == 'attention':
            nir_self_attn = self.nir_self_attn(NI_cash, NI_global)
            nir_rgb_cross = self.nir_cross_rgb(NI_cash, RGB_global)
            nir_tir_cross = self.nir_cross_tir(NI_cash, TI_global)
        else:
            nir_self_attn = self._compute_self_attention_cosine(NI_cash, NI_global)
            nir_rgb_cross = self._compute_cross_attention_cosine(NI_cash, RGB_global)
            nir_tir_cross = self._compute_cross_attention_cosine(NI_cash, TI_global)

        nir_select, nir_extra, nir_mask, nir_importance, nir_indices = self.nir_sparse(
            tokens=NI_cash,
            self_attention=nir_self_attn,
            cross_attention_m2=nir_rgb_cross,
            cross_attention_m3=nir_tir_cross,
            beta=self.beta,
        )

        nir_aggr = self.nir_aggr(
            x=nir_select,
            importance_weights=nir_importance  # ← 使用软重要性权重
        )
        NI_enhanced = torch.cat([nir_aggr, nir_extra], dim=1)  # (B, N_c+1, C)

        # ==================== TIR 模态 ====================
        if self.cross_attn_type == 'attention':
            tir_self_attn = self.tir_self_attn(TI_cash, TI_global)
            tir_rgb_cross = self.tir_cross_rgb(TI_cash, RGB_global)
            tir_nir_cross = self.tir_cross_nir(TI_cash, NI_global)
        else:
            tir_self_attn = self._compute_self_attention_cosine(TI_cash, TI_global)
            tir_rgb_cross = self._compute_cross_attention_cosine(TI_cash, RGB_global)
            tir_nir_cross = self._compute_cross_attention_cosine(TI_cash, NI_global)

        tir_select, tir_extra, tir_mask, tir_importance, tir_indices = self.tir_sparse(
            tokens=TI_cash,
            self_attention=tir_self_attn,
            cross_attention_m2=tir_rgb_cross,
            cross_attention_m3=tir_nir_cross,
            beta=self.beta,
        )

        tir_aggr = self.tir_aggr(
            x=tir_select,
            importance_weights=tir_importance  # ← 使用软重要性权重
        )
        TI_enhanced = torch.cat([tir_aggr, tir_extra], dim=1)  # (B, N_c+1, C)

        return RGB_enhanced, NI_enhanced, TI_enhanced, rgb_mask, nir_mask, tir_mask
