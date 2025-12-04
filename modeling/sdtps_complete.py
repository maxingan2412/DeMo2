"""
SDTPS: Sparse and Dense Token-Aware Patch Selection - 完整修复版

严格按照以下三个来源实现：
1. 论文 tex: iclr2026_conference.tex
2. 论文版本代码: seps_modules_reviewed_v2_enhanced.py
3. 开源代码: seps(copy)/lib/cross_net.py

改动：将原始的"图像-文本"对齐改为"RGB-NIR-TIR"多模态对齐
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


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
            selected_mask: (B, N_s) - 选中patches对应的mask值（传给aggregation）
            keep_indices: (B, N_s) - 选中patches的索引
        """
        B, N, C = tokens.size()

        # ========== 公式1: MLP 预测 ==========
        s_pred = self.score_predictor(tokens).squeeze(-1)  # (B, N)

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
        score = (1 - 2 * beta) * s_pred + beta * (s_m2 + s_m3 + 2 * s_im)

        # ========== Decision: 生成决策矩阵 D ==========
        num_keep = max(1, math.ceil(N * self.sparse_ratio))

        # 排序获取Top-K
        score_sorted, score_indices = torch.sort(score, dim=1, descending=True)
        keep_policy = score_indices[:, :num_keep]  # (B, N_s)

        # Gumbel-Softmax（可选）
        if self.training and self.use_gumbel:
            # Gumbel噪声
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(score) + 1e-9) + 1e-9)
            soft_mask = F.softmax((score + gumbel_noise) / self.gumbel_tau, dim=1)

            # Hard mask
            hard_mask = torch.zeros_like(score).scatter(1, keep_policy, 1.0)

            # Straight-Through Estimator
            score_mask = hard_mask + (soft_mask - soft_mask.detach())
        else:
            # 标准 Top-K
            score_mask = torch.zeros_like(score).scatter(1, keep_policy, 1.0)

        # ========== Selection: 提取选中的patches ==========
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

        # ========== 提取选中patches对应的mask值 ==========
        # 用于传递给TokenAggregation（论文第169行要求）
        selected_mask = torch.gather(score_mask, dim=1, index=keep_policy)  # (B, N_s)

        return select_tokens, extra_token, score_mask, selected_mask, keep_policy


class TokenAggregation(nn.Module):
    """
    Token 聚合模块

    对应论文公式4（单分支版本）:
    v̂_j = Σ_{i=1}^{N_s} W_{ij} · v_i^s

    其中:
    - W ∈ R^{N_c × N_s}: 聚合权重矩阵
    - W = Softmax(MLP(V_s))
    - Σ_i W_{ji} = 1 (每行归一化)

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
        x: torch.Tensor,                      # (B, N_s, C)
        keep_policy: Optional[torch.Tensor] = None,  # (B, N_s) - 可选的mask
    ) -> torch.Tensor:
        """
        聚合tokens

        Args:
            x: (B, N_s, C) - 选中的tokens
            keep_policy: (B, N_s) or None - 决策矩阵D（可选）

        Returns:
            (B, N_c, C) - 聚合后的tokens

        流程:
            x (B, N_s, C)
              ↓ MLP
            weight (B, N_s, N_c)
              ↓ transpose
            weight (B, N_c, N_s)
              ↓ (可选) mask by keep_policy
              ↓ softmax
            weight (B, N_c, N_s), Σ_i W[b,j,i]=1
              ↓ bmm
            output (B, N_c, C)
        """
        # 生成聚合权重 logits
        weight = self.weight(x)               # (B, N_s, C) → (B, N_s, N_c)
        weight = weight.transpose(2, 1)       # (B, N_s, N_c) → (B, N_c, N_s)
        weight = weight * self.scale          # 可学习缩放

        # 如果有keep_policy，用它来mask无效位置
        if keep_policy is not None:
            keep_policy = keep_policy.unsqueeze(1)  # (B, N_s) → (B, 1, N_s)
            weight = weight - (1 - keep_policy) * 1e10  # mask

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
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.sparse_ratio = sparse_ratio
        self.aggr_ratio = aggr_ratio
        self.beta = beta

        # 计算聚合后的patch数量 N_c
        self.keeped_patches = int(num_patches * aggr_ratio * sparse_ratio)
        # = int(128 × 0.4 × 0.5) = 25

        print(f"[SDTPS] 参数设置:")
        print(f"  输入patches: {num_patches}")
        print(f"  sparse_ratio: {sparse_ratio} → 选中 {math.ceil(num_patches * sparse_ratio)} patches")
        print(f"  aggr_ratio: {aggr_ratio} → 聚合到 {self.keeped_patches} patches")
        print(f"  最终比例: {self.keeped_patches / num_patches:.3f}")

        # ========== RGB 模态 ==========
        # Stage 1: TokenSparse
        self.rgb_sparse = TokenSparse(
            embed_dim=embed_dim,
            sparse_ratio=sparse_ratio,
            use_gumbel=use_gumbel,
            gumbel_tau=gumbel_tau,
        )
        # Stage 2: TokenAggregation
        self.rgb_aggr = TokenAggregation(
            dim=embed_dim,
            keeped_patches=self.keeped_patches,
            dim_ratio=dim_ratio,
        )

        # ========== NIR 模态 ==========
        self.nir_sparse = TokenSparse(
            embed_dim=embed_dim,
            sparse_ratio=sparse_ratio,
            use_gumbel=use_gumbel,
            gumbel_tau=gumbel_tau,
        )
        self.nir_aggr = TokenAggregation(
            dim=embed_dim,
            keeped_patches=self.keeped_patches,
            dim_ratio=dim_ratio,
        )

        # ========== TIR 模态 ==========
        self.tir_sparse = TokenSparse(
            embed_dim=embed_dim,
            sparse_ratio=sparse_ratio,
            use_gumbel=use_gumbel,
            gumbel_tau=gumbel_tau,
        )
        self.tir_aggr = TokenAggregation(
            dim=embed_dim,
            keeped_patches=self.keeped_patches,
            dim_ratio=dim_ratio,
        )

    def _compute_self_attention(
        self,
        patches: torch.Tensor,       # (B, N, C)
        global_feat: torch.Tensor,   # (B, C)
    ) -> torch.Tensor:
        """
        计算自注意力 s^{im}

        注意：移除了 no_grad 以允许 Backbone finetune
        原 SEPS 论文使用 no_grad 因为 Backbone 冻结
        但重识别任务需要 Backbone 学习判别性特征
        """
        if global_feat.dim() == 2:
            global_feat = global_feat.unsqueeze(1)

        # 移除 no_grad 以允许梯度传播到 Backbone
        # L2归一化
        patches_norm = F.normalize(patches, dim=-1)
        global_norm = F.normalize(global_feat, dim=-1)

        # 点积相似度
        self_attn = (patches_norm * global_norm).sum(dim=-1)

        return self_attn  # (B, N)

    def _compute_cross_attention(
        self,
        patches: torch.Tensor,       # (B, N, C)
        cross_global: torch.Tensor,  # (B, C)
    ) -> torch.Tensor:
        """
        计算交叉注意力 s^{st} / s^{dt}

        注意：移除了 no_grad 以允许跨模态学习
        允许梯度在不同模态之间传播，实现跨模态引导的特征学习
        """
        if cross_global.dim() == 2:
            cross_global = cross_global.unsqueeze(1)

        # 移除 no_grad 以允许跨模态梯度传播
        patches_norm = F.normalize(patches, dim=-1)
        cross_norm = F.normalize(cross_global, dim=-1)
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
        rgb_self_attn = self._compute_self_attention(RGB_cash, RGB_global)
        rgb_nir_cross = self._compute_cross_attention(RGB_cash, NI_global)
        rgb_tir_cross = self._compute_cross_attention(RGB_cash, TI_global)

        # Step 2: TokenSparse - 选择显著patches
        rgb_select, rgb_extra, rgb_mask, rgb_selected_mask, rgb_indices = self.rgb_sparse(
            tokens=RGB_cash,
            self_attention=rgb_self_attn,
            cross_attention_m2=rgb_nir_cross,
            cross_attention_m3=rgb_tir_cross,
            beta=self.beta,
        )  # (B, N_s, C), (B, 1, C), (B, N), (B, N_s), (B, N_s)

        # Step 3: TokenAggregation - 聚合patches（传递 selected_mask！）
        rgb_aggr = self.rgb_aggr(
            x=rgb_select,
            keep_policy=rgb_selected_mask  # ← 关键：传递 Gumbel 生成的 mask！
        )  # (B, N_s, C) → (B, N_c, C)

        # Step 4: 拼接聚合tokens和extra token
        RGB_enhanced = torch.cat([rgb_aggr, rgb_extra], dim=1)  # (B, N_c+1, C)

        # ==================== NIR 模态 ====================
        nir_self_attn = self._compute_self_attention(NI_cash, NI_global)
        nir_rgb_cross = self._compute_cross_attention(NI_cash, RGB_global)
        nir_tir_cross = self._compute_cross_attention(NI_cash, TI_global)

        nir_select, nir_extra, nir_mask, nir_selected_mask, nir_indices = self.nir_sparse(
            tokens=NI_cash,
            self_attention=nir_self_attn,
            cross_attention_m2=nir_rgb_cross,
            cross_attention_m3=nir_tir_cross,
            beta=self.beta,
        )

        nir_aggr = self.nir_aggr(
            x=nir_select,
            keep_policy=nir_selected_mask  # ← 传递 mask
        )
        NI_enhanced = torch.cat([nir_aggr, nir_extra], dim=1)  # (B, N_c+1, C)

        # ==================== TIR 模态 ====================
        tir_self_attn = self._compute_self_attention(TI_cash, TI_global)
        tir_rgb_cross = self._compute_cross_attention(TI_cash, RGB_global)
        tir_nir_cross = self._compute_cross_attention(TI_cash, NI_global)

        tir_select, tir_extra, tir_mask, tir_selected_mask, tir_indices = self.tir_sparse(
            tokens=TI_cash,
            self_attention=tir_self_attn,
            cross_attention_m2=tir_rgb_cross,
            cross_attention_m3=tir_nir_cross,
            beta=self.beta,
        )

        tir_aggr = self.tir_aggr(
            x=tir_select,
            keep_policy=tir_selected_mask  # ← 传递 mask
        )
        TI_enhanced = torch.cat([tir_aggr, tir_extra], dim=1)  # (B, N_c+1, C)

        return RGB_enhanced, NI_enhanced, TI_enhanced, rgb_mask, nir_mask, tir_mask
