"""
SDTPS: 修复版本

修复内容：
1. 在 attention 计算中添加 with torch.no_grad()
2. 实现真正的 Gumbel-Softmax 可微采样
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TokenSparse(nn.Module):
    """
    Token 稀疏选择模块 - 修复版

    修复内容：
    1. Attention 计算使用 no_grad
    2. Gumbel-Softmax 提供真正的可微采样
    """

    def __init__(
        self,
        embed_dim: int = 512,
        sparse_ratio: float = 0.6,
        use_gumbel: bool = False,
        gumbel_tau: float = 1.0,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.sparse_ratio = sparse_ratio
        self.use_gumbel = use_gumbel
        self.gumbel_tau = gumbel_tau

        # MLP Score Predictor
        self.score_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        self_attention: torch.Tensor,
        cross_attention_m2: torch.Tensor,
        cross_attention_m3: torch.Tensor,
        beta: float = 0.25,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Token selection with Gumbel-Softmax

        Args:
            tokens: (B, N, C)
            self_attention: (B, N)
            cross_attention_m2: (B, N)
            cross_attention_m3: (B, N)
            beta: score combination weight

        Returns:
            select_tokens: (B, K, C) - selected tokens
            extra_token: (B, 1, C) - aggregated redundant tokens
            score_mask: (B, N) - decision matrix
        """
        B, N, C = tokens.size()

        # ========== Step 1: 计算综合得分 ==========
        s_pred = self.score_predictor(tokens).squeeze(-1)

        def normalize_score(s: torch.Tensor) -> torch.Tensor:
            s_min = s.min(dim=-1, keepdim=True)[0]
            s_max = s.max(dim=-1, keepdim=True)[0]
            return (s - s_min) / (s_max - s_min + 1e-8)

        s_im = normalize_score(self_attention)
        s_m2 = normalize_score(cross_attention_m2)
        s_m3 = normalize_score(cross_attention_m3)

        score = (1 - 2 * beta) * s_pred + beta * (s_m2 + s_m3 + 2 * s_im)

        # ========== Step 2: Token Selection ==========
        num_keep = max(1, math.ceil(N * self.sparse_ratio))

        if self.training and self.use_gumbel:
            # ========== 训练时：Gumbel-Softmax 可微采样 ==========
            # 1. 添加 Gumbel 噪声
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(score) + 1e-9) + 1e-9)
            logits = (score + gumbel_noise) / self.gumbel_tau

            # 2. 生成软权重（所有 tokens 的概率分布）
            soft_weights = F.softmax(logits, dim=1)  # (B, N)

            # 3. Top-K 用于生成硬选择（前向传播）
            _, top_k_indices = torch.topk(score, num_keep, dim=1)
            hard_mask = torch.zeros_like(score).scatter(1, top_k_indices, 1.0)

            # 4. Straight-Through Estimator
            # 前向：使用 hard_mask；反向：使用 soft_weights
            selection_weights = hard_mask + (soft_weights - soft_weights.detach())

            # 5. 使用软权重进行加权选择（可微）
            # 方式 1：加权所有 tokens，然后提取非零部分
            weighted_tokens = tokens * selection_weights.unsqueeze(-1)  # (B, N, C)

            # 提取选中的 tokens（通过 gather）
            select_tokens = torch.gather(
                weighted_tokens, dim=1,
                index=top_k_indices.unsqueeze(-1).expand(-1, -1, C)
            )  # (B, K, C)

            # 6. Extra token：未选中的 tokens 的加权平均
            non_selection_weights = 1 - selection_weights
            non_selection_weights = non_selection_weights / (non_selection_weights.sum(dim=1, keepdim=True) + 1e-8)
            extra_token = torch.sum(
                tokens * non_selection_weights.unsqueeze(-1), dim=1, keepdim=True
            )  # (B, 1, C)

            score_mask = selection_weights

        else:
            # ========== 推理时：Top-K 硬选择（快速） ==========
            score_sorted, score_indices = torch.sort(score, dim=1, descending=True)
            keep_policy = score_indices[:, :num_keep]

            # 硬选择
            select_tokens = torch.gather(
                tokens, dim=1,
                index=keep_policy.unsqueeze(-1).expand(-1, -1, C)
            )

            # Extra token
            non_keep_policy = score_indices[:, num_keep:]
            non_tokens = torch.gather(
                tokens, dim=1,
                index=non_keep_policy.unsqueeze(-1).expand(-1, -1, C)
            )
            non_keep_score = score_sorted[:, num_keep:]
            non_keep_score = F.softmax(non_keep_score, dim=1).unsqueeze(-1)
            extra_token = torch.sum(non_tokens * non_keep_score, dim=1, keepdim=True)

            score_mask = torch.zeros_like(score).scatter(1, keep_policy, 1.0)

        return select_tokens, extra_token, score_mask


class MultiModalSDTPS(nn.Module):
    """
    多模态 SDTPS 模块 - 修复版

    修复内容：
    1. Attention 计算使用 with torch.no_grad()
    2. 支持 Gumbel-Softmax 可微采样
    """

    def __init__(
        self,
        embed_dim: int = 512,
        sparse_ratio: float = 0.6,
        use_gumbel: bool = False,
        gumbel_tau: float = 1.0,
        beta: float = 0.25,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.sparse_ratio = sparse_ratio
        self.beta = beta

        self.rgb_sparse = TokenSparse(
            embed_dim=embed_dim,
            sparse_ratio=sparse_ratio,
            use_gumbel=use_gumbel,
            gumbel_tau=gumbel_tau,
        )

        self.nir_sparse = TokenSparse(
            embed_dim=embed_dim,
            sparse_ratio=sparse_ratio,
            use_gumbel=use_gumbel,
            gumbel_tau=gumbel_tau,
        )

        self.tir_sparse = TokenSparse(
            embed_dim=embed_dim,
            sparse_ratio=sparse_ratio,
            use_gumbel=use_gumbel,
            gumbel_tau=gumbel_tau,
        )

    def _compute_self_attention(
        self,
        patches: torch.Tensor,
        global_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算自注意力（修复版：添加 no_grad）

        Args:
            patches: (B, N, C)
            global_feat: (B, C) or (B, 1, C)

        Returns:
            (B, N) - attention scores
        """
        if global_feat.dim() == 2:
            global_feat = global_feat.unsqueeze(1)

        # 修复：添加 with torch.no_grad()
        with torch.no_grad():
            patches_norm = F.normalize(patches, dim=-1)
            global_norm = F.normalize(global_feat, dim=-1)
            self_attn = (patches_norm * global_norm).sum(dim=-1)

        return self_attn

    def _compute_cross_attention(
        self,
        patches: torch.Tensor,
        cross_global: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算交叉注意力（修复版：添加 no_grad）

        Args:
            patches: (B, N, C)
            cross_global: (B, C) or (B, 1, C)

        Returns:
            (B, N) - attention scores
        """
        if cross_global.dim() == 2:
            cross_global = cross_global.unsqueeze(1)

        # 修复：添加 with torch.no_grad()
        with torch.no_grad():
            patches_norm = F.normalize(patches, dim=-1)
            cross_norm = F.normalize(cross_global, dim=-1)
            cross_attn = (patches_norm * cross_norm).sum(dim=-1)

        return cross_attn

    def forward(
        self,
        RGB_cash: torch.Tensor,
        NI_cash: torch.Tensor,
        TI_cash: torch.Tensor,
        RGB_global: torch.Tensor,
        NI_global: torch.Tensor,
        TI_global: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Multi-modal token selection

        Args:
            RGB_cash, NI_cash, TI_cash: (B, N, C)
            RGB_global, NI_global, TI_global: (B, C)

        Returns:
            RGB_enhanced, NI_enhanced, TI_enhanced: (B, K+1, C)
            rgb_mask, nir_mask, tir_mask: (B, N)
        """

        # ==================== RGB ====================
        rgb_self_attn = self._compute_self_attention(RGB_cash, RGB_global)
        rgb_nir_cross = self._compute_cross_attention(RGB_cash, NI_global)
        rgb_tir_cross = self._compute_cross_attention(RGB_cash, TI_global)

        rgb_select, rgb_extra, rgb_mask = self.rgb_sparse(
            tokens=RGB_cash,
            self_attention=rgb_self_attn,
            cross_attention_m2=rgb_nir_cross,
            cross_attention_m3=rgb_tir_cross,
            beta=self.beta,
        )

        RGB_enhanced = torch.cat([rgb_select, rgb_extra], dim=1)

        # ==================== NIR ====================
        nir_self_attn = self._compute_self_attention(NI_cash, NI_global)
        nir_rgb_cross = self._compute_cross_attention(NI_cash, RGB_global)
        nir_tir_cross = self._compute_cross_attention(NI_cash, TI_global)

        nir_select, nir_extra, nir_mask = self.nir_sparse(
            tokens=NI_cash,
            self_attention=nir_self_attn,
            cross_attention_m2=nir_rgb_cross,
            cross_attention_m3=nir_tir_cross,
            beta=self.beta,
        )

        NI_enhanced = torch.cat([nir_select, nir_extra], dim=1)

        # ==================== TIR ====================
        tir_self_attn = self._compute_self_attention(TI_cash, TI_global)
        tir_rgb_cross = self._compute_cross_attention(TI_cash, RGB_global)
        tir_nir_cross = self._compute_cross_attention(TI_cash, NI_global)

        tir_select, tir_extra, tir_mask = self.tir_sparse(
            tokens=TI_cash,
            self_attention=tir_self_attn,
            cross_attention_m2=tir_rgb_cross,
            cross_attention_m3=tir_nir_cross,
            beta=self.beta,
        )

        TI_enhanced = torch.cat([tir_select, tir_extra], dim=1)

        return RGB_enhanced, NI_enhanced, TI_enhanced, rgb_mask, nir_mask, tir_mask
