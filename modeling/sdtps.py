"""
SDTPS: Sparse and Dense Token-Aware Patch Selection for Multi-Modal ReID
基于 SEPS 论文的 TokenSparse 模块，适配多模态行人/车辆重识别

改动说明：
- 原始：使用 图像自注意力 + 稀疏文本 + 稠密文本 生成 score
- 新版：使用 图像自注意力 + 其他两个模态的全局特征 生成 score
- 应用场景：RGB/NIR/TIR 三模态重识别
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TokenSparse(nn.Module):
    """
    Token 稀疏选择模块 - 简化版（保持空间结构）

    功能：从 N 个 patch 中选择 K 个最显著的 patch，但不提取，而是将未选中的置零
          K = ceil(N × sparse_ratio)

    改动说明：
        - 移除 MLP predictor，直接用三个注意力得分的平均
        - 不做 token 提取，用 mask 将未选中的 token 置零
        - 保持输出形状 (B, N, C) 不变，维持空间结构

    综合得分公式（简化）：
        score = (s_im + s_m2 + s_m3) / 3

    Args:
        sparse_ratio: 保留比例 (如 0.6 表示保留 60%)
        use_gumbel: 是否使用 Gumbel-Softmax 可微采样
        gumbel_tau: Gumbel 温度参数
    """

    def __init__(
        self,
        sparse_ratio: float = 0.6,
        use_gumbel: bool = False,
        gumbel_tau: float = 1.0,
    ):
        super().__init__()

        self.sparse_ratio = sparse_ratio
        self.use_gumbel = use_gumbel
        self.gumbel_tau = gumbel_tau

    def forward(
        self,
        tokens: torch.Tensor,                           # (B, N, C) - patch 特征
        self_attention: torch.Tensor,                   # (B, N) - 自注意力 score
        cross_attention_m2: torch.Tensor,               # (B, N) - 模态2交叉注意力
        cross_attention_m3: torch.Tensor,               # (B, N) - 模态3交叉注意力
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        执行语义评分和 soft masking

        Args:
            tokens: (B, N, C) - patch 特征
            self_attention: (B, N) - 自注意力 score (s^im)
            cross_attention_m2: (B, N) - 模态2交叉注意力 (s^{m2})
            cross_attention_m3: (B, N) - 模态3交叉注意力 (s^{m3})

        Returns:
            masked_tokens: (B, N, C) - 置零后的 tokens（保持空间结构）
            score_mask: (B, N) - 决策矩阵 D，1=选中，0=丢弃
        """
        B, N, C = tokens.size()

        # ========== Step 1: 计算综合得分 ==========
        # 归一化各注意力得分
        def normalize_score(s: torch.Tensor) -> torch.Tensor:
            """Min-Max 归一化到 [0,1]"""
            s_min = s.min(dim=-1, keepdim=True)[0]
            s_max = s.max(dim=-1, keepdim=True)[0]
            return (s - s_min) / (s_max - s_min + 1e-8)

        s_im = normalize_score(self_attention)         # (B, N) - 自注意力
        s_m2 = normalize_score(cross_attention_m2)     # (B, N) - 模态2
        s_m3 = normalize_score(cross_attention_m3)     # (B, N) - 模态3

        # 简化得分：三个注意力的平均
        score = (s_im + s_m2 + s_m3) / 3  # (B, N)

        # ========== Step 2: Top-K 选择 ==========
        num_keep = max(1, math.ceil(N * self.sparse_ratio))  # K = ceil(N×ρ)

        # 降序排序
        score_sorted, score_indices = torch.sort(score, dim=1, descending=True)
        keep_policy = score_indices[:, :num_keep]  # (B, K) - 保留的索引

        # ========== Step 3: 生成决策矩阵 D ==========
        if self.use_gumbel:
            # Gumbel-Softmax + Straight-Through Estimator
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(score) + 1e-9) + 1e-9)
            soft_mask = F.softmax((score + gumbel_noise) / self.gumbel_tau, dim=1)
            hard_mask = torch.zeros_like(score).scatter(1, keep_policy, 1.0)
            score_mask = hard_mask + (soft_mask - soft_mask.detach())  # STE
        else:
            # 标准 Top-K (不可微)
            score_mask = torch.zeros_like(score).scatter(1, keep_policy, 1.0)

        # ========== Step 4: 应用 mask（置零未选中的 tokens） ==========
        # score_mask: (B, N) → (B, N, 1) 用于广播
        masked_tokens = tokens * score_mask.unsqueeze(-1)  # (B, N, C) * (B, N, 1) = (B, N, C)

        return masked_tokens, score_mask


class MultiModalSDTPS(nn.Module):
    """
    多模态 SDTPS 模块 - 简化版（保持空间结构）

    功能：
        1. 对每个模态的 patch 特征进行稀疏选择（soft masking）
        2. 每个模态使用其他两个模态的全局特征作为引导
        3. 输出 masked 的多模态特征，保持原始 token 数量

    改动说明：
        - 移除 TokenAggregation 层
        - 不提取 tokens，直接返回 masked tokens (B, N, C)
        - 简化得分计算：三个注意力的平均

    流程（以 RGB 为例）：
        RGB_patches + RGB_global → 自注意力
        NIR_global × RGB_patches → 交叉注意力
        TIR_global × RGB_patches → 交叉注意力
        ↓
        综合 score → Top-K mask → 置零未选中的 tokens

    Args:
        embed_dim: 特征维度（未使用，但保留用于兼容）
        sparse_ratio: token 保留比例
        use_gumbel: 是否使用 Gumbel-Softmax
        gumbel_tau: Gumbel 温度
    """

    def __init__(
        self,
        embed_dim: int = 512,
        sparse_ratio: float = 0.6,
        use_gumbel: bool = False,
        gumbel_tau: float = 1.0,
        beta: float = 0.25,  # 保留参数用于兼容，但不使用
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.sparse_ratio = sparse_ratio

        # 为每个模态创建独立的 TokenSparse 模块（简化版）
        self.rgb_sparse = TokenSparse(
            sparse_ratio=sparse_ratio,
            use_gumbel=use_gumbel,
            gumbel_tau=gumbel_tau,
        )

        self.nir_sparse = TokenSparse(
            sparse_ratio=sparse_ratio,
            use_gumbel=use_gumbel,
            gumbel_tau=gumbel_tau,
        )

        self.tir_sparse = TokenSparse(
            sparse_ratio=sparse_ratio,
            use_gumbel=use_gumbel,
            gumbel_tau=gumbel_tau,
        )

    def _compute_self_attention(
        self,
        patches: torch.Tensor,       # (B, N, C)
        global_feat: torch.Tensor,   # (B, C) or (B, 1, C)
    ) -> torch.Tensor:
        """
        计算自注意力 score: 每个 patch 与模态全局特征的相似度

        Args:
            patches: (B, N, C) - patch 特征
            global_feat: (B, C) or (B, 1, C) - 全局特征

        Returns:
            (B, N) - 自注意力 score
        """
        if global_feat.dim() == 2:
            global_feat = global_feat.unsqueeze(1)  # (B, C) → (B, 1, C)

        # L2 归一化
        patches_norm = F.normalize(patches, dim=-1)           # (B, N, C)
        global_norm = F.normalize(global_feat, dim=-1)        # (B, 1, C)

        # 计算相似度
        self_attn = (patches_norm * global_norm).sum(dim=-1)  # (B, N)

        return self_attn

    def _compute_cross_attention(
        self,
        patches: torch.Tensor,       # (B, N, C) - 目标模态的 patches
        cross_global: torch.Tensor,  # (B, C) or (B, 1, C) - 其他模态的全局特征
    ) -> torch.Tensor:
        """
        计算交叉注意力 score: 目标模态的 patches 与其他模态全局特征的相似度

        Args:
            patches: (B, N, C) - 目标模态的 patch 特征
            cross_global: (B, C) or (B, 1, C) - 其他模态的全局特征

        Returns:
            (B, N) - 交叉注意力 score
        """
        if cross_global.dim() == 2:
            cross_global = cross_global.unsqueeze(1)  # (B, C) → (B, 1, C)

        # L2 归一化
        patches_norm = F.normalize(patches, dim=-1)           # (B, N, C)
        cross_norm = F.normalize(cross_global, dim=-1)        # (B, 1, C)

        # 计算相似度
        cross_attn = (patches_norm * cross_norm).sum(dim=-1)  # (B, N)

        return cross_attn

    def forward(
        self,
        RGB_cash: torch.Tensor,      # (B, N, C) - RGB patch 特征
        NI_cash: torch.Tensor,       # (B, N, C) - NIR patch 特征
        TI_cash: torch.Tensor,       # (B, N, C) - TIR patch 特征
        RGB_global: torch.Tensor,    # (B, C) - RGB 全局特征
        NI_global: torch.Tensor,     # (B, C) - NIR 全局特征
        TI_global: torch.Tensor,     # (B, C) - TIR 全局特征
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        多模态 token selection（保持空间结构）

        Args:
            RGB_cash: (B, N, C) - RGB patch 特征
            NI_cash: (B, N, C) - NIR patch 特征
            TI_cash: (B, N, C) - TIR patch 特征
            RGB_global: (B, C) - RGB 全局特征
            NI_global: (B, C) - NIR 全局特征
            TI_global: (B, C) - TIR 全局特征

        Returns:
            RGB_enhanced: (B, N, C) - masked RGB 特征
            NI_enhanced: (B, N, C) - masked NIR 特征
            TI_enhanced: (B, N, C) - masked TIR 特征
            RGB_mask: (B, N) - RGB 决策矩阵
            NI_mask: (B, N) - NIR 决策矩阵
            TI_mask: (B, N) - TIR 决策矩阵
        """

        # ==================== RGB 模态 ====================
        # 1. RGB 自注意力
        rgb_self_attn = self._compute_self_attention(RGB_cash, RGB_global)  # (B, N)

        # 2. NIR → RGB 交叉注意力
        rgb_nir_cross = self._compute_cross_attention(RGB_cash, NI_global)  # (B, N)

        # 3. TIR → RGB 交叉注意力
        rgb_tir_cross = self._compute_cross_attention(RGB_cash, TI_global)  # (B, N)

        # 4. RGB Token Masking
        RGB_enhanced, rgb_mask = self.rgb_sparse(
            tokens=RGB_cash,
            self_attention=rgb_self_attn,
            cross_attention_m2=rgb_nir_cross,
            cross_attention_m3=rgb_tir_cross,
        )  # (B, N, C), (B, N)

        # ==================== NIR 模态 ====================
        nir_self_attn = self._compute_self_attention(NI_cash, NI_global)
        nir_rgb_cross = self._compute_cross_attention(NI_cash, RGB_global)
        nir_tir_cross = self._compute_cross_attention(NI_cash, TI_global)

        NI_enhanced, nir_mask = self.nir_sparse(
            tokens=NI_cash,
            self_attention=nir_self_attn,
            cross_attention_m2=nir_rgb_cross,
            cross_attention_m3=nir_tir_cross,
        )  # (B, N, C), (B, N)

        # ==================== TIR 模态 ====================
        tir_self_attn = self._compute_self_attention(TI_cash, TI_global)
        tir_rgb_cross = self._compute_cross_attention(TI_cash, RGB_global)
        tir_nir_cross = self._compute_cross_attention(TI_cash, NI_global)

        TI_enhanced, tir_mask = self.tir_sparse(
            tokens=TI_cash,
            self_attention=tir_self_attn,
            cross_attention_m2=tir_rgb_cross,
            cross_attention_m3=tir_nir_cross,
        )  # (B, N, C), (B, N)

        return RGB_enhanced, NI_enhanced, TI_enhanced, rgb_mask, nir_mask, tir_mask
