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
    Token 稀疏选择模块 - 多模态版本

    功能：从 N 个 patch 中选择 K 个最显著的 patch
          K = ceil(N × sparse_ratio)

    改动：
        原始输入：img_self_attn + sparse_text_attn + dense_text_attn
        新版输入：img_self_attn + modal2_cross_attn + modal3_cross_attn

    示例（以 RGB 为例）：
        - img_self_attn: RGB 自注意力 score
        - modal2_cross_attn: NIR_global → RGB_patches 的 cross-attention
        - modal3_cross_attn: TIR_global → RGB_patches 的 cross-attention
        - predictive_score: MLP 预测的 RGB 自身重要性

    综合得分公式：
        score = (1-2β)·s^p + β·(s^{m2} + s^{m3} + 2·s^{im})

    Args:
        embed_dim: 特征维度 (如 512)
        sparse_ratio: 保留比例 (如 0.6 表示保留 60%)
        use_gumbel: 是否使用 Gumbel-Softmax 可微采样
        gumbel_tau: Gumbel 温度参数
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

        # MLP Score Predictor: 学习每个 patch 自身的重要性
        self.score_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),  # (*, C) → (*, C//4)
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1),           # (*, C//4) → (*, 1)
            nn.Sigmoid(),                            # 输出 ∈ [0,1]
        )

    def forward(
        self,
        tokens: torch.Tensor,                           # (B, N, C) - patch 特征
        self_attention: torch.Tensor,                   # (B, N) - 自注意力 score
        cross_attention_m2: torch.Tensor,               # (B, N) - 模态2交叉注意力
        cross_attention_m3: torch.Tensor,               # (B, N) - 模态3交叉注意力
        beta: float = 0.25,                             # β 权重参数
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        执行语义评分和 patch 选择

        Args:
            tokens: (B, N, C) - patch 特征
            self_attention: (B, N) - 自注意力 score (s^im)
            cross_attention_m2: (B, N) - 模态2交叉注意力 (s^{m2})
            cross_attention_m3: (B, N) - 模态3交叉注意力 (s^{m3})
            beta: β 权重参数，默认 0.25

        Returns:
            select_tokens: (B, K, C) - 选中的显著 patch
            extra_token: (B, 1, C) - 融合的冗余 patch
            score_mask: (B, N) - 决策矩阵 D，1=选中，0=丢弃
        """
        B, N, C = tokens.size()

        # ========== Step 1: 计算综合得分 ==========
        # 1.1 MLP 预测自身重要性: s^p
        s_pred = self.score_predictor(tokens).squeeze(-1)  # (B, N, C) → (B, N)

        # 1.2 归一化各注意力得分
        def normalize_score(s: torch.Tensor) -> torch.Tensor:
            """Min-Max 归一化到 [0,1]"""
            s_min = s.min(dim=-1, keepdim=True)[0]
            s_max = s.max(dim=-1, keepdim=True)[0]
            return (s - s_min) / (s_max - s_min + 1e-8)

        s_im = normalize_score(self_attention)         # (B, N) - 自注意力
        s_m2 = normalize_score(cross_attention_m2)     # (B, N) - 模态2
        s_m3 = normalize_score(cross_attention_m3)     # (B, N) - 模态3

        # 1.3 综合得分
        # score = (1-2β)·s_pred + β·(s_m2 + s_m3 + 2·s_im)
        score = (1 - 2 * beta) * s_pred + beta * (s_m2 + s_m3 + 2 * s_im)  # (B, N)

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

        # ========== Step 4: 提取选中的 patch ==========
        select_tokens = torch.gather(
            tokens, dim=1,
            index=keep_policy.unsqueeze(-1).expand(-1, -1, C)
        )  # (B, N, C) → (B, K, C)

        # ========== Step 5: 融合被丢弃的 patch ==========
        non_keep_policy = score_indices[:, num_keep:]  # (B, N-K)
        non_tokens = torch.gather(
            tokens, dim=1,
            index=non_keep_policy.unsqueeze(-1).expand(-1, -1, C)
        )  # (B, N, C) → (B, N-K, C)

        non_keep_score = score_sorted[:, num_keep:]  # (B, N-K)
        non_keep_score = F.softmax(non_keep_score, dim=1).unsqueeze(-1)  # (B, N-K, 1)
        extra_token = torch.sum(non_tokens * non_keep_score, dim=1, keepdim=True)  # (B, 1, C)

        return select_tokens, extra_token, score_mask


class MultiModalSDTPS(nn.Module):
    """
    多模态 SDTPS 模块 - 为 RGB/NIR/TIR 三个模态分别进行 token selection

    功能：
        1. 对每个模态的 patch 特征进行选择和增强
        2. 每个模态使用其他两个模态的全局特征作为引导
        3. 输出增强后的多模态特征

    流程（以 RGB 为例）：
        RGB_patches + RGB_global → 自注意力
        NIR_global × RGB_patches → 交叉注意力
        TIR_global × RGB_patches → 交叉注意力
        ↓
        综合 score → Top-K 选择 → 增强的 RGB 特征

    Args:
        embed_dim: 特征维度
        sparse_ratio: token 保留比例
        use_gumbel: 是否使用 Gumbel-Softmax
        gumbel_tau: Gumbel 温度
        beta: score 组合的权重参数
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

        # 为每个模态创建独立的 TokenSparse 模块
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
        多模态 token selection

        Args:
            RGB_cash: (B, N, C) - RGB patch 特征
            NI_cash: (B, N, C) - NIR patch 特征
            TI_cash: (B, N, C) - TIR patch 特征
            RGB_global: (B, C) - RGB 全局特征
            NI_global: (B, C) - NIR 全局特征
            TI_global: (B, C) - TIR 全局特征

        Returns:
            RGB_enhanced: (B, K+1, C) - 增强的 RGB 特征 (K 个选中 + 1 个 extra)
            NI_enhanced: (B, K+1, C) - 增强的 NIR 特征
            TI_enhanced: (B, K+1, C) - 增强的 TIR 特征
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

        # 4. RGB Token Selection
        rgb_select, rgb_extra, rgb_mask = self.rgb_sparse(
            tokens=RGB_cash,
            self_attention=rgb_self_attn,
            cross_attention_m2=rgb_nir_cross,
            cross_attention_m3=rgb_tir_cross,
            beta=self.beta,
        )  # (B, K, C), (B, 1, C), (B, N)

        # 5. 拼接选中的和 extra token
        RGB_enhanced = torch.cat([rgb_select, rgb_extra], dim=1)  # (B, K+1, C)

        # ==================== NIR 模态 ====================
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

        NI_enhanced = torch.cat([nir_select, nir_extra], dim=1)  # (B, K+1, C)

        # ==================== TIR 模态 ====================
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

        TI_enhanced = torch.cat([tir_select, tir_extra], dim=1)  # (B, K+1, C)

        return RGB_enhanced, NI_enhanced, TI_enhanced, rgb_mask, nir_mask, tir_mask
