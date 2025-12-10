"""
SDTPS: Sparse and Dense Token-Aware Patch Selection for Multi-Modal ReID
基于 SEPS 论文的 TokenSparse 模块，适配多模态行人/车辆重识别

改动说明：
- 原始：使用 图像自注意力 + 稀疏文本 + 稠密文本 生成 score
- 新版：使用 图像自注意力 + 其他两个模态的全局特征 生成 score
- 应用场景：RGB/NIR/TIR 三模态重识别
- 引入 Cross-Attention 机制，用逐 head 余弦门控
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CrossModalAttention(nn.Module):
    """
    Cross-Attention 模块 - 单头版本（尺度匹配优化）

    核心设计：
    1. Q/K 反向：global 作为 Query，patches 作为 Key
    2. Cosine 经温度缩放后作为 bias 加到 attention logits
    3. 单头：无需分割和平均，最简单
    4. 尺度匹配：cosine_tau 确保 cosine 与 attention 贡献平衡

    计算流程：
    1. Q = W_q @ global: (B, 1, C)
    2. K = W_k @ patches: (B, N, C)
    3. logits = Q @ K^T / sqrt(C) + cosine / cosine_tau
    4. score = softmax(logits): (B, N)

    Args:
        embed_dim: 特征维度
        cosine_tau: cosine 温度（控制 cosine 贡献强度，默认 0.3）
        attn_drop: dropout 比率
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 1,  # 保留兼容性，但固定为1
        cosine_tau: float = 0.3,  # cosine 温度（越小贡献越大）
        attn_drop: float = 0.0,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.scale = embed_dim ** -0.5  # 单头：直接用 embed_dim
        self.cosine_tau = cosine_tau  # cosine 温度

        # Q, K 投影
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.k_proj.bias)

    def forward(
        self,
        patches: torch.Tensor,       # (B, N, C)
        global_feat: torch.Tensor,   # (B, C) or (B, 1, C)
        cosine_sim: torch.Tensor,    # (B, N)
    ) -> torch.Tensor:
        """
        单头 Cross-Attention（尺度匹配版）

        Args:
            patches: (B, N, C) - patch tokens
            global_feat: (B, C) or (B, 1, C) - global feature
            cosine_sim: (B, N) - 预计算的余弦相似度（范围 [-1, 1]）

        Returns:
            score: (B, N) - patch 重要性得分（概率分布）
        """
        B, N, C = patches.shape

        if global_feat.dim() == 2:
            global_feat = global_feat.unsqueeze(1)  # (B, C) -> (B, 1, C)

        # ========== 单头 Q/K 投影（无分割）==========
        q = self.q_proj(global_feat)  # (B, 1, C)
        k = self.k_proj(patches)      # (B, N, C)

        # ========== Attention + Cosine Bias（尺度匹配）==========
        # Q @ K^T / sqrt(C)，典型范围约 [-3, 3]
        attn_logits = (q @ k.transpose(-2, -1)) * self.scale  # (B, 1, N)
        attn_logits = attn_logits.squeeze(1)  # (B, N)

        # cosine / tau 进行温度缩放，使 cosine（范围 [-1,1]）贡献与 attn 匹配
        # tau=0.3 时，cosine 贡献范围约 [-3.3, 3.3]，与 attn_logits 尺度相当
        cosine_scaled = cosine_sim / self.cosine_tau  # (B, N)
        attn_logits = attn_logits + cosine_scaled  # (B, N)

        # Softmax（输出自动归一化）
        score = attn_logits.softmax(dim=-1)  # (B, N)
        score = self.attn_drop(score)

        return score


class TokenSparse(nn.Module):
    """
    Token 稀疏选择模块 - 改进版

    功能：从 N 个 patch 中选择 K 个最显著的 patch

    改进点：
    1. Z-score 归一化替代 Min-Max（数值更稳定）
    2. 分位数阈值替代均值中心化（稀疏性可控）
    3. Gumbel-Sigmoid 替代 Gumbel-Softmax（语义正确）
    4. 改进的 MLP 结构（LayerNorm + GELU + Dropout）

    综合得分公式：
        score = w_1*s_im + w_2*s_m2 + w_3*s_m3（样本自适应权重）

    Masking 策略：
        - Soft: soft_mask = sigmoid((score - quantile_threshold) / tau)
        - Hard: Top-K 硬选择 + Gumbel-Sigmoid STE

    Args:
        embed_dim: 特征维度
        sparse_ratio: 保留比例（如 0.6 表示保留 60%）
        use_gumbel: 是否使用 Gumbel-Sigmoid 可微采样
        gumbel_tau: Gumbel 温度参数
        use_adaptive_weights: 是否使用样本自适应模态权重
        use_soft_masking: 是否使用 soft masking
        soft_mask_tau: Soft masking 温度（0.1~0.5）
    """

    def __init__(
        self,
        embed_dim: int = 512,
        sparse_ratio: float = 0.6,
        use_gumbel: bool = False,
        gumbel_tau: float = 1.0,
        use_adaptive_weights: bool = True,
        use_soft_masking: bool = True,
        soft_mask_tau: float = 0.3,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.sparse_ratio = sparse_ratio
        self.use_gumbel = use_gumbel
        self.gumbel_tau = gumbel_tau
        self.use_adaptive_weights = use_adaptive_weights
        self.use_soft_masking = use_soft_masking
        self.soft_mask_tau = soft_mask_tau

        # ========== 问题5修复：改进的样本自适应模态权重 MLP ==========
        if use_adaptive_weights:
            # 改进版 MLP：更深、有正则化、使用 GELU 保留负值信息
            self.modal_weight_mlp = nn.Sequential(
                nn.Linear(embed_dim * 3, 256, bias=True),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(256, 64, bias=True),
                nn.GELU(),
                nn.Linear(64, 3, bias=True)
            )
            # 初始化：最后一层接近0，使初始权重接近均等
            nn.init.xavier_uniform_(self.modal_weight_mlp[0].weight, gain=0.5)
            nn.init.zeros_(self.modal_weight_mlp[0].bias)
            nn.init.xavier_uniform_(self.modal_weight_mlp[4].weight, gain=0.5)
            nn.init.zeros_(self.modal_weight_mlp[4].bias)
            nn.init.zeros_(self.modal_weight_mlp[6].weight)
            nn.init.zeros_(self.modal_weight_mlp[6].bias)
        else:
            self.modal_weight_mlp = None

    def _normalize_score(self, s: torch.Tensor) -> torch.Tensor:
        """
        问题2修复：Z-score 归一化 + Sigmoid（更鲁棒）

        相比 Min-Max 的优势：
        1. 对离群值不敏感
        2. 当 s_max ≈ s_min 时仍然稳定
        3. 输出分布更均匀

        Args:
            s: (B, N) - 输入分数

        Returns:
            (B, N) - 归一化后的分数，范围 [0, 1]
        """
        s_mean = s.mean(dim=-1, keepdim=True)
        s_std = s.std(dim=-1, keepdim=True) + 1e-5  # 更大的 epsilon
        z_score = (s - s_mean) / s_std
        return torch.sigmoid(z_score)  # 映射到 [0, 1]

    def forward(
        self,
        tokens: torch.Tensor,                           # (B, N, C) - patch 特征
        self_attention: torch.Tensor,                   # (B, N) - 自注意力 score
        cross_attention_m2: torch.Tensor,               # (B, N) - 模态2交叉注意力
        cross_attention_m3: torch.Tensor,               # (B, N) - 模态3交叉注意力
        global_feats: torch.Tensor = None,              # (B, 3*C) - 3个模态的global特征拼接
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        执行语义评分和 masking

        Args:
            tokens: (B, N, C) - patch 特征
            self_attention: (B, N) - 自注意力 score
            cross_attention_m2: (B, N) - 模态2交叉注意力
            cross_attention_m3: (B, N) - 模态3交叉注意力
            global_feats: (B, 3*C) - 3个模态的 global 特征拼接

        Returns:
            masked_tokens: (B, N, C) - 加权后的 tokens
            score_mask: (B, N) - 决策矩阵
        """
        B, N, C = tokens.size()

        # ========== Step 1: 归一化各注意力得分（问题2修复）==========
        s_im = self._normalize_score(self_attention)      # (B, N)
        s_m2 = self._normalize_score(cross_attention_m2)  # (B, N)
        s_m3 = self._normalize_score(cross_attention_m3)  # (B, N)

        # ========== Step 2: 计算综合得分 ==========
        if self.use_adaptive_weights and global_feats is not None:
            # 样本自适应权重（问题5已在 __init__ 中修复 MLP 结构）
            modal_logits = self.modal_weight_mlp(global_feats)  # (B, 3)
            weights = F.softmax(modal_logits, dim=-1)  # (B, 3)
            score = weights[:, 0:1] * s_im + weights[:, 1:2] * s_m2 + weights[:, 2:3] * s_m3
        else:
            # 简单平均
            score = (s_im + s_m2 + s_m3) / 3  # (B, N)

        # ========== Step 3: Masking 策略 ==========
        if self.use_soft_masking:
            # ========== 问题4修复：使用分位数阈值替代均值中心化 ==========
            # 计算 (1 - sparse_ratio) 分位数作为阈值
            # 例如 sparse_ratio=0.6 → 取 0.4 分位数 → 约 60% 的 token 高于阈值
            threshold = torch.quantile(score, 1 - self.sparse_ratio, dim=1, keepdim=True)
            centered_score = score - threshold  # (B, N)

            # Sigmoid 映射，tau 控制软硬程度
            soft_mask = torch.sigmoid(centered_score / self.soft_mask_tau)  # (B, N)
            masked_tokens = tokens * soft_mask.unsqueeze(-1)  # (B, N, C)

            return masked_tokens, soft_mask

        else:
            # Hard Top-K 选择
            num_keep = max(1, math.ceil(N * self.sparse_ratio))

            # 降序排序
            score_sorted, score_indices = torch.sort(score, dim=1, descending=True)
            keep_policy = score_indices[:, :num_keep]  # (B, K)

            if self.use_gumbel:
                # ========== 问题3修复：Gumbel-Sigmoid（逐 token 独立采样）==========
                # Gumbel 噪声
                gumbel_noise = -torch.log(-torch.log(torch.rand_like(score) + 1e-9) + 1e-9)
                # Gumbel-Sigmoid（每个 token 独立决策）
                soft_mask = torch.sigmoid((score + gumbel_noise - 0.5) / self.gumbel_tau)
                # Hard mask（Top-K）
                hard_mask = torch.zeros_like(score).scatter(1, keep_policy, 1.0)
                # Straight-Through Estimator：前向用 hard，反向用 soft
                score_mask = hard_mask + (soft_mask - soft_mask.detach())
            else:
                # 标准 Top-K（不可微）
                score_mask = torch.zeros_like(score).scatter(1, keep_policy, 1.0)

            masked_tokens = tokens * score_mask.unsqueeze(-1)  # (B, N, C)

            return masked_tokens, score_mask


class MultiModalSDTPS(nn.Module):
    """
    多模态 SDTPS 模块 - Sparse and Dense Token-Aware Patch Selection

    功能：
    1. 对每个模态的 patch 特征进行稀疏选择
    2. 每个模态使用其他两个模态的全局特征作为引导
    3. 结合余弦相似度和 Cross-Attention
    4. 输出 masked 的多模态特征，保持原始 token 数量

    流程（以 RGB 为例）：
    1. 计算余弦相似度：cos_sim = cosine(RGB_patches, global)
    2. Cross-Attention：score = softmax(Q@K/√d + cos/tau)
    3. Token Masking：综合三个 score，应用 soft/hard mask

    Args:
        embed_dim: 特征维度
        sparse_ratio: token 保留比例
        use_gumbel: 是否使用 Gumbel-Sigmoid
        gumbel_tau: Gumbel 温度
        use_cross_attn: 是否使用 Cross-Attention（默认 True）
        share_cross_attn_weights: 是否共享 attention 权重
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_patches: int = 128,
        sparse_ratio: float = 0.6,
        aggr_ratio: float = 0.5,
        use_gumbel: bool = False,
        gumbel_tau: float = 1.0,
        beta: float = 0.25,
        cross_attn_type: str = 'attention',
        cross_attn_heads: int = 4,
        use_cross_attn: bool = None,
        share_cross_attn_weights: bool = False,  # 新增：是否共享 attention 权重
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.sparse_ratio = sparse_ratio

        # 处理兼容性
        if use_cross_attn is None:
            use_cross_attn = (cross_attn_type == 'attention')
        self.use_cross_attn = use_cross_attn
        self.share_cross_attn_weights = share_cross_attn_weights

        # 设置注意力头数
        num_heads = cross_attn_heads

        # TokenSparse 模块
        self.rgb_sparse = TokenSparse(embed_dim=embed_dim, sparse_ratio=sparse_ratio,
                                      use_gumbel=use_gumbel, gumbel_tau=gumbel_tau)
        self.nir_sparse = TokenSparse(embed_dim=embed_dim, sparse_ratio=sparse_ratio,
                                      use_gumbel=use_gumbel, gumbel_tau=gumbel_tau)
        self.tir_sparse = TokenSparse(embed_dim=embed_dim, sparse_ratio=sparse_ratio,
                                      use_gumbel=use_gumbel, gumbel_tau=gumbel_tau)

        # ========== Cross-Attention 权重共享选项 ==========
        if self.use_cross_attn:
            if share_cross_attn_weights:
                # 部分共享：每个模态内部共享，模态间独立
                # 参数量：3 个模块（RGB、NIR、TIR 各 1 个）
                print("使用部分共享 CrossModalAttention 权重（每个模态共享，参数量减少 67%）")

                # RGB 模态的共享 attention
                self.rgb_shared_attn = CrossModalAttention(embed_dim, num_heads)
                self.rgb_self_attn = self.rgb_shared_attn
                self.rgb_cross_nir = self.rgb_shared_attn
                self.rgb_cross_tir = self.rgb_shared_attn

                # NIR 模态的共享 attention
                self.nir_shared_attn = CrossModalAttention(embed_dim, num_heads)
                self.nir_self_attn = self.nir_shared_attn
                self.nir_cross_rgb = self.nir_shared_attn
                self.nir_cross_tir = self.nir_shared_attn

                # TIR 模态的共享 attention
                self.tir_shared_attn = CrossModalAttention(embed_dim, num_heads)
                self.tir_self_attn = self.tir_shared_attn
                self.tir_cross_rgb = self.tir_shared_attn
                self.tir_cross_nir = self.tir_shared_attn
            else:
                # 独立权重：每个模态每个交叉都有独立的 CrossModalAttention
                # 参数量：9 个独立模块
                # RGB 模态
                self.rgb_self_attn = CrossModalAttention(embed_dim, num_heads)
                self.rgb_cross_nir = CrossModalAttention(embed_dim, num_heads)
                self.rgb_cross_tir = CrossModalAttention(embed_dim, num_heads)

                # NIR 模态
                self.nir_self_attn = CrossModalAttention(embed_dim, num_heads)
                self.nir_cross_rgb = CrossModalAttention(embed_dim, num_heads)
                self.nir_cross_tir = CrossModalAttention(embed_dim, num_heads)

                # TIR 模态
                self.tir_self_attn = CrossModalAttention(embed_dim, num_heads)
                self.tir_cross_rgb = CrossModalAttention(embed_dim, num_heads)
                self.tir_cross_nir = CrossModalAttention(embed_dim, num_heads)

    def _compute_cosine_similarity(
        self,
        patches: torch.Tensor,       # (B, N, C)
        global_feat: torch.Tensor,   # (B, C) or (B, 1, C)
    ) -> torch.Tensor:
        """
        计算余弦相似度（优化版）

        Args:
            patches: (B, N, C) - patch 特征
            global_feat: (B, C) or (B, 1, C) - 全局特征

        Returns:
            (B, N) - 余弦相似度 score
        """
        if global_feat.dim() == 2:
            global_feat = global_feat.unsqueeze(1)  # (B, C) → (B, 1, C)

        # ========== 修复4：优化余弦计算，减少内存占用 ==========
        # 旧版（生成 (B,N,C) 中间张量）：
        # patches_norm = F.normalize(patches, dim=-1)           # (B, N, C)
        # global_norm = F.normalize(global_feat, dim=-1)        # (B, 1, C)
        # cosine_sim = (patches_norm * global_norm).sum(dim=-1)  # (B, N)

        # 新版（使用 einsum，避免中间张量）：
        patches_norm = F.normalize(patches, dim=-1)           # (B, N, C)
        global_norm = F.normalize(global_feat, dim=-1)        # (B, 1, C)
        cosine_sim = torch.einsum('bnc,boc->bn', patches_norm, global_norm)  # (B, N)
        # 或使用矩阵乘法（等价）：
        # cosine_sim = (patches_norm @ global_norm.transpose(-2, -1)).squeeze(-1)

        return cosine_sim

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

        # ==================== 准备样本自适应权重的输入 ====================
        # ========== 专家建议4：拼接3个模态的 global 特征 ==========
        global_feats = torch.cat([RGB_global, NI_global, TI_global], dim=-1)  # (B, 3*C)

        # ==================== RGB 模态 ====================
        # 回退：预计算原始空间余弦后再送入 Cross-Attention（训练更稳，语义一致）
        rgb_cos_self = self._compute_cosine_similarity(RGB_cash, RGB_global)  # (B, N)
        rgb_cos_nir = self._compute_cosine_similarity(RGB_cash, NI_global)    # (B, N)
        rgb_cos_tir = self._compute_cosine_similarity(RGB_cash, TI_global)    # (B, N)

        if self.use_cross_attn:
            rgb_self_attn = self.rgb_self_attn(RGB_cash, RGB_global, rgb_cos_self)
            rgb_nir_cross = self.rgb_cross_nir(RGB_cash, NI_global, rgb_cos_nir)
            rgb_tir_cross = self.rgb_cross_tir(RGB_cash, TI_global, rgb_cos_tir)
        else:
            # 仅使用余弦相似度
            rgb_self_attn = rgb_cos_self
            rgb_nir_cross = rgb_cos_nir
            rgb_tir_cross = rgb_cos_tir

        # 保留投影空间余弦的实现作为注释（未来可通过超参数切换）
        # if self.use_cross_attn:
        #     rgb_self_attn = self.rgb_self_attn(RGB_cash, RGB_global, cosine_sim=None)
        #     rgb_nir_cross = self.rgb_cross_nir(RGB_cash, NI_global, cosine_sim=None)
        #     rgb_tir_cross = self.rgb_cross_tir(RGB_cash, TI_global, cosine_sim=None)

        # Step 3: Token Masking（传递 global_feats）
        RGB_enhanced, rgb_mask = self.rgb_sparse(
            tokens=RGB_cash,
            self_attention=rgb_self_attn,
            cross_attention_m2=rgb_nir_cross,
            cross_attention_m3=rgb_tir_cross,
            global_feats=global_feats,  # 专家建议4
        )  # (B, N, C), (B, N)

        # ==================== NIR 模态 ====================
        nir_cos_self = self._compute_cosine_similarity(NI_cash, NI_global)
        nir_cos_rgb = self._compute_cosine_similarity(NI_cash, RGB_global)
        nir_cos_tir = self._compute_cosine_similarity(NI_cash, TI_global)

        if self.use_cross_attn:
            nir_self_attn = self.nir_self_attn(NI_cash, NI_global, nir_cos_self)
            nir_rgb_cross = self.nir_cross_rgb(NI_cash, RGB_global, nir_cos_rgb)
            nir_tir_cross = self.nir_cross_tir(NI_cash, TI_global, nir_cos_tir)
        else:
            nir_self_attn = nir_cos_self
            nir_rgb_cross = nir_cos_rgb
            nir_tir_cross = nir_cos_tir

        # 保留投影空间余弦的实现作为注释（未来可通过超参数切换）
        # if self.use_cross_attn:
        #     nir_self_attn = self.nir_self_attn(NI_cash, NI_global, cosine_sim=None)
        #     nir_rgb_cross = self.nir_cross_rgb(NI_cash, RGB_global, cosine_sim=None)
        #     nir_tir_cross = self.nir_cross_tir(NI_cash, TI_global, cosine_sim=None)

        NI_enhanced, nir_mask = self.nir_sparse(
            tokens=NI_cash,
            self_attention=nir_self_attn,
            cross_attention_m2=nir_rgb_cross,
            cross_attention_m3=nir_tir_cross,
            global_feats=global_feats,  # 专家建议4
        )  # (B, N, C), (B, N)

        # ==================== TIR 模态 ====================
        tir_cos_self = self._compute_cosine_similarity(TI_cash, TI_global)
        tir_cos_rgb = self._compute_cosine_similarity(TI_cash, RGB_global)
        tir_cos_nir = self._compute_cosine_similarity(TI_cash, NI_global)

        if self.use_cross_attn:
            tir_self_attn = self.tir_self_attn(TI_cash, TI_global, tir_cos_self)
            tir_rgb_cross = self.tir_cross_rgb(TI_cash, RGB_global, tir_cos_rgb)
            tir_nir_cross = self.tir_cross_nir(TI_cash, NI_global, tir_cos_nir)
        else:
            tir_self_attn = tir_cos_self
            tir_rgb_cross = tir_cos_rgb
            tir_nir_cross = tir_cos_nir

        # 保留投影空间余弦的实现作为注释（未来可通过超参数切换）
        # if self.use_cross_attn:
        #     tir_self_attn = self.tir_self_attn(TI_cash, TI_global, cosine_sim=None)
        #     tir_rgb_cross = self.tir_cross_rgb(TI_cash, RGB_global, cosine_sim=None)
        #     tir_nir_cross = self.tir_cross_nir(TI_cash, NI_global, cosine_sim=None)

        TI_enhanced, tir_mask = self.tir_sparse(
            tokens=TI_cash,
            self_attention=tir_self_attn,
            cross_attention_m2=tir_rgb_cross,
            cross_attention_m3=tir_nir_cross,
            global_feats=global_feats,  # 专家建议4
        )  # (B, N, C), (B, N)

        return RGB_enhanced, NI_enhanced, TI_enhanced, rgb_mask, nir_mask, tir_mask
