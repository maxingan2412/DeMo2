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
    Cross-Attention 模块（带逐 head 余弦门控）

    核心设计：
    1. Q/K 反向：global 作为 Query，patches 作为 Key
    2. 逐 head 余弦门控：每个 head 学习独立的 scale 和 bias
    3. 在多头平均前进行门控，保留多头差异性
    4. 在 N（patch）维度做 softmax

    计算流程：
    1. 余弦相似度：cos_sim = normalize(patches) @ normalize(global)^T  # (B, N)
    2. Q/K 投影 + Attention：
       Q = W_q @ global, K = W_k @ patches
       attn = softmax(Q @ K^T / sqrt(d), dim=N)  # (B, num_heads, N)
    3. 逐 head 余弦门控：
       gate_logits = cosine * scale[h] + bias[h]  # (B, num_heads, N)
       [可选] gate_logits = LayerNorm(gate_logits)  # 稳定尺度
       gate = sigmoid(gate_logits)  # (B, num_heads, N)
    4. 门控注意力：
       attn_gated = attn * gate  # 逐元素点乘
       [可选] attn_gated = renormalize(attn_gated)  # 保持概率性
    5. 多头平均：
       score = mean(attn_gated, dim=heads)  # (B, N)

    优势：
    - 参数极少：num_heads × 2 个参数（如 4 heads = 8 params）
    - 逐 head 自适应：每个 head 学习不同的门控策略
    - 物理意义清晰：scale 控制温度，bias 控制偏置
    - 简洁高效：基础版本不需要额外的 LayerNorm

    改进选项：
    - use_gate_norm: 在 sigmoid 前使用 LayerNorm 稳定逐 head 尺度
    - renormalize_attn: 在门控后重新归一化，保持权重和为 1

    初始化策略（避免过稀疏）：
    - gate_scale = 0.5 (较小，使曲线平缓)
    - gate_bias = 0.5 (正值，避免门控值过低)
    - 初始门控值：sigmoid(0.5*cosine+0.5) ∈ [0.62, 0.73] (保守)
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 4,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        use_gate_norm: bool = False,  # 是否在 gate 前使用 LayerNorm
        renormalize_attn: bool = False,  # 是否在门控后重新归一化注意力
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.renormalize_attn = renormalize_attn

        # Q, K 投影（Q 用于 global，K 用于 patches）
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)

        # ========== 反向设计：移除 gate 参数，直接用 attention 加权 cosine ==========
        # 旧版（需要 gate_linear/gate_scale/gate_bias）：
        # self.gate_linear = nn.Linear(1, num_heads, bias=True)
        # 或：self.gate_scale/gate_bias = nn.Parameter(...)

        # 新版（反向设计，无需额外参数）：
        # 直接用 attention 对 cosine 加权，零参数开销
        # score = (attn * cosine).mean(dim=1)

        # 可选：LayerNorm（如果需要的话，当前不使用）
        # self.use_gate_norm = use_gate_norm
        # if use_gate_norm:
        #     self.gate_norm = nn.LayerNorm(num_heads)

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
        if self.k_proj.bias is not None:
            nn.init.zeros_(self.k_proj.bias)

        # ========== 反向设计：无需 gate 参数，因此无需初始化 ==========
        # 旧版需要初始化 gate_linear 或 gate_scale/bias
        # 新版：直接用 attention 加权，无额外参数

    def forward(
        self,
        patches: torch.Tensor,       # (B, N, C) - 键（被检索）
        global_feat: torch.Tensor,   # (B, C) or (B, 1, C) - 查询（检索者）
        cosine_sim: torch.Tensor,    # (B, N) - 预计算的原始空间余弦（各 head 共享）
    ) -> torch.Tensor:
        """
        计算 Cross-Attention score（带逐 head 余弦门控）

        Args:
            patches: (B, N, C) - patch tokens 作为 Key（被检索）
            global_feat: (B, C) or (B, 1, C) - global feature 作为 Query（检索者）
            cosine_sim: (B, N) - 预计算的余弦相似度（原始特征空间，各 head 共享）

        Returns:
            score: (B, N) - 每个 patch 的重要性得分
        """
        B, N, C = patches.shape

        if global_feat.dim() == 2:
            global_feat = global_feat.unsqueeze(1)  # (B, C) -> (B, 1, C)

        # ========== Step 1: Q/K 投影 ==========
        # Q 投影: global 作为查询
        q = self.q_proj(global_feat)  # (B, 1, C)
        q = q.reshape(B, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # (B, num_heads, 1, head_dim)

        # K 投影: patches 作为键
        k = self.k_proj(patches)  # (B, N, C)
        k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # (B, num_heads, N, head_dim)

        # ========== Step 2: Attention Score ==========
        # Q @ K^T / sqrt(d)
        # q: (B, num_heads, 1, head_dim) @ k^T: (B, num_heads, head_dim, N)
        # = (B, num_heads, 1, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, 1, N)

        # 在 N（patch）维度做 softmax
        attn = attn.softmax(dim=-1)  # (B, num_heads, 1, N)
        attn = attn.squeeze(2)  # (B, num_heads, N)

        # 可选：应用 dropout（训练时）
        attn = self.attn_drop(attn)

        # ========== 反向设计：用 attention 调整 cosine（而非 cosine 门控 attention）==========
        # 旧版（用固定 cosine 限制可学习 attention，表达力受限）：
        # cosine_for_gate = cosine_sim.unsqueeze(-1)  # (B, N, 1)
        # gate = sigmoid(self.gate_linear(cosine_for_gate))  # 固定先验
        # attn_gated = attn * gate.transpose(1, 2)  # 被固定先验限制
        # score = normalize(attn_gated).mean(dim=1)

        # 新版（反向：用可学习 attention 调整固定 cosine，更灵活）：
        # 思路：cosine 提供稳定语义基础，attention 学习自适应权重
        cosine_expanded = cosine_sim.unsqueeze(1)  # (B, 1, N)
        # Attention 对 cosine 做逐 head 加权
        weighted_cosine = attn * cosine_expanded  # (B, num_heads, N)
        # 多头平均
        score = weighted_cosine.mean(dim=1)  # (B, N)

        # 优势：
        # 1. Cosine 作为稳定的主体信号（不会乱飘）
        # 2. Attention 学习如何动态调整权重（自适应）
        # 3. 不限制 attention 的表达力
        # 4. 零额外参数，最简洁

        return score


class TokenSparse(nn.Module):
    """
    Token 稀疏选择模块 - 简化版（保持空间结构）

    功能：从 N 个 patch 中选择 K 个最显著的 patch，但不提取，而是将未选中的置零
          K = ceil(N × sparse_ratio)

    改动说明：
        - 移除 MLP predictor，直接用三个注意力得分的平均
        - 不做 token 提取，用 mask 将未选中的 token 置零
        - 保持输出形状 (B, N, C) 不变，维持空间结构
        - 添加可学习的模态权重（修复3）

    综合得分公式：
        旧版：score = (s_im + s_m2 + s_m3) / 3
        新版：score = w1*s_im + w2*s_m2 + w3*s_m3 (w 通过 softmax 归一化)

    Args:
        sparse_ratio: 保留比例 (如 0.6 表示保留 60%)
        use_gumbel: 是否使用 Gumbel-Softmax 可微采样
        gumbel_tau: Gumbel 温度参数
        use_adaptive_weights: 是否使用自适应模态权重（默认 True）
    """

    def __init__(
        self,
        sparse_ratio: float = 0.6,
        use_gumbel: bool = False,
        gumbel_tau: float = 1.0,
        use_adaptive_weights: bool = True,  # 新增：自适应权重
    ):
        super().__init__()

        self.sparse_ratio = sparse_ratio
        self.use_gumbel = use_gumbel
        self.gumbel_tau = gumbel_tau
        self.use_adaptive_weights = use_adaptive_weights

        # ========== 修复3：添加可学习的模态权重 ==========
        if use_adaptive_weights:
            # 3 个可学习权重（通过 softmax 归一化为概率）
            self.modal_weights = nn.Parameter(torch.ones(3))  # 初始化为均等权重

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

        # ========== 修复3：使用自适应模态权重替代简单平均 ==========
        # 旧版（等权平均，无法处理噪声/缺失模态）：
        # score = (s_im + s_m2 + s_m3) / 3  # (B, N)

        # 新版（可学习权重，自适应调整各模态贡献）：
        if self.use_adaptive_weights:
            # 通过 softmax 归一化权重，确保和为 1
            weights = F.softmax(self.modal_weights, dim=0)  # (3,)
            # 加权求和
            score = weights[0] * s_im + weights[1] * s_m2 + weights[2] * s_m3  # (B, N)
        else:
            # 回退到简单平均（兼容旧版）
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
    多模态 SDTPS 模块 - 带 Cross-Attention 和逐 head 余弦门控

    功能：
        1. 对每个模态的 patch 特征进行稀疏选择（soft masking）
        2. 每个模态使用其他两个模态的全局特征作为引导
        3. 结合余弦相似度和可学习的 Cross-Attention
        4. 输出 masked 的多模态特征，保持原始 token 数量

    改动说明：
        - 同时使用余弦相似度和 Cross-Attention（不再二选一）
        - Q/K 反向：global 作为 Query，patches 作为 Key
        - 逐 head 余弦门控：scale 和 bias
        - 在 patch 维度（N）做 softmax

    流程（以 RGB 为例）：
        1. 计算余弦相似度：
           cos_sim = cosine(RGB_patches, RGB_global)  # (B, N)

        2. 计算 Cross-Attention：
           attn = CrossAttn(RGB_patches, RGB_global)  # (B, num_heads, N)

        3. 逐 head 门控：
           gate = sigmoid(cos_sim * scale[h] + bias[h])  # (B, num_heads, N)
           attn_gated = attn * gate

        4. 多头平均 + masking

    Args:
        embed_dim: 特征维度
        sparse_ratio: token 保留比例
        use_gumbel: 是否使用 Gumbel-Softmax
        gumbel_tau: Gumbel 温度
        use_cross_attn: 是否使用 Cross-Attention（默认 True）
        num_heads: 注意力头数（默认 4）
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_patches: int = 128,  # 兼容参数，未使用
        sparse_ratio: float = 0.6,
        aggr_ratio: float = 0.5,  # 兼容参数，未使用
        use_gumbel: bool = False,
        gumbel_tau: float = 1.0,
        beta: float = 0.25,  # 保留参数用于兼容，但不使用
        cross_attn_type: str = 'attention',  # 'attention' or 'cosine'
        cross_attn_heads: int = 4,  # 注意力头数
        use_cross_attn: bool = None,  # 向后兼容，如果为 None 则从 cross_attn_type 推断
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.sparse_ratio = sparse_ratio

        # 处理兼容性：如果 use_cross_attn 未指定，从 cross_attn_type 推断
        if use_cross_attn is None:
            use_cross_attn = (cross_attn_type == 'attention')
        self.use_cross_attn = use_cross_attn

        # 设置注意力头数
        num_heads = cross_attn_heads

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

        # Cross-Attention 模块（如果启用）
        if self.use_cross_attn:
            # RGB 模态的 Cross-Attention
            self.rgb_self_attn = CrossModalAttention(embed_dim, num_heads)
            self.rgb_cross_nir = CrossModalAttention(embed_dim, num_heads)
            self.rgb_cross_tir = CrossModalAttention(embed_dim, num_heads)

            # NIR 模态的 Cross-Attention
            self.nir_self_attn = CrossModalAttention(embed_dim, num_heads)
            self.nir_cross_rgb = CrossModalAttention(embed_dim, num_heads)
            self.nir_cross_tir = CrossModalAttention(embed_dim, num_heads)

            # TIR 模态的 Cross-Attention
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

        # Step 3: Token Masking
        RGB_enhanced, rgb_mask = self.rgb_sparse(
            tokens=RGB_cash,
            self_attention=rgb_self_attn,
            cross_attention_m2=rgb_nir_cross,
            cross_attention_m3=rgb_tir_cross,
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
        )  # (B, N, C), (B, N)

        return RGB_enhanced, NI_enhanced, TI_enhanced, rgb_mask, nir_mask, tir_mask
