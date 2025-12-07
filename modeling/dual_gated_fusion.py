"""
Dual-Gated Adaptive Fusion (DGAF) Module
=========================================

基于 AGFN 论文：Beyond Simple Fusion: Adaptive Gated Fusion for Robust Multimodal Sentiment Analysis

核心思想：
1. Information Entropy Gate (IEG): 基于信息熵评估模态可靠性，低熵=高确定性=高权重
2. Modality Importance Gate (MIG): 学习样本级别的模态重要性权重
3. 可学习参数 α 自适应平衡两个门控的贡献

适配场景：RGB/NIR/TIR 三模态行人/车辆重识别

数学公式：
    # Information Entropy Gate
    H(h_m) = -sum(p * log(p))  # 特征熵
    h_entropy = sum_m softmax(z_m * exp(-H(h_m)/τ)) * h_m

    # Modality Importance Gate
    g = sigmoid(W_g * concat(h_rgb, h_nir, h_tir))  # 门控因子
    h_importance = W_f * [g ⊙ h_rgb, g ⊙ h_nir, g ⊙ h_tir]

    # Adaptive Fusion
    h_fused = α * h_entropy + (1-α) * h_importance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class InformationEntropyGate(nn.Module):
    """
    信息熵门控 (IEG)

    原理：低熵特征表示高确定性/高可靠性，应该给予更高权重

    计算步骤：
    1. 计算每个模态特征的信息熵 H(h_m)
    2. 用 exp(-H/τ) 将熵转换为可靠性得分（熵越低得分越高）
    3. Softmax 归一化得到权重
    4. 加权融合
    """

    def __init__(self, feat_dim: int, tau: float = 1.0):
        super().__init__()
        self.feat_dim = feat_dim
        self.tau = tau

        # 将特征投影到 logits 空间（用于计算注意力）
        self.proj = nn.Linear(feat_dim, feat_dim)

    def compute_entropy(self, feat: torch.Tensor) -> torch.Tensor:
        """
        计算特征的信息熵

        Args:
            feat: (B, C) 特征向量

        Returns:
            entropy: (B,) 每个样本的熵
        """
        # 将特征转换为概率分布（softmax）
        # 使用绝对值确保正数，然后归一化
        feat_abs = torch.abs(feat) + 1e-8
        prob = feat_abs / feat_abs.sum(dim=-1, keepdim=True)

        # 计算熵: H = -sum(p * log(p))
        entropy = -torch.sum(prob * torch.log(prob + 1e-8), dim=-1)  # (B,)

        return entropy

    def forward(
        self,
        h_rgb: torch.Tensor,   # (B, C)
        h_nir: torch.Tensor,   # (B, C)
        h_tir: torch.Tensor,   # (B, C)
    ) -> torch.Tensor:
        """
        信息熵门控融合

        Returns:
            h_entropy: (B, C) 熵加权融合特征
        """
        B, C = h_rgb.shape

        # 1. 计算每个模态的熵
        H_rgb = self.compute_entropy(h_rgb)  # (B,)
        H_nir = self.compute_entropy(h_nir)  # (B,)
        H_tir = self.compute_entropy(h_tir)  # (B,)

        # 2. 计算投影后的 logits（用于注意力计算）
        z_rgb = self.proj(h_rgb).mean(dim=-1)  # (B,)
        z_nir = self.proj(h_nir).mean(dim=-1)  # (B,)
        z_tir = self.proj(h_tir).mean(dim=-1)  # (B,)

        # 3. 用熵调制 logits：低熵 → 高权重
        # score = z * exp(-H/τ)
        score_rgb = z_rgb * torch.exp(-H_rgb / self.tau)  # (B,)
        score_nir = z_nir * torch.exp(-H_nir / self.tau)  # (B,)
        score_tir = z_tir * torch.exp(-H_tir / self.tau)  # (B,)

        # 4. Softmax 归一化
        scores = torch.stack([score_rgb, score_nir, score_tir], dim=-1)  # (B, 3)
        weights = F.softmax(scores, dim=-1)  # (B, 3)

        # 5. 加权融合
        h_entropy = (
            weights[:, 0:1] * h_rgb +  # (B, 1) * (B, C)
            weights[:, 1:2] * h_nir +
            weights[:, 2:3] * h_tir
        )  # (B, C)

        return h_entropy


class ModalityImportanceGate(nn.Module):
    """
    模态重要性门控 (MIG)

    原理：学习样本级别的模态重要性，通过门控因子动态调整每个模态的贡献

    计算步骤：
    1. 拼接所有模态特征
    2. 通过门控网络生成门控因子 g ∈ (0, 1)
    3. 用门控因子对各模态特征进行缩放
    4. 投影融合
    """

    def __init__(self, feat_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        self.feat_dim = feat_dim
        hidden_dim = hidden_dim or feat_dim

        # 门控网络：生成 3 个模态的门控因子
        self.gate_net = nn.Sequential(
            nn.Linear(3 * feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid()  # 输出 ∈ (0, 1)
        )

        # 融合投影
        self.fusion_proj = nn.Sequential(
            nn.Linear(3 * feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True)
        )

    def forward(
        self,
        h_rgb: torch.Tensor,   # (B, C)
        h_nir: torch.Tensor,   # (B, C)
        h_tir: torch.Tensor,   # (B, C)
    ) -> torch.Tensor:
        """
        模态重要性门控融合

        Returns:
            h_importance: (B, C) 门控融合特征
        """
        # 1. 拼接特征
        h_concat = torch.cat([h_rgb, h_nir, h_tir], dim=-1)  # (B, 3C)

        # 2. 生成门控因子
        gates = self.gate_net(h_concat)  # (B, 3)
        g_rgb = gates[:, 0:1]  # (B, 1)
        g_nir = gates[:, 1:2]  # (B, 1)
        g_tir = gates[:, 2:3]  # (B, 1)

        # 3. 门控缩放
        h_rgb_gated = g_rgb * h_rgb  # (B, C)
        h_nir_gated = g_nir * h_nir  # (B, C)
        h_tir_gated = g_tir * h_tir  # (B, C)

        # 4. 拼接并投影
        h_gated_concat = torch.cat([h_rgb_gated, h_nir_gated, h_tir_gated], dim=-1)  # (B, 3C)
        h_importance = self.fusion_proj(h_gated_concat)  # (B, C)

        return h_importance


class DualGatedAdaptiveFusion(nn.Module):
    """
    双门控自适应融合 (DGAF) 模块

    核心公式：
        h_fused = α * h_entropy + (1-α) * h_importance

    其中：
        - h_entropy: 信息熵门控输出（关注可靠性）
        - h_importance: 模态重要性门控输出（关注显著性）
        - α: 可学习参数，平衡两种门控的贡献

    Args:
        feat_dim: 特征维度
        tau: 熵门控的温度参数（默认1.0）
        init_alpha: α 的初始值（默认0.5）
        output_mode: 输出模式
            - 'single': 输出单一融合特征 (B, C)
            - 'concat': 输出拼接特征 (B, 3C) - 保留各模态信息
            - 'both': 返回融合特征和拼接特征
    """

    def __init__(
        self,
        feat_dim: int,
        tau: float = 1.0,
        init_alpha: float = 0.5,
        output_mode: str = 'concat',
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.output_mode = output_mode

        # 信息熵门控
        self.entropy_gate = InformationEntropyGate(feat_dim, tau)

        # 模态重要性门控
        self.importance_gate = ModalityImportanceGate(feat_dim)

        # 可学习的平衡参数 α ∈ [0, 1]
        # 使用 sigmoid 确保范围
        self._alpha = nn.Parameter(torch.tensor(init_alpha))

        # 如果输出模式是 concat，需要额外的融合层
        if output_mode in ['concat', 'both']:
            self.concat_fusion = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.LayerNorm(feat_dim),
            )

    @property
    def alpha(self) -> torch.Tensor:
        """获取约束后的 α 值"""
        return torch.sigmoid(self._alpha)

    def forward(
        self,
        h_rgb: torch.Tensor,   # (B, C)
        h_nir: torch.Tensor,   # (B, C)
        h_tir: torch.Tensor,   # (B, C)
    ) -> Tuple[torch.Tensor, ...]:
        """
        双门控自适应融合

        Args:
            h_rgb: RGB 全局特征 (B, C)
            h_nir: NIR 全局特征 (B, C)
            h_tir: TIR 全局特征 (B, C)

        Returns:
            根据 output_mode 返回不同结果：
            - 'single': fused_feat (B, C)
            - 'concat': concat_feat (B, 3C)
            - 'both': (fused_feat, concat_feat)
        """
        # 1. 信息熵门控
        h_entropy = self.entropy_gate(h_rgb, h_nir, h_tir)  # (B, C)

        # 2. 模态重要性门控
        h_importance = self.importance_gate(h_rgb, h_nir, h_tir)  # (B, C)

        # 3. 自适应融合
        alpha = self.alpha
        h_fused = alpha * h_entropy + (1 - alpha) * h_importance  # (B, C)

        # 4. 根据输出模式返回
        if self.output_mode == 'single':
            return h_fused

        elif self.output_mode == 'concat':
            # 用融合特征增强原始特征后拼接
            h_rgb_enhanced = h_rgb + self.concat_fusion(h_fused)
            h_nir_enhanced = h_nir + self.concat_fusion(h_fused)
            h_tir_enhanced = h_tir + self.concat_fusion(h_fused)
            concat_feat = torch.cat([h_rgb_enhanced, h_nir_enhanced, h_tir_enhanced], dim=-1)
            return concat_feat

        else:  # 'both'
            h_rgb_enhanced = h_rgb + self.concat_fusion(h_fused)
            h_nir_enhanced = h_nir + self.concat_fusion(h_fused)
            h_tir_enhanced = h_tir + self.concat_fusion(h_fused)
            concat_feat = torch.cat([h_rgb_enhanced, h_nir_enhanced, h_tir_enhanced], dim=-1)
            return h_fused, concat_feat


class DualGatedAdaptiveFusionV2(nn.Module):
    """
    双门控自适应融合 V2 - 增强版

    相比 V1 的改进：
    1. 支持 patch 级别的融合（不仅仅是全局特征）
    2. 添加跨模态注意力增强
    3. 更灵活的输出选项

    适用于 DeMo 框架中替代 LIF 和 SACR 模块
    """

    def __init__(
        self,
        feat_dim: int,
        tau: float = 1.0,
        init_alpha: float = 0.5,
        use_cross_modal_attn: bool = True,
        num_heads: int = 4,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.use_cross_modal_attn = use_cross_modal_attn

        # 信息熵门控
        self.entropy_gate = InformationEntropyGate(feat_dim, tau)

        # 模态重要性门控
        self.importance_gate = ModalityImportanceGate(feat_dim)

        # 可学习的平衡参数
        self._alpha = nn.Parameter(torch.tensor(init_alpha))

        # 跨模态注意力（可选）
        if use_cross_modal_attn:
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=feat_dim,
                num_heads=num_heads,
                batch_first=True
            )
            self.cross_attn_norm = nn.LayerNorm(feat_dim)

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
        )

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self._alpha)

    def forward(
        self,
        h_rgb: torch.Tensor,   # (B, C) 全局特征
        h_nir: torch.Tensor,   # (B, C)
        h_tir: torch.Tensor,   # (B, C)
        rgb_tokens: Optional[torch.Tensor] = None,  # (B, N, C) patch 特征（可选）
        nir_tokens: Optional[torch.Tensor] = None,
        tir_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        双门控自适应融合

        Returns:
            rgb_enhanced: (B, C) 增强的 RGB 特征
            nir_enhanced: (B, C) 增强的 NIR 特征
            tir_enhanced: (B, C) 增强的 TIR 特征
        """
        # 1. 双门控融合
        h_entropy = self.entropy_gate(h_rgb, h_nir, h_tir)
        h_importance = self.importance_gate(h_rgb, h_nir, h_tir)

        alpha = self.alpha
        h_fused = alpha * h_entropy + (1 - alpha) * h_importance  # (B, C)

        # 2. 跨模态注意力增强（可选）
        if self.use_cross_modal_attn and rgb_tokens is not None:
            # 使用融合特征作为 query，各模态 tokens 作为 key/value
            h_fused_expanded = h_fused.unsqueeze(1)  # (B, 1, C)

            # RGB 模态增强
            rgb_attn_out, _ = self.cross_attn(
                h_fused_expanded,
                rgb_tokens,
                rgb_tokens
            )
            h_rgb = h_rgb + self.cross_attn_norm(rgb_attn_out.squeeze(1))

            # NIR 模态增强
            nir_attn_out, _ = self.cross_attn(
                h_fused_expanded,
                nir_tokens,
                nir_tokens
            )
            h_nir = h_nir + self.cross_attn_norm(nir_attn_out.squeeze(1))

            # TIR 模态增强
            tir_attn_out, _ = self.cross_attn(
                h_fused_expanded,
                tir_tokens,
                tir_tokens
            )
            h_tir = h_tir + self.cross_attn_norm(tir_attn_out.squeeze(1))

        # 3. 融合特征增强各模态
        h_fused_proj = self.output_proj(h_fused)

        rgb_enhanced = h_rgb + h_fused_proj
        nir_enhanced = h_nir + h_fused_proj
        tir_enhanced = h_tir + h_fused_proj

        return rgb_enhanced, nir_enhanced, tir_enhanced


class DualGatedPostFusion(nn.Module):
    """
    双门控后融合模块 - 专门用于 SDTPS 输出的融合

    设计目的：
        替代 SDTPS 后的简单 concat，使用双门控机制自适应融合三个模态

    输入：
        RGB_sdtps, NI_sdtps, TI_sdtps: 各 (B, C) - SDTPS 输出的增强特征

    输出：
        fused_feat: (B, output_dim) - 融合后的特征
        - 如果 output_dim = C: 单一融合特征
        - 如果 output_dim = 3C: 增强后的拼接特征（保持与原 SDTPS 输出维度一致）

    核心公式：
        1. 信息熵门控: 低熵模态获得更高权重
        2. 重要性门控: 学习样本级别的模态重要性
        3. 自适应融合: h_fused = α * h_entropy + (1-α) * h_importance
    """

    def __init__(
        self,
        feat_dim: int,
        output_dim: int = None,  # 默认为 3 * feat_dim（与原 concat 一致）
        tau: float = 1.0,
        init_alpha: float = 0.5,
        hidden_dim: int = None,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.output_dim = output_dim or (3 * feat_dim)
        self.tau = tau
        hidden_dim = hidden_dim or feat_dim

        # ========== 信息熵门控 (IEG) ==========
        # 投影层用于计算 attention logits
        self.entropy_proj = nn.Linear(feat_dim, feat_dim)

        # ========== 模态重要性门控 (MIG) ==========
        self.gate_net = nn.Sequential(
            nn.Linear(3 * feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid()
        )

        # ========== 可学习的平衡参数 α ==========
        self._alpha = nn.Parameter(torch.tensor(init_alpha))

        # ========== 输出投影 ==========
        if self.output_dim == feat_dim:
            # 单一融合特征
            self.output_proj = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.LayerNorm(feat_dim),
            )
        else:
            # 增强后拼接（保持 3C 维度）
            self.modal_enhance = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.LayerNorm(feat_dim),
            )
            self.output_proj = None

    @property
    def alpha(self) -> torch.Tensor:
        """获取约束后的 α 值"""
        return torch.sigmoid(self._alpha)

    def compute_entropy(self, feat: torch.Tensor) -> torch.Tensor:
        """计算特征的信息熵"""
        feat_abs = torch.abs(feat) + 1e-8
        prob = feat_abs / feat_abs.sum(dim=-1, keepdim=True)
        entropy = -torch.sum(prob * torch.log(prob + 1e-8), dim=-1)
        return entropy

    def forward(
        self,
        h_rgb: torch.Tensor,   # (B, C)
        h_nir: torch.Tensor,   # (B, C)
        h_tir: torch.Tensor,   # (B, C)
    ) -> torch.Tensor:
        """
        双门控自适应融合

        Returns:
            fused_feat: (B, output_dim)
        """
        B, C = h_rgb.shape

        # ========== 1. 信息熵门控 (IEG) ==========
        # 计算每个模态的熵
        H_rgb = self.compute_entropy(h_rgb)
        H_nir = self.compute_entropy(h_nir)
        H_tir = self.compute_entropy(h_tir)

        # 计算投影后的 logits
        z_rgb = self.entropy_proj(h_rgb).mean(dim=-1)
        z_nir = self.entropy_proj(h_nir).mean(dim=-1)
        z_tir = self.entropy_proj(h_tir).mean(dim=-1)

        # 熵调制: 低熵 → 高权重
        score_rgb = z_rgb * torch.exp(-H_rgb / self.tau)
        score_nir = z_nir * torch.exp(-H_nir / self.tau)
        score_tir = z_tir * torch.exp(-H_tir / self.tau)

        # Softmax 归一化
        scores = torch.stack([score_rgb, score_nir, score_tir], dim=-1)
        entropy_weights = F.softmax(scores, dim=-1)  # (B, 3)

        # 熵加权融合
        h_entropy = (
            entropy_weights[:, 0:1] * h_rgb +
            entropy_weights[:, 1:2] * h_nir +
            entropy_weights[:, 2:3] * h_tir
        )  # (B, C)

        # ========== 2. 模态重要性门控 (MIG) ==========
        h_concat = torch.cat([h_rgb, h_nir, h_tir], dim=-1)  # (B, 3C)
        gates = self.gate_net(h_concat)  # (B, 3)

        # 门控缩放
        h_rgb_gated = gates[:, 0:1] * h_rgb
        h_nir_gated = gates[:, 1:2] * h_nir
        h_tir_gated = gates[:, 2:3] * h_tir

        # 门控加权融合
        h_importance = h_rgb_gated + h_nir_gated + h_tir_gated  # (B, C)

        # ========== 3. 自适应融合 ==========
        alpha = self.alpha
        h_fused = alpha * h_entropy + (1 - alpha) * h_importance  # (B, C)

        # ========== 4. 输出 ==========
        if self.output_dim == self.feat_dim:
            # 单一融合特征
            return self.output_proj(h_fused)
        else:
            # 增强后拼接（保持 3C 维度，与原 concat 兼容）
            h_enhance = self.modal_enhance(h_fused)

            # 用融合特征增强各模态
            h_rgb_out = h_rgb + h_enhance
            h_nir_out = h_nir + h_enhance
            h_tir_out = h_tir + h_enhance

            # 拼接输出
            return torch.cat([h_rgb_out, h_nir_out, h_tir_out], dim=-1)  # (B, 3C)


class DualGatedAdaptiveFusionV3(nn.Module):
    """
    双门控自适应融合 V3 - 直接处理 SDTPS 输出

    核心改进：
    1. 内置 Attention Pooling：用可学习 query 从 tokens 中聚合信息（替代 mean pooling）
    2. 直接接受 SDTPS 输出 (B, K+1, C)，无需外部池化
    3. 保留双门控机制：IEG（熵门控）+ MIG（重要性门控）

    流程：
        输入: 3 × (B, K+1, C)  ← SDTPS 直接输出
              ↓
        Attention Pooling (内置)
              ↓
        3 × (B, C)
              ↓
        双门控融合 (IEG + MIG)
              ↓
        输出: (B, 3C)

    Args:
        feat_dim: 特征维度 (512 for CLIP, 768 for ViT)
        tau: 熵门控的温度参数
        init_alpha: α 的初始值（平衡 IEG 和 MIG）
        num_heads: attention pooling 的头数
    """

    def __init__(
        self,
        feat_dim: int,
        output_dim: int = None,
        tau: float = 1.0,
        init_alpha: float = 0.5,
        num_heads: int = 8,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.output_dim = output_dim or (3 * feat_dim)
        self.tau = tau
        self.num_heads = num_heads

        # ========== Attention Pooling (替代 mean pooling) ==========
        # 每个模态有独立的可学习 query
        scale = feat_dim ** -0.5
        self.rgb_query = nn.Parameter(scale * torch.randn(1, 1, feat_dim))
        self.nir_query = nn.Parameter(scale * torch.randn(1, 1, feat_dim))
        self.tir_query = nn.Parameter(scale * torch.randn(1, 1, feat_dim))

        # 共享的 attention 层（减少参数）
        self.attn_pool = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(feat_dim)

        # ========== 信息熵门控 (IEG) ==========
        self.entropy_proj = nn.Linear(feat_dim, feat_dim)

        # ========== 模态重要性门控 (MIG) ==========
        self.gate_net = nn.Sequential(
            nn.Linear(3 * feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, 3),
            nn.Sigmoid()
        )

        # ========== 可学习的平衡参数 α ==========
        self._alpha = nn.Parameter(torch.tensor(init_alpha))

        # ========== 输出投影 ==========
        if self.output_dim == feat_dim:
            self.output_proj = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.LayerNorm(feat_dim),
            )
            self.modal_enhance = None
        else:
            self.output_proj = None
            self.modal_enhance = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.LayerNorm(feat_dim),
            )

    @property
    def alpha(self) -> torch.Tensor:
        """获取约束后的 α 值"""
        return torch.sigmoid(self._alpha)

    def attention_pooling(
        self,
        tokens: torch.Tensor,  # (B, K+1, C)
        query: nn.Parameter,   # (1, 1, C)
    ) -> torch.Tensor:
        """
        Attention Pooling: 用可学习 query 从 tokens 中聚合信息

        Returns:
            pooled: (B, C) 池化后的全局特征
        """
        B = tokens.shape[0]
        query_expanded = query.expand(B, -1, -1)  # (B, 1, C)

        # Cross-attention: query attends to tokens
        pooled, _ = self.attn_pool(query_expanded, tokens, tokens)  # (B, 1, C)
        pooled = self.attn_norm(pooled.squeeze(1))  # (B, C)

        return pooled

    def compute_entropy(self, feat: torch.Tensor) -> torch.Tensor:
        """计算特征的信息熵"""
        feat_abs = torch.abs(feat) + 1e-8
        prob = feat_abs / feat_abs.sum(dim=-1, keepdim=True)
        entropy = -torch.sum(prob * torch.log(prob + 1e-8), dim=-1)
        return entropy

    def forward(
        self,
        rgb_tokens: torch.Tensor,   # (B, K+1, C) - SDTPS 输出
        nir_tokens: torch.Tensor,   # (B, K+1, C)
        tir_tokens: torch.Tensor,   # (B, K+1, C)
    ) -> torch.Tensor:
        """
        双门控自适应融合（直接处理 SDTPS 输出）

        Args:
            rgb_tokens: RGB SDTPS 输出 (B, K+1, C)
            nir_tokens: NIR SDTPS 输出 (B, K+1, C)
            tir_tokens: TIR SDTPS 输出 (B, K+1, C)

        Returns:
            fused_feat: (B, output_dim) - 融合后的特征
        """
        # ========== 1. Attention Pooling ==========
        # 用可学习 query 从 tokens 中聚合信息（替代 mean）
        h_rgb = self.attention_pooling(rgb_tokens, self.rgb_query)  # (B, C)
        h_nir = self.attention_pooling(nir_tokens, self.nir_query)  # (B, C)
        h_tir = self.attention_pooling(tir_tokens, self.tir_query)  # (B, C)

        # ========== 2. 信息熵门控 (IEG) ==========
        H_rgb = self.compute_entropy(h_rgb)
        H_nir = self.compute_entropy(h_nir)
        H_tir = self.compute_entropy(h_tir)

        z_rgb = self.entropy_proj(h_rgb).mean(dim=-1)
        z_nir = self.entropy_proj(h_nir).mean(dim=-1)
        z_tir = self.entropy_proj(h_tir).mean(dim=-1)

        score_rgb = z_rgb * torch.exp(-H_rgb / self.tau)
        score_nir = z_nir * torch.exp(-H_nir / self.tau)
        score_tir = z_tir * torch.exp(-H_tir / self.tau)

        scores = torch.stack([score_rgb, score_nir, score_tir], dim=-1)
        entropy_weights = F.softmax(scores, dim=-1)

        h_entropy = (
            entropy_weights[:, 0:1] * h_rgb +
            entropy_weights[:, 1:2] * h_nir +
            entropy_weights[:, 2:3] * h_tir
        )

        # ========== 3. 模态重要性门控 (MIG) ==========
        h_concat = torch.cat([h_rgb, h_nir, h_tir], dim=-1)
        gates = self.gate_net(h_concat)

        h_rgb_gated = gates[:, 0:1] * h_rgb
        h_nir_gated = gates[:, 1:2] * h_nir
        h_tir_gated = gates[:, 2:3] * h_tir

        h_importance = h_rgb_gated + h_nir_gated + h_tir_gated

        # ========== 4. 自适应融合 ==========
        alpha = self.alpha
        h_fused = alpha * h_entropy + (1 - alpha) * h_importance

        # ========== 5. 输出 ==========
        if self.output_dim == self.feat_dim:
            return self.output_proj(h_fused)
        else:
            h_enhance = self.modal_enhance(h_fused)
            h_rgb_out = h_rgb + h_enhance
            h_nir_out = h_nir + h_enhance
            h_tir_out = h_tir + h_enhance
            return torch.cat([h_rgb_out, h_nir_out, h_tir_out], dim=-1)


if __name__ == "__main__":
    print("=" * 60)
    print("测试 Dual-Gated Adaptive Fusion 模块")
    print("=" * 60)

    B, C = 8, 512

    # 模拟输入
    h_rgb = torch.randn(B, C)
    h_nir = torch.randn(B, C)
    h_tir = torch.randn(B, C)

    # 测试 V1 - single mode
    print("\n[V1 - single mode]")
    dgaf_single = DualGatedAdaptiveFusion(feat_dim=C, output_mode='single')
    out_single = dgaf_single(h_rgb, h_nir, h_tir)
    print(f"  输入: 3 x (B={B}, C={C})")
    print(f"  输出: {out_single.shape}")
    print(f"  α = {dgaf_single.alpha.item():.4f}")

    # 测试 V1 - concat mode
    print("\n[V1 - concat mode]")
    dgaf_concat = DualGatedAdaptiveFusion(feat_dim=C, output_mode='concat')
    out_concat = dgaf_concat(h_rgb, h_nir, h_tir)
    print(f"  输入: 3 x (B={B}, C={C})")
    print(f"  输出: {out_concat.shape}")

    # 测试 V2 - with cross-modal attention
    print("\n[V2 - with cross-modal attention]")
    N = 128  # patch 数量
    rgb_tokens = torch.randn(B, N, C)
    nir_tokens = torch.randn(B, N, C)
    tir_tokens = torch.randn(B, N, C)

    dgaf_v2 = DualGatedAdaptiveFusionV2(feat_dim=C, use_cross_modal_attn=True)
    rgb_out, nir_out, tir_out = dgaf_v2(
        h_rgb, h_nir, h_tir,
        rgb_tokens, nir_tokens, tir_tokens
    )
    print(f"  输入: 全局 3 x (B={B}, C={C}), tokens 3 x (B={B}, N={N}, C={C})")
    print(f"  输出: 3 x {rgb_out.shape}")
    print(f"  α = {dgaf_v2.alpha.item():.4f}")

    # 测试 PostSDTPS fusion
    print("\n[PostSDTPSFusion - 用于 SDTPS 后的融合]")
    post_fusion = DualGatedPostFusion(feat_dim=C, output_dim=3*C)
    fused_out = post_fusion(h_rgb, h_nir, h_tir)
    print(f"  输入: 3 x (B={B}, C={C})")
    print(f"  输出: {fused_out.shape}")
    print(f"  α = {post_fusion.alpha.item():.4f}")

    # 参数统计
    print("\n[参数量统计]")
    print(f"  V1 (single): {sum(p.numel() for p in dgaf_single.parameters()):,}")
    print(f"  V1 (concat): {sum(p.numel() for p in dgaf_concat.parameters()):,}")
    print(f"  V2: {sum(p.numel() for p in dgaf_v2.parameters()):,}")
    print(f"  PostSDTPS: {sum(p.numel() for p in post_fusion.parameters()):,}")

    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)
