"""
Multi-Modal Scale-Adaptive Contextual Refinement (MultiModalSACR) Module

基于研究者建议：将三个模态的特征沿 token 维度拼接，在 SACR 中进行多模态交互，然后拆分回三个模态。

架构流程：
    RGB_cash (B, N, C)  ─┐
    NI_cash  (B, N, C)  ─┼─→ Concat along token_dim → (B, 3N, C) → SACR → Split → 三个模态
    TI_cash  (B, N, C)  ─┘

优势：
    - 多模态交互：SACR 的空洞卷积可以捕捉跨模态的上下文信息
    - 参数共享：三个模态共用一个 SACR 模块
    - 保持模态独立性：最后拆分回三个模态，便于后续处理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiModalSACR(nn.Module):
    """
    多模态 Scale-Adaptive Contextual Refinement

    功能：将三个模态特征拼接后进行多尺度上下文增强，实现跨模态交互

    Args:
        token_dim: token 维度 (512 for CLIP, 768 for ViT)
        height: patch grid 高度 (256/16 = 16)
        width: patch grid 宽度 (128/16 = 8)
        dilation_rates: 空洞卷积膨胀率列表
        num_modalities: 模态数量，默认 3 (RGB, NIR, TIR)

    输入：
        rgb_tokens: (B, N, C) - RGB 模态 patch 特征
        nir_tokens: (B, N, C) - NIR 模态 patch 特征
        tir_tokens: (B, N, C) - TIR 模态 patch 特征

    输出：
        rgb_enhanced: (B, N, C) - 增强后的 RGB 特征
        nir_enhanced: (B, N, C) - 增强后的 NIR 特征
        tir_enhanced: (B, N, C) - 增强后的 TIR 特征
    """

    def __init__(
        self,
        token_dim: int,
        height: int,
        width: int,
        dilation_rates: list = [2, 3, 4],
        num_modalities: int = 3,
    ):
        super().__init__()

        self.token_dim = token_dim
        self.height = height
        self.width = width
        self.num_tokens = height * width
        self.num_modalities = num_modalities
        num_branches = 1 + len(dilation_rates)

        # Part 1: 多尺度空洞卷积
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(token_dim, token_dim, 1, bias=False),
            nn.BatchNorm2d(token_dim),
            nn.ReLU(inplace=True)
        )

        self.atrous_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(token_dim, token_dim, 3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(token_dim),
                nn.ReLU(inplace=True)
            ) for r in dilation_rates
        ])

        # 融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(token_dim * num_branches, token_dim, 1, bias=False),
            nn.BatchNorm2d(token_dim),
            nn.ReLU(inplace=True)
        )

        # Part 2: 自适应通道注意力
        k = int(abs((math.log2(token_dim) + 1) / 2))
        k = k if k % 2 else k + 1
        k = max(k, 3)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.channel_attn = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()

        # Part 3: 跨模态交互增强（可选）
        # 使用 1x1 卷积在拼接后的特征上进行模态间信息交换
        self.cross_modal_conv = nn.Sequential(
            nn.Conv2d(token_dim, token_dim, 1, bias=False),
            nn.BatchNorm2d(token_dim),
            nn.ReLU(inplace=True)
        )

    def forward(
        self,
        rgb_tokens: torch.Tensor,  # (B, N, C)
        nir_tokens: torch.Tensor,  # (B, N, C)
        tir_tokens: torch.Tensor,  # (B, N, C)
    ):
        """
        多模态 SACR 前向传播

        实现：
            1. 三个模态沿 token 维度拼接: (B, N, C) x 3 -> (B, 3N, C)
            2. Reshape 为 2D: (B, 3N, C) -> (B, C, 3*H, W) 或 (B, C, H, 3*W)
            3. 应用 SACR（多尺度空洞卷积 + 通道注意力）
            4. Reshape 回 1D 并拆分: (B, C, 3*H, W) -> (B, 3N, C) -> 3 x (B, N, C)
        """
        B, N, C = rgb_tokens.shape
        assert N == self.num_tokens, f"Expected {self.num_tokens} tokens, got {N}"

        # Step 1: 拼接三个模态 (沿 token 维度)
        # (B, N, C) x 3 -> (B, 3N, C)
        concat_tokens = torch.cat([rgb_tokens, nir_tokens, tir_tokens], dim=1)

        # Step 2: Reshape 为 2D 特征图
        # 选择沿高度方向拼接: (B, 3N, C) -> (B, C, 3*H, W)
        # 这样空洞卷积可以在垂直方向上捕捉跨模态信息
        concat_2d = concat_tokens.permute(0, 2, 1).view(
            B, C, self.height * self.num_modalities, self.width
        )  # (B, C, 3*H, W) = (B, C, 48, 8) for H=16, W=8

        # Step 3: 多尺度上下文聚合
        feat_1x1 = self.conv1x1(concat_2d)
        feat_atrous = [conv(concat_2d) for conv in self.atrous_convs]
        feat_cat = torch.cat([feat_1x1] + feat_atrous, dim=1)
        feat = self.fusion(feat_cat)

        # Step 4: 通道注意力
        b, c, _, _ = feat.shape
        attn = self.gap(feat).view(b, 1, c)
        attn = self.sigmoid(self.channel_attn(attn)).view(b, c, 1, 1)
        feat = feat * attn

        # Step 5: 跨模态交互增强
        feat = self.cross_modal_conv(feat) + feat  # 残差连接

        # Step 6: Reshape 回 1D 并拆分
        # (B, C, 3*H, W) -> (B, C, 3*N) -> (B, 3*N, C)
        out_tokens = feat.view(B, C, -1).permute(0, 2, 1)  # (B, 3N, C)

        # Step 7: 拆分为三个模态
        rgb_enhanced = out_tokens[:, :N, :]           # (B, N, C)
        nir_enhanced = out_tokens[:, N:2*N, :]        # (B, N, C)
        tir_enhanced = out_tokens[:, 2*N:, :]         # (B, N, C)

        return rgb_enhanced, nir_enhanced, tir_enhanced


class MultiModalSACRv2(nn.Module):
    """
    多模态 SACR V2 - 增强版

    与 V1 的区别：
    - 在拼接前对每个模态进行预处理
    - 添加模态位置编码
    - 更强的跨模态交互
    """

    def __init__(
        self,
        token_dim: int,
        height: int,
        width: int,
        dilation_rates: list = [2, 3, 4],
        num_modalities: int = 3,
    ):
        super().__init__()

        self.token_dim = token_dim
        self.height = height
        self.width = width
        self.num_tokens = height * width
        self.num_modalities = num_modalities
        num_branches = 1 + len(dilation_rates)

        # 模态位置编码（可学习）
        self.modal_embed = nn.Parameter(
            torch.zeros(num_modalities, 1, token_dim)
        )
        nn.init.trunc_normal_(self.modal_embed, std=0.02)

        # 多尺度空洞卷积
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(token_dim, token_dim, 1, bias=False),
            nn.BatchNorm2d(token_dim),
            nn.ReLU(inplace=True)
        )

        self.atrous_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(token_dim, token_dim, 3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(token_dim),
                nn.ReLU(inplace=True)
            ) for r in dilation_rates
        ])

        # 融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(token_dim * num_branches, token_dim, 1, bias=False),
            nn.BatchNorm2d(token_dim),
            nn.ReLU(inplace=True)
        )

        # 通道注意力
        k = int(abs((math.log2(token_dim) + 1) / 2))
        k = k if k % 2 else k + 1
        k = max(k, 3)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.channel_attn = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()

        # 跨模态注意力
        self.cross_modal_attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=8,
            batch_first=True
        )
        self.cross_modal_norm = nn.LayerNorm(token_dim)

    def forward(
        self,
        rgb_tokens: torch.Tensor,
        nir_tokens: torch.Tensor,
        tir_tokens: torch.Tensor,
    ):
        B, N, C = rgb_tokens.shape

        # 添加模态位置编码
        rgb_tokens = rgb_tokens + self.modal_embed[0]
        nir_tokens = nir_tokens + self.modal_embed[1]
        tir_tokens = tir_tokens + self.modal_embed[2]

        # 拼接
        concat_tokens = torch.cat([rgb_tokens, nir_tokens, tir_tokens], dim=1)

        # Reshape 为 2D
        concat_2d = concat_tokens.permute(0, 2, 1).view(
            B, C, self.height * self.num_modalities, self.width
        )

        # 多尺度上下文聚合
        feat_1x1 = self.conv1x1(concat_2d)
        feat_atrous = [conv(concat_2d) for conv in self.atrous_convs]
        feat_cat = torch.cat([feat_1x1] + feat_atrous, dim=1)
        feat = self.fusion(feat_cat)

        # 通道注意力
        b, c, _, _ = feat.shape
        attn = self.gap(feat).view(b, 1, c)
        attn = self.sigmoid(self.channel_attn(attn)).view(b, c, 1, 1)
        feat = feat * attn

        # Reshape 回 1D
        out_tokens = feat.view(B, C, -1).permute(0, 2, 1)

        # 跨模态注意力增强
        out_tokens = out_tokens + self.cross_modal_attn(
            self.cross_modal_norm(out_tokens),
            self.cross_modal_norm(out_tokens),
            self.cross_modal_norm(out_tokens)
        )[0]

        # 拆分
        rgb_enhanced = out_tokens[:, :N, :]
        nir_enhanced = out_tokens[:, N:2*N, :]
        tir_enhanced = out_tokens[:, 2*N:, :]

        return rgb_enhanced, nir_enhanced, tir_enhanced


if __name__ == "__main__":
    print("=" * 60)
    print("测试 MultiModalSACR")
    print("=" * 60)

    B, N, C = 8, 128, 512  # batch=8, tokens=128 (16x8), dim=512
    H, W = 16, 8

    rgb = torch.randn(B, N, C)
    nir = torch.randn(B, N, C)
    tir = torch.randn(B, N, C)

    # 测试 V1
    print("\n[MultiModalSACR V1]")
    sacr_v1 = MultiModalSACR(token_dim=C, height=H, width=W)
    rgb_out, nir_out, tir_out = sacr_v1(rgb, nir, tir)
    print(f"  输入: 3 x (B={B}, N={N}, C={C})")
    print(f"  输出: 3 x {rgb_out.shape}")
    print(f"  参数量: {sum(p.numel() for p in sacr_v1.parameters()):,}")

    # 测试 V2
    print("\n[MultiModalSACR V2 (with cross-modal attention)]")
    sacr_v2 = MultiModalSACRv2(token_dim=C, height=H, width=W)
    rgb_out2, nir_out2, tir_out2 = sacr_v2(rgb, nir, tir)
    print(f"  输入: 3 x (B={B}, N={N}, C={C})")
    print(f"  输出: 3 x {rgb_out2.shape}")
    print(f"  参数量: {sum(p.numel() for p in sacr_v2.parameters()):,}")

    print("\n" + "=" * 60)
    print("测试通过!")
    print("=" * 60)
