"""
Scale-Adaptive Contextual Refinement (SACR) Module
即插即用版本

来源: AerialMind: Towards Referring Multi-Object Tracking in UAV Scenarios

用法:
    from sacr import SACR

    # Transformer特征输入 (B, N, D) - batch_size, num_tokens, token_dim
    sacr = SACR(token_dim=512, height=8, width=16)  # 8*16=128 tokens
    x = torch.randn(64, 128, 512)
    out = sacr(x)  # (64, 128, 512)

    # 2D特征图输入 (B, C, H, W)
    sacr = SACR(token_dim=256)
    x = torch.randn(2, 256, 64, 64)
    out = sacr(x)  # (2, 256, 64, 64)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SACR(nn.Module):
    """
    Scale-Adaptive Contextual Refinement

    功能: 扩展感受野 + 通道注意力，增强特征表示
    特点: 输入输出形状相同，即插即用

    Args:
        token_dim: token维度 (对于Transformer特征) 或 通道数 (对于2D特征图)
        height: 指定高度（用于1D序列输入时reshape）
        width: 指定宽度（用于1D序列输入时reshape）
        dilation_rates: 空洞卷积膨胀率列表，默认[6, 12, 18]

    支持两种输入:
        - Transformer特征: (B, N, D) -> (B, N, D)  需指定height和width，且H*W=N
        - 2D特征图:        (B, C, H, W) -> (B, C, H, W)
    """

    def __init__(self, token_dim, height=None, width=None, dilation_rates=[6, 12, 18]):
        super().__init__()

        self.token_dim = token_dim
        self.height = height
        self.width = width
        num_branches = 1 + len(dilation_rates)

        # Part 1: 多尺度空洞卷积 (在token_dim维度上做卷积)
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

    def forward(self, x):
        # 判断输入维度
        if x.dim() == 3:
            # Transformer特征: (B, N, D) -> (B, D, H, W)
            B, N, D = x.shape
            assert self.height is not None and self.width is not None, \
                "Transformer特征输入需要指定height和width"
            assert self.height * self.width == N, \
                f"height*width ({self.height}*{self.width}={self.height*self.width}) 必须等于 num_tokens ({N})"
            # (B, N, D) -> (B, D, H, W)
            x = x.permute(0, 2, 1).view(B, D, self.height, self.width)
            reshape_back = True
        else:
            # 2D特征图: (B, C, H, W)
            reshape_back = False
            B = x.shape[0]

        # 多尺度上下文聚合
        feat_1x1 = self.conv1x1(x)
        feat_atrous = [conv(x) for conv in self.atrous_convs]
        feat_cat = torch.cat([feat_1x1] + feat_atrous, dim=1)
        feat = self.fusion(feat_cat)

        # 通道注意力
        b, c, _, _ = feat.shape
        attn = self.gap(feat).view(b, 1, c)
        attn = self.sigmoid(self.channel_attn(attn)).view(b, c, 1, 1)

        out = feat * attn

        # 如果输入是Transformer特征，转回 (B, N, D)
        if reshape_back:
            # (B, D, H, W) -> (B, D, N) -> (B, N, D)
            out = out.view(B, D, -1).permute(0, 2, 1)

        return out


if __name__ == "__main__":
    print("=" * 50)
    print("测试1: Transformer特征 (B, N, D) = (64, 128, 512)")
    print("       B=64 (batch), N=128 (tokens), D=512 (dim)")
    print("=" * 50)
    sacr_1d = SACR(token_dim=512, height=8, width=16)  # 8*16=128 tokens
    x_1d = torch.randn(64, 128, 512)
    out_1d = sacr_1d(x_1d)
    print(f"输入: {x_1d.shape}")
    print(f"输出: {out_1d.shape}")

    print("\n" + "=" * 50)
    print("测试2: 2D特征图 (B, C, H, W)")
    print("=" * 50)
    sacr_2d = SACR(token_dim=256)
    x_2d = torch.randn(2, 256, 64, 64)
    out_2d = sacr_2d(x_2d)
    print(f"输入: {x_2d.shape}")
    print(f"输出: {out_2d.shape}")

    print("\n" + "=" * 50)
    print(f"参数量 (token_dim=512): {sum(p.numel() for p in sacr_1d.parameters()):,}")
    print("=" * 50)
