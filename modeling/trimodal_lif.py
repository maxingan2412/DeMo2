"""
Trimodal Local Illumination-aware Fusion (Trimodal-LIF) Module
===============================================================

三模态自监督融合模块，支持 RGB + NIR + TIR 图像融合。

每个模态使用独立的质量评估指标（均可从图像自动计算，无需额外标注）：
- RGB: 光照强度 (亮度)
- NIR: 清晰度 (Laplacian方差)
- TIR: 热对比度 (局部标准差)

当前使用方式 (DeMo ReID):
    - 使用 TrimodalLIF.predict_quality() 预测质量图
    - 使用 TrimodalLIFLoss 计算自监督损失
    - 在 make_model.py 中手动实现 patch 级别的质量加权

输入尺寸 (RGBNT201):
    - 图像: (B, 3, 256, 128) - 所有模态都是 3 通道
    - QualityPredictor 输出: (B, 1, 32, 16) - 经过 3 次 AvgPool2d(2,2)
    - Patch 特征: (B, 128, 512) - ViT-B-16 输出，128 = 16×8 patches

Author: Based on M2D-LIF framework (ICCV 2025)
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class Conv(nn.Module):
    """标准卷积块: Conv2d + BatchNorm2d + SiLU"""

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: int = None, g: int = 1, act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p if p is not None else k // 2, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class QualityPredictor(nn.Module):
    """
    质量预测网络，输入图像，输出质量图

    对于 256×128 输入:
        Conv(3, 32) → 256×128
        AvgPool2d   → 128×64
        Conv(32, 64) → 128×64
        AvgPool2d   → 64×32
        Conv(64, 64) → 64×32
        AvgPool2d   → 32×16
        Conv(64, 1) → 32×16

    输出: (B, 1, 32, 16)
    """

    def __init__(self, in_channels: int = 3, mid_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            Conv(in_channels, 32, k=3, p=1),
            nn.AvgPool2d(2, 2),
            Conv(32, mid_channels, k=3, p=1),
            nn.AvgPool2d(2, 2),
            Conv(mid_channels, mid_channels, k=3, p=1),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(mid_channels, 1, 1),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QualityGroundTruth:
    """计算各模态的质量真值（自监督信号）"""

    @staticmethod
    def compute_rgb_quality(rgb: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        """RGB 质量 = 亮度 (ITU-R BT.601)"""
        r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        return F.interpolate(luminance, size=target_size, mode='bilinear', align_corners=False)

    @staticmethod
    def compute_nir_quality(nir: torch.Tensor, target_size: Tuple[int, int], kernel_size: int = 15) -> torch.Tensor:
        """NIR 质量 = Laplacian 方差 (清晰度)"""
        # NIR 图像是 3 通道（复制的），先转为单通道
        if nir.shape[1] == 3:
            nir = nir.mean(dim=1, keepdim=True)

        laplacian_kernel = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
            dtype=nir.dtype, device=nir.device
        ).view(1, 1, 3, 3)

        laplacian = F.conv2d(nir, laplacian_kernel, padding=1)

        # 局部方差
        pad = kernel_size // 2
        local_mean = F.avg_pool2d(laplacian, kernel_size, stride=1, padding=pad)
        local_mean_sq = F.avg_pool2d(laplacian ** 2, kernel_size, stride=1, padding=pad)
        local_var = torch.clamp(local_mean_sq - local_mean ** 2, min=0)

        # 下采样并归一化到 [0, 1]
        quality = F.interpolate(local_var, size=target_size, mode='bilinear', align_corners=False)
        quality = quality / (quality.amax(dim=[2, 3], keepdim=True) + 1e-6)
        return quality

    @staticmethod
    def compute_tir_quality(tir: torch.Tensor, target_size: Tuple[int, int], kernel_size: int = 15) -> torch.Tensor:
        """TIR 质量 = 局部标准差 (热对比度)"""
        # TIR 图像是 3 通道（复制的），先转为单通道
        if tir.shape[1] == 3:
            tir = tir.mean(dim=1, keepdim=True)

        pad = kernel_size // 2
        local_mean = F.avg_pool2d(tir, kernel_size, stride=1, padding=pad)
        local_mean_sq = F.avg_pool2d(tir ** 2, kernel_size, stride=1, padding=pad)
        local_std = torch.sqrt(torch.clamp(local_mean_sq - local_mean ** 2, min=0) + 1e-6)

        # 下采样并归一化到 [0, 1]
        quality = F.interpolate(local_std, size=target_size, mode='bilinear', align_corners=False)
        quality = quality / (quality.amax(dim=[2, 3], keepdim=True) + 1e-6)
        return quality


# NOTE: TrimodalLIFAdd 目前未在 DeMo 中使用
# 该类设计用于 2D 检测特征图 (B, C, H, W)，与 Transformer 特征 (B, N, D) 不兼容
# 保留此类以备将来扩展
class TrimodalLIFAdd(nn.Module):
    """
    三模态特征融合模块 (用于 2D 特征图)

    NOTE: 当前未使用。此类期望输入为 2D 特征图 (B, C, H, W)，
    但 DeMo 使用的是 Transformer 特征 (B, N, D)。
    实际的 patch 级别加权在 make_model.py 中实现。
    """

    def __init__(self, layer: int = 3, beta: float = 0.4):
        super().__init__()
        self.layer = layer
        self.beta = beta
        self.pool2 = nn.AvgPool2d(2, 2)
        self.pool4 = nn.AvgPool2d(4, 4)

    def forward(self, rgb_feat, nir_feat, tir_feat, q_rgb, q_nir, q_tir):
        # 调整质量图尺寸
        if self.layer == 4:
            q_rgb, q_nir, q_tir = self.pool2(q_rgb), self.pool2(q_nir), self.pool2(q_tir)
        elif self.layer == 5:
            q_rgb, q_nir, q_tir = self.pool4(q_rgb), self.pool4(q_nir), self.pool4(q_tir)

        # Softmax 归一化权重
        logits = torch.cat([q_rgb, q_nir, q_tir], dim=1)
        weights = F.softmax(logits * self.beta * 10, dim=1)

        # 加权融合
        return weights[:, 0:1] * rgb_feat + weights[:, 1:2] * nir_feat + weights[:, 2:3] * tir_feat


class TrimodalLIF(nn.Module):
    """
    完整的三模态 LIF 融合模块

    当前 DeMo 中只使用 predict_quality() 方法。
    forward() 方法保留用于 2D 检测特征图场景。
    """

    def __init__(self, beta: float = 0.4, mid_channels: int = 64):
        super().__init__()
        # 注意：RGBNT201 数据集中 NIR 和 TIR 也是 3 通道（复制的）
        self.rgb_predictor = QualityPredictor(in_channels=3, mid_channels=mid_channels)
        self.nir_predictor = QualityPredictor(in_channels=3, mid_channels=mid_channels)
        self.tir_predictor = QualityPredictor(in_channels=3, mid_channels=mid_channels)

        # NOTE: 以下融合模块当前未使用，保留以备将来扩展
        self.fusion_p3 = TrimodalLIFAdd(layer=3, beta=beta)
        self.fusion_p4 = TrimodalLIFAdd(layer=4, beta=beta)
        self.fusion_p5 = TrimodalLIFAdd(layer=5, beta=beta)

    def predict_quality(self, rgb, nir, tir):
        """
        预测三个模态的质量图 (当前使用的主要方法)

        Args:
            rgb: (B, 3, H, W) RGB 图像
            nir: (B, 3, H, W) NIR 图像 (3通道复制)
            tir: (B, 3, H, W) TIR 图像 (3通道复制)

        Returns:
            q_rgb, q_nir, q_tir: 每个形状为 (B, 1, H/8, W/8)
            对于 256×128 输入，输出为 (B, 1, 32, 16)
        """
        return self.rgb_predictor(rgb), self.nir_predictor(nir), self.tir_predictor(tir)

    def forward(self, rgb_img, nir_img, tir_img, rgb_feat, nir_feat, tir_feat, layer: int = 3):
        """
        NOTE: 当前未使用。此方法设计用于 2D 检测特征图。
        DeMo 中的 patch 级别加权在 make_model.py 中直接实现。
        """
        q_rgb, q_nir, q_tir = self.predict_quality(rgb_img, nir_img, tir_img)

        if layer == 3:
            return self.fusion_p3(rgb_feat, nir_feat, tir_feat, q_rgb, q_nir, q_tir)
        elif layer == 4:
            return self.fusion_p4(rgb_feat, nir_feat, tir_feat, q_rgb, q_nir, q_tir)
        elif layer == 5:
            return self.fusion_p5(rgb_feat, nir_feat, tir_feat, q_rgb, q_nir, q_tir)
        else:
            raise ValueError(f"Unsupported layer: {layer}")


class TrimodalLIFLoss(nn.Module):
    """三模态 LIF 自监督损失"""

    def __init__(self, weight_rgb: float = 1.0, weight_nir: float = 1.0, weight_tir: float = 1.0):
        super().__init__()
        self.weight_rgb = weight_rgb
        self.weight_nir = weight_nir
        self.weight_tir = weight_tir
        self.mse = nn.MSELoss()

    def forward(self, q_rgb, q_nir, q_tir, rgb_img, nir_img, tir_img) -> Dict[str, torch.Tensor]:
        target_size = q_rgb.shape[2:]

        gt_rgb = QualityGroundTruth.compute_rgb_quality(rgb_img, target_size)
        gt_nir = QualityGroundTruth.compute_nir_quality(nir_img, target_size)
        gt_tir = QualityGroundTruth.compute_tir_quality(tir_img, target_size)

        loss_rgb = self.mse(q_rgb, gt_rgb)
        loss_nir = self.mse(q_nir, gt_nir)
        loss_tir = self.mse(q_tir, gt_tir)

        total = self.weight_rgb * loss_rgb + self.weight_nir * loss_nir + self.weight_tir * loss_tir

        return {'total': total, 'rgb': loss_rgb, 'nir': loss_nir, 'tir': loss_tir}


if __name__ == "__main__":
    print("=" * 60)
    print("测试 Trimodal-LIF 模块 (DeMo ReID 场景)")
    print("=" * 60)

    # DeMo ReID 场景: 256×128 图像，3 通道
    lif = TrimodalLIF(beta=0.4)
    loss_fn = TrimodalLIFLoss()

    # 模拟 RGBNT201 数据: 所有模态都是 3 通道
    rgb_img = torch.rand(2, 3, 256, 128)  # (B, 3, H, W)
    nir_img = torch.rand(2, 3, 256, 128)  # NIR 也是 3 通道
    tir_img = torch.rand(2, 3, 256, 128)  # TIR 也是 3 通道

    # 测试质量预测
    q_rgb, q_nir, q_tir = lif.predict_quality(rgb_img, nir_img, tir_img)
    print(f"\n输入图像尺寸: {rgb_img.shape}")
    print(f"质量图尺寸: {q_rgb.shape}")  # 期望: (2, 1, 32, 16)

    # 测试损失计算
    losses = loss_fn(q_rgb, q_nir, q_tir, rgb_img, nir_img, tir_img)
    print(f"\nLIF 损失:")
    print(f"  RGB: {losses['rgb'].item():.4f}")
    print(f"  NIR: {losses['nir'].item():.4f}")
    print(f"  TIR: {losses['tir'].item():.4f}")
    print(f"  Total: {losses['total'].item():.4f}")

    # 测试 patch 级别加权 (模拟 make_model.py 中的逻辑)
    print("\n" + "=" * 60)
    print("模拟 patch 级别加权 (make_model.py 中的实际使用方式)")
    print("=" * 60)

    # ViT-B-16 的 patch 特征: 16×8 = 128 patches, 512 维
    patch_h, patch_w = 16, 8
    feat_dim = 512
    RGB_cash = torch.rand(2, 128, feat_dim)  # (B, N, D)
    NI_cash = torch.rand(2, 128, feat_dim)
    TI_cash = torch.rand(2, 128, feat_dim)

    # Resize 质量图到 patch grid 尺寸
    q_rgb_patch = F.interpolate(q_rgb, size=(patch_h, patch_w), mode='bilinear')  # (B, 1, 16, 8)
    q_nir_patch = F.interpolate(q_nir, size=(patch_h, patch_w), mode='bilinear')
    q_tir_patch = F.interpolate(q_tir, size=(patch_h, patch_w), mode='bilinear')
    print(f"\nPatch 质量图尺寸: {q_rgb_patch.shape}")

    # 计算逐位置权重
    q_logits = torch.cat([q_rgb_patch, q_nir_patch, q_tir_patch], dim=1)  # (B, 3, 16, 8)
    q_weights_spatial = F.softmax(q_logits * 10.0, dim=1)  # (B, 3, 16, 8)
    print(f"空间权重图尺寸: {q_weights_spatial.shape}")

    # Reshape 为 token 维度
    w_rgb_token = q_weights_spatial[:, 0:1].flatten(2).transpose(1, 2)  # (B, 128, 1)
    w_nir_token = q_weights_spatial[:, 1:2].flatten(2).transpose(1, 2)
    w_tir_token = q_weights_spatial[:, 2:3].flatten(2).transpose(1, 2)
    print(f"Token 权重尺寸: {w_rgb_token.shape}")

    # 加权 patch 特征
    RGB_weighted = RGB_cash * w_rgb_token  # (B, 128, 512) * (B, 128, 1) → (B, 128, 512)
    NI_weighted = NI_cash * w_nir_token
    TI_weighted = TI_cash * w_tir_token
    print(f"加权后特征尺寸: {RGB_weighted.shape}")

    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)
