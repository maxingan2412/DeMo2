"""
Trimodal Local Illumination-aware Fusion (Trimodal-LIF) Module
===============================================================

三模态自监督融合模块，支持 RGB + NIR + TIR 图像融合。

每个模态使用独立的质量评估指标（均可从图像自动计算，无需额外标注）：
- RGB: 光照强度 (亮度)
- NIR: 清晰度 (Laplacian方差)
- TIR: 热对比度 (局部标准差)

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
    """质量预测网络，输入图像，输出质量图"""

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
        pad = kernel_size // 2
        local_mean = F.avg_pool2d(tir, kernel_size, stride=1, padding=pad)
        local_mean_sq = F.avg_pool2d(tir ** 2, kernel_size, stride=1, padding=pad)
        local_std = torch.sqrt(torch.clamp(local_mean_sq - local_mean ** 2, min=0) + 1e-6)

        # 下采样并归一化到 [0, 1]
        quality = F.interpolate(local_std, size=target_size, mode='bilinear', align_corners=False)
        quality = quality / (quality.amax(dim=[2, 3], keepdim=True) + 1e-6)
        return quality


class TrimodalLIFAdd(nn.Module):
    """三模态特征融合模块"""

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
    """完整的三模态 LIF 融合模块"""

    def __init__(self, beta: float = 0.4, mid_channels: int = 64):
        super().__init__()
        # 注意：RGBNT201 数据集中 NIR 和 TIR 也是 3 通道（复制的）
        self.rgb_predictor = QualityPredictor(in_channels=3, mid_channels=mid_channels)
        self.nir_predictor = QualityPredictor(in_channels=3, mid_channels=mid_channels)  # 3→3
        self.tir_predictor = QualityPredictor(in_channels=3, mid_channels=mid_channels)  # 3→3

        self.fusion_p3 = TrimodalLIFAdd(layer=3, beta=beta)
        self.fusion_p4 = TrimodalLIFAdd(layer=4, beta=beta)
        self.fusion_p5 = TrimodalLIFAdd(layer=5, beta=beta)

    def predict_quality(self, rgb, nir, tir):
        return self.rgb_predictor(rgb), self.nir_predictor(nir), self.tir_predictor(tir)

    def forward(self, rgb_img, nir_img, tir_img, rgb_feat, nir_feat, tir_feat, layer: int = 3):
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
    # 简单测试
    lif = TrimodalLIF(beta=0.4)
    loss_fn = TrimodalLIFLoss()

    rgb_img = torch.rand(2, 3, 640, 640)
    nir_img = torch.rand(2, 1, 640, 640)
    tir_img = torch.rand(2, 1, 640, 640)
    rgb_feat = torch.rand(2, 256, 80, 80)
    nir_feat = torch.rand(2, 256, 80, 80)
    tir_feat = torch.rand(2, 256, 80, 80)

    fused = lif(rgb_img, nir_img, tir_img, rgb_feat, nir_feat, tir_feat, layer=3)
    q_rgb, q_nir, q_tir = lif.predict_quality(rgb_img, nir_img, tir_img)
    losses = loss_fn(q_rgb, q_nir, q_tir, rgb_img, nir_img, tir_img)

    print(f"融合特征: {fused.shape}")
    print(f"总损失: {losses['total'].item():.4f}")
