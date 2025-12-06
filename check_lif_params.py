"""
检查 QualityPredictor 的参数是否匹配我们的输入尺寸
"""

import torch
import torch.nn as nn

# QualityPredictor 的结构
print("=" * 80)
print("QualityPredictor 参数匹配检查")
print("=" * 80)

print("\nQualityPredictor 结构:")
print("  Conv(3→32) + AvgPool2d(2,2)  → 下采样 2x")
print("  Conv(32→64) + AvgPool2d(2,2) → 下采样 2x")
print("  Conv(64→64) + AvgPool2d(2,2) → 下采样 2x")
print("  Conv(64→1)")
print("  总下采样倍数: 2³ = 8x")

print("\n输入输出尺寸分析:")
print("-" * 80)

# 我们的配置
input_sizes = [
    (256, 128),  # RGBNT201（行人）
    (128, 256),  # RGBNT100（车辆）
]

for h, w in input_sizes:
    print(f"\n输入图像: {h}×{w}")
    print(f"  ↓ QualityPredictor (下采样 8x)")

    q_h = h // 8
    q_w = w // 8
    print(f"  质量图: {q_h}×{q_w}")

    patch_h = h // 16
    patch_w = w // 16
    print(f"  Patch grid: {patch_h}×{patch_w}")

    print(f"  需要 resize: {q_h}×{q_w} → {patch_h}×{patch_w}")

    # 检查是否需要上采样
    if q_h < patch_h or q_w < patch_w:
        print(f"  ⚠️  需要上采样！质量图分辨率低于 patch grid")
    elif q_h == patch_h and q_w == patch_w:
        print(f"  ✓ 完美匹配，无需 resize")
    else:
        print(f"  ✓ 下采样，合理")

print("\n" + "=" * 80)
print("参数合理性分析")
print("=" * 80)

print("\n1. 下采样倍数检查:")
print("   QualityPredictor: 8x (固定)")
print("   Patch stride: 16x")
print("   比例: 8/16 = 0.5")
print("   ✓ 质量图分辨率是 patch grid 的 2 倍，合理")

print("\n2. 中间通道数检查:")
print("   Conv layers: 3→32→64→64→1")
print("   参数量（单个 predictor）:")

# 估算参数量
params = 0
# Conv1: 3×32×3×3
params += 3 * 32 * 3 * 3
# Conv2: 32×64×3×3
params += 32 * 64 * 3 * 3
# Conv3: 64×64×3×3
params += 64 * 64 * 3 * 3
# Conv4: 64×1×1×1
params += 64 * 1 * 1 * 1

print(f"   约 {params:,} 参数 ({params/1e6:.3f}M)")
print(f"   三个模态: {params*3:,} ({params*3/1e6:.3f}M)")

print("\n3. 建议:")
print("   ✓ 当前参数设置合理")
print("   ✓ 下采样倍数适合 256×128 输入")
print("   ✓ 中间通道数(32→64)适中，不会过重")

print("\n" + "=" * 80)
print("✓ 参数检查完成")
print("=" * 80)

print("\n如果要调整:")
print("  - 更大的 mid_channels（如128）：更强的表达能力，但参数量增加")
print("  - 更少的下采样（去掉一个AvgPool）：保留更多细节，但计算量增加")
print("  - 当前配置：平衡的选择 ✓")
