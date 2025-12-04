"""
验证 SDTPS 三个模态输出形状的一致性
"""

import torch
from config import cfg
from modeling import make_model

print("=" * 70)
print("验证三个模态增强特征的形状一致性")
print("=" * 70)

# 加载配置
cfg.merge_from_file("configs/RGBNT201/DeMo_SDTPS.yml")
cfg.freeze()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建模型
model = make_model(cfg, num_class=201, camera_num=15, view_num=1)
model = model.to(device)
model.eval()

print(f"\n配置信息:")
print(f"  SDTPS_SPARSE_RATIO: {cfg.MODEL.SDTPS_SPARSE_RATIO}")

# 准备测试数据
batch_size = 4
RGB = torch.randn(batch_size, 3, 256, 128).to(device)
NI = torch.randn(batch_size, 3, 256, 128).to(device)
TI = torch.randn(batch_size, 3, 256, 128).to(device)
cam_label = torch.zeros(batch_size, dtype=torch.long).to(device)

print(f"\n输入形状:")
print(f"  RGB: {RGB.shape}")
print(f"  NI: {NI.shape}")
print(f"  TI: {TI.shape}")

# 前向传播获取 patch 特征
with torch.no_grad():
    # 获取三个模态的 patch 特征
    RGB_cash, RGB_global = model.BACKBONE(RGB, cam_label=cam_label)
    NI_cash, NI_global = model.BACKBONE(NI, cam_label=cam_label)
    TI_cash, TI_global = model.BACKBONE(TI, cam_label=cam_label)

    print(f"\n经过 Backbone 后的 patch 特征:")
    print(f"  RGB_cash: {RGB_cash.shape}")
    print(f"  NI_cash: {NI_cash.shape}")
    print(f"  TI_cash: {TI_cash.shape}")

    # 检查是否一致
    assert RGB_cash.shape == NI_cash.shape == TI_cash.shape, \
        "❌ 三个模态的 patch 特征形状不一致！"
    print(f"  ✅ 三个模态的 patch 数量相同: {RGB_cash.shape[1]}")

    # 处理 GLOBAL_LOCAL
    if model.GLOBAL_LOCAL:
        RGB_local = model.pool(RGB_cash.permute(0, 2, 1)).squeeze(-1)
        NI_local = model.pool(NI_cash.permute(0, 2, 1)).squeeze(-1)
        TI_local = model.pool(TI_cash.permute(0, 2, 1)).squeeze(-1)
        RGB_global = model.rgb_reduce(torch.cat([RGB_global, RGB_local], dim=-1))
        NI_global = model.nir_reduce(torch.cat([NI_global, NI_local], dim=-1))
        TI_global = model.tir_reduce(torch.cat([TI_global, TI_local], dim=-1))

    print(f"\n全局特征:")
    print(f"  RGB_global: {RGB_global.shape}")
    print(f"  NI_global: {NI_global.shape}")
    print(f"  TI_global: {TI_global.shape}")

    # 运行 SDTPS
    RGB_enhanced, NI_enhanced, TI_enhanced, rgb_mask, nir_mask, tir_mask = model.sdtps(
        RGB_cash, NI_cash, TI_cash,
        RGB_global, NI_global, TI_global
    )

    print(f"\n经过 SDTPS 后的增强特征:")
    print(f"  RGB_enhanced: {RGB_enhanced.shape}")
    print(f"  NI_enhanced: {NI_enhanced.shape}")
    print(f"  TI_enhanced: {TI_enhanced.shape}")

    # 检查形状一致性
    print(f"\n形状一致性检查:")
    if RGB_enhanced.shape == NI_enhanced.shape == TI_enhanced.shape:
        print(f"  ✅ 三个模态增强特征形状完全一致: {RGB_enhanced.shape}")
    else:
        print(f"  ❌ 形状不一致！")
        print(f"     RGB: {RGB_enhanced.shape}")
        print(f"     NI: {NI_enhanced.shape}")
        print(f"     TI: {TI_enhanced.shape}")

    # 检查决策矩阵
    print(f"\n决策矩阵 (选择的 patch 数量):")
    print(f"  RGB 选中: {rgb_mask.sum(dim=1).float().mean().item():.1f} / {rgb_mask.shape[1]}")
    print(f"  NI 选中: {nir_mask.sum(dim=1).float().mean().item():.1f} / {nir_mask.shape[1]}")
    print(f"  TI 选中: {tir_mask.sum(dim=1).float().mean().item():.1f} / {tir_mask.shape[1]}")

    # 计算理论上应该选择的数量
    import math
    N = RGB_cash.shape[1]
    expected_K = math.ceil(N * cfg.MODEL.SDTPS_SPARSE_RATIO)
    expected_total = expected_K + 1  # K 个选中的 + 1 个 extra token

    print(f"\n理论分析:")
    print(f"  输入 patch 数量 N: {N}")
    print(f"  sparse_ratio: {cfg.MODEL.SDTPS_SPARSE_RATIO}")
    print(f"  预期选中数量 K: ceil({N} × {cfg.MODEL.SDTPS_SPARSE_RATIO}) = {expected_K}")
    print(f"  预期输出数量: K + 1 (extra) = {expected_total}")
    print(f"  实际输出数量: {RGB_enhanced.shape[1]}")

    if RGB_enhanced.shape[1] == expected_total:
        print(f"  ✅ 实际输出与理论一致！")
    else:
        print(f"  ⚠️ 实际输出与理论不一致")

print("\n" + "=" * 70)
print("✅ 验证完成！")
print("=" * 70)

print("\n结论:")
print("  三个模态增强特征的形状保持一致是因为:")
print("  1. 三个模态使用相同的 Backbone")
print("  2. 输入图像尺寸相同 (256×128)")
print("  3. 输出的 patch 数量 N 相同")
print("  4. 三个模态使用相同的 sparse_ratio")
print("  5. 因此 K = ceil(N × sparse_ratio) 对三个模态是相同的")
print("  6. 最终输出形状都是 (B, K+1, C)")
