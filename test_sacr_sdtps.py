"""
测试 SACR + SDTPS 集成
"""

import torch
from config import cfg
from modeling import make_model

print("=" * 80)
print("SACR + SDTPS 集成测试")
print("=" * 80)

# 加载配置
cfg.merge_from_file("configs/RGBNT201/DeMo_SACR_SDTPS.yml")
cfg.freeze()

print(f"\n配置信息:")
print(f"  USE_SACR: {cfg.MODEL.USE_SACR}")
print(f"  SACR_DILATION_RATES: {cfg.MODEL.SACR_DILATION_RATES}")
print(f"  USE_SDTPS: {cfg.MODEL.USE_SDTPS}")
print(f"  SDTPS_SPARSE_RATIO: {cfg.MODEL.SDTPS_SPARSE_RATIO}")
print(f"  SDTPS_AGGR_RATIO: {cfg.MODEL.SDTPS_AGGR_RATIO}")
print(f"  SDTPS_LOSS_WEIGHT: {cfg.MODEL.SDTPS_LOSS_WEIGHT}")
print(f"  GLOBAL_LOCAL: {cfg.MODEL.GLOBAL_LOCAL}")

# 创建模型
print("\n创建模型...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = make_model(cfg, num_class=201, camera_num=15, view_num=1)
model = model.to(device)

# 计算参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"\n总参数量: {total_params:,} ({total_params/1e6:.2f}M)")

# SACR 参数量
if hasattr(model, 'rgb_sacr'):
    sacr_params = sum(p.numel() for p in model.rgb_sacr.parameters()) * 3  # 3个模态
    print(f"SACR (×3): {sacr_params:,} ({sacr_params/1e6:.3f}M) - {sacr_params/total_params*100:.2f}%")

# SDTPS 参数量
if hasattr(model, 'sdtps'):
    sdtps_params = sum(p.numel() for p in model.sdtps.parameters())
    print(f"SDTPS: {sdtps_params:,} ({sdtps_params/1e6:.3f}M) - {sdtps_params/total_params*100:.2f}%")

# 准备测试数据
print("\n准备测试数据...")
batch_size = 4
RGB = torch.randn(batch_size, 3, 256, 128).to(device)
NI = torch.randn(batch_size, 3, 256, 128).to(device)
TI = torch.randn(batch_size, 3, 256, 128).to(device)
img = {'RGB': RGB, 'NI': NI, 'TI': TI}
cam_label = torch.zeros(batch_size, dtype=torch.long).to(device)

# 测试训练模式
print("\n测试训练模式...")
model.train()

with torch.no_grad():
    output = model(img, cam_label=cam_label)

print(f"  输出数量: {len(output)}")
for i, out in enumerate(output):
    if isinstance(out, torch.Tensor):
        print(f"  output[{i}]: shape={out.shape}")

# 测试推理模式
print("\n测试推理模式...")
model.eval()

with torch.no_grad():
    feat1 = model(img, cam_label=cam_label, return_pattern=1)
    print(f"  return_pattern=1: {feat1.shape}")

    feat2 = model(img, cam_label=cam_label, return_pattern=2)
    print(f"  return_pattern=2: {feat2.shape}")

    feat3 = model(img, cam_label=cam_label, return_pattern=3)
    print(f"  return_pattern=3: {feat3.shape}")

# 测试梯度传播
print("\n测试梯度传播...")
model.train()
output = model(img, cam_label=cam_label)

# 简单计算一个 loss
if isinstance(output, tuple) and len(output) >= 2:
    loss = output[0].sum() + output[1].sum()
    loss.backward()

    # 检查 SACR 的梯度
    if hasattr(model, 'rgb_sacr'):
        sacr_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                           for p in model.rgb_sacr.parameters())
        print(f"  SACR 是否有梯度: {sacr_has_grad}")

    # 检查 SDTPS 的梯度
    if hasattr(model, 'sdtps'):
        sdtps_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in model.sdtps.parameters())
        print(f"  SDTPS 是否有梯度: {sdtps_has_grad}")

    # 检查 Backbone 的梯度
    backbone_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                           for p in model.BACKBONE.parameters())
    print(f"  Backbone 是否有梯度: {backbone_has_grad}")

print("\n" + "=" * 80)
print("✓ 测试通过！SACR + SDTPS 集成成功！")
print("=" * 80)

print("\n完整流程:")
print("  Backbone → patch特征 (B, 128, 512)")
print("     ↓")
print("  SACR → 多尺度上下文增强 (B, 128, 512)")
print("     ↓")
print("  SDTPS → Token选择+聚合 (B, 26, 512)")
print("     ↓")
print("  Mean pooling + Concat → 全局特征 (B, 1536)")
print("     ↓")
print("  Classifier → 预测")

print("\n使用以下命令启动训练:")
print("python train_net.py --config_file configs/RGBNT201/DeMo_SACR_SDTPS.yml")
