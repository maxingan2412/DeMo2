"""验证计算的patch数量是否和实际一致"""
import torch
from config import cfg
from modeling import make_model

cfg.merge_from_file('configs/RGBNT201/DeMo_SDTPS.yml')
cfg.freeze()

# 配置计算的patch数量
h, w = cfg.INPUT.SIZE_TRAIN
stride_h, stride_w = cfg.MODEL.STRIDE_SIZE
calculated_patches = (h // stride_h) * (w // stride_w)

print("="*70)
print("验证 patch 数量计算")
print("="*70)
print(f'\n配置计算的patch数量: {calculated_patches}')
print(f'  SIZE_TRAIN: {h}×{w}')
print(f'  STRIDE_SIZE: {stride_h}×{stride_w}')
print(f'  计算: ({h}//{stride_h}) × ({w}//{stride_w}) = {calculated_patches}')

# 创建模型
model = make_model(cfg, num_class=201, camera_num=15, view_num=1)
model = model.cuda()

# 实际运行获取patch数量
RGB = torch.randn(2, 3, 256, 128).cuda()
cam_label = torch.zeros(2, dtype=torch.long).cuda()

with torch.no_grad():
    RGB_cash, RGB_global = model.BACKBONE(RGB, cam_label=cam_label)

print(f'\n实际backbone输出的patch数量: {RGB_cash.shape[1]}')
print(f'  RGB_cash.shape: {RGB_cash.shape}')

print("\n"+"="*70)
if RGB_cash.shape[1] != calculated_patches:
    print(f'⚠️  不一致！计算={calculated_patches}, 实际={RGB_cash.shape[1]}')
    print(f'   差异: {RGB_cash.shape[1] - calculated_patches}')
    print(f'\n这可能导致 aggregation 的 keeped_patches 计算错误！')
else:
    print(f'✓ 一致！patch数量匹配')
print("="*70)
