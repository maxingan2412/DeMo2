"""
诊断 SDTPS 训练问题

检查项：
1. 模型输出格式是否正确
2. 梯度是否正常传播到 SDTPS 模块
3. SDTPS 特征的数值范围
4. 损失计算是否正确
5. 与 Baseline 对比
"""

import torch
import torch.nn.functional as F
from config import cfg
from modeling import make_model
from layers.make_loss import make_loss

print("=" * 80)
print("SDTPS 训练问题诊断")
print("=" * 80)

# 加载配置
cfg.merge_from_file("configs/RGBNT201/DeMo_SDTPS.yml")
cfg.freeze()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建模型
print("\n[1/5] 创建模型...")
model = make_model(cfg, num_class=201, camera_num=15, view_num=1)
model = model.to(device)
model.train()

print(f"  模型配置:")
print(f"    USE_SDTPS: {cfg.MODEL.USE_SDTPS}")
print(f"    HDM: {cfg.MODEL.HDM}")
print(f"    ATM: {cfg.MODEL.ATM}")
print(f"    GLOBAL_LOCAL: {cfg.MODEL.GLOBAL_LOCAL}")

# 准备测试数据
print("\n[2/5] 准备测试数据...")
batch_size = 8
RGB = torch.randn(batch_size, 3, 256, 128).to(device)
NI = torch.randn(batch_size, 3, 256, 128).to(device)
TI = torch.randn(batch_size, 3, 256, 128).to(device)
img = {'RGB': RGB, 'NI': NI, 'TI': TI}
target = torch.randint(0, 201, (batch_size,)).to(device)
cam_label = torch.zeros(batch_size, dtype=torch.long).to(device)

# 前向传播
print("\n[3/5] 前向传播...")
output = model(img, label=target, cam_label=cam_label)

print(f"  输出格式:")
print(f"    type: {type(output)}")
print(f"    length: {len(output)}")
for i, out in enumerate(output):
    if isinstance(out, torch.Tensor):
        print(f"    output[{i}]: shape={out.shape}, dtype={out.dtype}")
        print(f"              mean={out.mean().item():.4f}, std={out.std().item():.4f}")
        print(f"              min={out.min().item():.4f}, max={out.max().item():.4f}")
        if torch.isnan(out).any():
            print(f"              ⚠️  包含 NaN!")
        if torch.isinf(out).any():
            print(f"              ⚠️  包含 Inf!")

# 计算准确率
print("\n[4/5] 计算准确率...")
if isinstance(output, list):
    acc = (output[0][0].max(1)[1] == target).float().mean()
else:
    acc = (output[0].max(1)[1] == target).float().mean()

print(f"  准确率: {acc.item():.3f} ({acc.item()*100:.1f}%)")
print(f"  预测分布: {output[0].argmax(dim=1)[:10].tolist()}")
print(f"  真实标签: {target[:10].tolist()}")

# 计算损失
print("\n[5/5] 计算损失...")
loss_fn, center_criterion = make_loss(cfg, num_classes=201)

total_loss = 0
if len(output) % 2 == 1:
    print(f"  输出长度为奇数: {len(output)}")
    index = len(output) - 1
    for i in range(0, index, 2):
        loss_tmp = loss_fn(score=output[i], feat=output[i+1], target=target, target_cam=cam_label)
        print(f"    loss[{i//2}] (score={i}, feat={i+1}): {loss_tmp.item():.4f}")
        total_loss = total_loss + loss_tmp
    print(f"    额外损失 output[-1]: {output[-1].item():.4f}")
    total_loss = total_loss + output[-1]
else:
    print(f"  输出长度为偶数: {len(output)}")
    for i in range(0, len(output), 2):
        loss_tmp = loss_fn(score=output[i], feat=output[i+1], target=target, target_cam=cam_label)
        print(f"    loss[{i//2}] (score={i}, feat={i+1}): {loss_tmp.item():.4f}")
        total_loss = total_loss + loss_tmp

print(f"\n  总损失: {total_loss.item():.4f}")

# 检查梯度
print("\n" + "=" * 80)
print("梯度检查")
print("=" * 80)

model.zero_grad()
total_loss.backward()

print("\n检查 SDTPS 模块的梯度:")
if hasattr(model, 'sdtps'):
    sdtps = model.sdtps

    # 检查 TokenSparse 的梯度
    for name, param in sdtps.rgb_sparse.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"  rgb_sparse.{name:40s}: grad_norm={grad_norm:.6f}")
        else:
            print(f"  rgb_sparse.{name:40s}: ⚠️  NO GRAD")

    # 检查 TokenAggregation 的梯度
    for name, param in sdtps.rgb_aggr.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"  rgb_aggr.{name:40s}: grad_norm={grad_norm:.6f}")
        else:
            print(f"  rgb_aggr.{name:40s}: ⚠️  NO GRAD")

print("\n检查分类器的梯度:")
for name, param in [('classifier_sdtps.weight', model.classifier_sdtps.weight),
                     ('classifier.weight', model.classifier.weight)]:
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        print(f"  {name:40s}: grad_norm={grad_norm:.6f}")
    else:
        print(f"  {name:40s}: ⚠️  NO GRAD")

print("\n检查 Backbone 的梯度（采样前几层）:")
for name, param in list(model.BACKBONE.named_parameters())[:5]:
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        print(f"  {name:40s}: grad_norm={grad_norm:.6f}")
    else:
        print(f"  {name:40s}: ⚠️  NO GRAD")

# 分析可能的问题
print("\n" + "=" * 80)
print("问题诊断")
print("=" * 80)

issues = []

if total_loss.item() > 10:
    issues.append("⚠️  损失过大 (>10)")

if acc.item() < 0.1:
    issues.append("⚠️  准确率过低 (<10%)")

if torch.isnan(total_loss):
    issues.append("❌ 损失为 NaN")

# 检查是否有 NaN 输出
has_nan = any(torch.isnan(out).any() if isinstance(out, torch.Tensor) else False for out in output)
if has_nan:
    issues.append("❌ 模型输出包含 NaN")

# 检查梯度
if hasattr(model, 'sdtps'):
    sdtps_has_grad = any(p.grad is not None for p in model.sdtps.parameters())
    if not sdtps_has_grad:
        issues.append("❌ SDTPS 模块没有梯度！")

if issues:
    print("\n发现的问题:")
    for issue in issues:
        print(f"  {issue}")
else:
    print("\n✓ 未发现明显问题")

print("\n可能的原因分析:")
print("  1. SDTPS 特征质量差 - 过度压缩导致信息丢失")
print("  2. 分类器未充分训练 - SDTPS 特征空间需要更多轮数适应")
print("  3. 学习率不匹配 - SDTPS 模块可能需要不同的学习率")
print("  4. Attention 在 no_grad 里 - 虽然是原论文设计，但可能影响学习")
print("  5. 初始化问题 - SDTPS 模块参数初始化可能不合适")

print("\n建议的调试步骤:")
print("  1. 对比 Baseline（不用 SDTPS）的训练曲线")
print("  2. 检查 SDTPS 输出特征的质量（可视化、相似度等）")
print("  3. 尝试更大的学习率或为 SDTPS 设置独立学习率")
print("  4. 检查是否需要预训练 SDTPS 模块")
print("  5. 尝试先不用 aggregation，只用 TokenSparse")
