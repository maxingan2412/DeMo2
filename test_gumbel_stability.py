"""
测试 Gumbel-Softmax 的数值稳定性
"""

import torch
import torch.nn.functional as F

print("=" * 80)
print("Gumbel-Softmax 数值稳定性测试")
print("=" * 80)

# 模拟 score
batch_size = 4
num_patches = 128

# 测试不同的 score 分布
score_distributions = {
    "随机初始化": torch.randn(batch_size, num_patches),
    "训练初期（接近0）": torch.randn(batch_size, num_patches) * 0.1,
    "训练后期（分布尖锐）": torch.randn(batch_size, num_patches) * 2.0,
}

for name, score in score_distributions.items():
    print(f"\n{'='*80}")
    print(f"测试场景: {name}")
    print(f"{'='*80}")

    print(f"\nScore 统计:")
    print(f"  mean={score.mean().item():.4f}, std={score.std().item():.4f}")
    print(f"  min={score.min().item():.4f}, max={score.max().item():.4f}")

    # 生成 Gumbel 噪声
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(score) + 1e-9) + 1e-9)

    print(f"\nGumbel Noise 统计:")
    print(f"  mean={gumbel_noise.mean().item():.4f}, std={gumbel_noise.std().item():.4f}")
    print(f"  min={gumbel_noise.min().item():.4f}, max={gumbel_noise.max().item():.4f}")

    # 测试不同温度
    for tau in [0.5, 1.0, 5.0, 10.0]:
        print(f"\n  温度 tau={tau}:")

        # 加噪声
        noisy_score = score + gumbel_noise
        print(f"    score+noise: mean={noisy_score.mean().item():.4f}, "
              f"min={noisy_score.min().item():.4f}, max={noisy_score.max().item():.4f}")

        # Softmax
        try:
            soft_mask = F.softmax(noisy_score / tau, dim=1)
            print(f"    soft_mask: mean={soft_mask.mean().item():.4f}, "
                  f"min={soft_mask.min().item():.4f}, max={soft_mask.max().item():.4f}")

            # 检查是否有异常值
            if torch.isnan(soft_mask).any():
                print(f"    ❌ 包含 NaN!")
            if torch.isinf(soft_mask).any():
                print(f"    ❌ 包含 Inf!")
            if soft_mask.max() > 0.99:
                print(f"    ⚠️  分布过于尖锐（max={soft_mask.max().item():.6f}）")

        except Exception as e:
            print(f"    ❌ Softmax 失败: {e}")

# 测试 STE 公式的数值稳定性
print(f"\n{'='*80}")
print(f"测试 STE 公式")
print(f"{'='*80}")

score = torch.randn(batch_size, num_patches)
keep_policy = torch.topk(score, k=64, dim=1)[1]

for tau in [1.0, 5.0]:
    print(f"\n温度 tau={tau}:")

    gumbel_noise = -torch.log(-torch.log(torch.rand_like(score) + 1e-9) + 1e-9)
    soft_mask = F.softmax((score + gumbel_noise) / tau, dim=1)
    hard_mask = torch.zeros_like(score).scatter(1, keep_policy, 1.0)

    # STE
    score_mask = hard_mask + (soft_mask - soft_mask.detach())

    print(f"  hard_mask: sum={hard_mask.sum(dim=1).mean().item():.1f}")
    print(f"  soft_mask: sum={soft_mask.sum(dim=1).mean().item():.6f}")
    print(f"  score_mask: mean={score_mask.mean().item():.6f}, "
          f"min={score_mask.min().item():.6f}, max={score_mask.max().item():.6f}")

    # 检查 selected_mask
    selected_mask = torch.gather(score_mask, dim=1, index=keep_policy)
    print(f"  selected_mask: mean={selected_mask.mean().item():.6f}, "
          f"min={selected_mask.min().item():.6f}, max={selected_mask.max().item():.6f}")

    # 检查梯度
    loss = score_mask.sum()
    loss.backward()

    if score.grad is not None:
        print(f"  score 梯度: mean={score.grad.mean().item():.6f}, "
              f"norm={score.grad.norm().item():.6f}")
        if torch.isnan(score.grad).any():
            print(f"    ❌ 梯度包含 NaN!")
        score.grad.zero_()

print("\n" + "="*80)
print("✓ 测试完成")
print("="*80)
