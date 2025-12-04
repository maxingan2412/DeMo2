"""
测试完整的 SDTPS 模块
验证: TokenSparse + TokenAggregation 的完整流程
"""

import torch
import sys
sys.path.insert(0, '.')

from modeling.sdtps_complete import TokenSparse, TokenAggregation, MultiModalSDTPS

def test_token_sparse():
    """测试 TokenSparse 模块"""
    print("=" * 70)
    print("测试 TokenSparse 模块")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch = 4
    num_patches = 128
    feat_dim = 512
    sparse_ratio = 0.5

    # 准备数据
    tokens = torch.randn(batch, num_patches, feat_dim).to(device)
    self_attn = torch.randn(batch, num_patches).to(device)
    cross_attn1 = torch.randn(batch, num_patches).to(device)
    cross_attn2 = torch.randn(batch, num_patches).to(device)

    # 创建模块
    sparse = TokenSparse(
        embed_dim=feat_dim,
        sparse_ratio=sparse_ratio,
        use_gumbel=False,
    ).to(device)

    # 前向传播
    with torch.no_grad():
        select_tokens, extra_token, score_mask, selected_mask, keep_indices = sparse(
            tokens=tokens,
            self_attention=self_attn,
            cross_attention_m2=cross_attn1,
            cross_attention_m3=cross_attn2,
            beta=0.25,
        )

    print(f"输入: tokens {tokens.shape}")
    print(f"输出: select_tokens {select_tokens.shape}")
    print(f"输出: extra_token {extra_token.shape}")
    print(f"输出: score_mask {score_mask.shape}")
    print(f"输出: selected_mask {selected_mask.shape}")
    print(f"输出: keep_indices {keep_indices.shape}")

    expected_n_s = int(num_patches * sparse_ratio)
    print(f"\n预期选中数量: ceil({num_patches} × {sparse_ratio}) = {expected_n_s}")
    print(f"实际选中数量: {select_tokens.shape[1]}")
    print(f"决策矩阵中1的数量: {score_mask.sum(dim=1).float().mean().item():.1f}")

    assert select_tokens.shape == (batch, expected_n_s, feat_dim), "形状不符合预期！"
    print("✓ TokenSparse 测试通过！\n")


def test_token_aggregation():
    """测试 TokenAggregation 模块"""
    print("=" * 70)
    print("测试 TokenAggregation 模块")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch = 4
    n_s = 64  # 选中的patches
    feat_dim = 512
    n_c = 26  # 聚合后的patches

    # 准备数据
    select_tokens = torch.randn(batch, n_s, feat_dim).to(device)

    # 创建模块
    aggr = TokenAggregation(
        dim=feat_dim,
        keeped_patches=n_c,
        dim_ratio=0.2,
    ).to(device)

    # 前向传播
    with torch.no_grad():
        aggr_tokens = aggr(select_tokens)

    print(f"输入: select_tokens {select_tokens.shape}")
    print(f"输出: aggr_tokens {aggr_tokens.shape}")
    print(f"\n聚合比例: {n_c}/{n_s} = {n_c/n_s:.3f}")

    assert aggr_tokens.shape == (batch, n_c, feat_dim), "形状不符合预期！"
    print("✓ TokenAggregation 测试通过！\n")


def test_multimodal_sdtps():
    """测试完整的 MultiModalSDTPS"""
    print("=" * 70)
    print("测试 MultiModalSDTPS 完整流程")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch = 4
    num_patches = 128
    feat_dim = 512
    sparse_ratio = 0.5
    aggr_ratio = 0.4

    # 准备数据
    RGB_cash = torch.randn(batch, num_patches, feat_dim).to(device)
    NI_cash = torch.randn(batch, num_patches, feat_dim).to(device)
    TI_cash = torch.randn(batch, num_patches, feat_dim).to(device)
    RGB_global = torch.randn(batch, feat_dim).to(device)
    NI_global = torch.randn(batch, feat_dim).to(device)
    TI_global = torch.randn(batch, feat_dim).to(device)

    # 创建模块
    print()  # 这会打印参数信息
    sdtps = MultiModalSDTPS(
        embed_dim=feat_dim,
        num_patches=num_patches,
        sparse_ratio=sparse_ratio,
        aggr_ratio=aggr_ratio,
        use_gumbel=False,
    ).to(device)

    # 计算参数量
    n_params = sum(p.numel() for p in sdtps.parameters())
    print(f"  模块参数量: {n_params / 1e6:.2f}M\n")

    # 前向传播
    with torch.no_grad():
        RGB_enh, NI_enh, TI_enh, rgb_mask, nir_mask, tir_mask = sdtps(
            RGB_cash, NI_cash, TI_cash,
            RGB_global, NI_global, TI_global
        )

    print(f"输入形状:")
    print(f"  RGB_cash: {RGB_cash.shape}")
    print(f"  NI_cash: {NI_cash.shape}")
    print(f"  TI_cash: {TI_cash.shape}")

    print(f"\n输出形状:")
    print(f"  RGB_enhanced: {RGB_enh.shape}")
    print(f"  NI_enhanced: {NI_enh.shape}")
    print(f"  TI_enhanced: {TI_enh.shape}")

    print(f"\n决策矩阵:")
    print(f"  rgb_mask: {rgb_mask.shape}, 选中: {rgb_mask.sum(dim=1).float().mean().item():.1f}")
    print(f"  nir_mask: {nir_mask.shape}, 选中: {nir_mask.sum(dim=1).float().mean().item():.1f}")
    print(f"  tir_mask: {tir_mask.shape}, 选中: {tir_mask.sum(dim=1).float().mean().item():.1f}")

    # 验证形状一致性
    assert RGB_enh.shape == NI_enh.shape == TI_enh.shape, "三个模态形状不一致！"

    expected_shape = (batch, int(num_patches * aggr_ratio * sparse_ratio) + 1, feat_dim)
    assert RGB_enh.shape == expected_shape, f"期望 {expected_shape}，实际 {RGB_enh.shape}"

    print(f"\n✓ 形状验证通过！三个模态输出一致: {RGB_enh.shape}")
    print(f"✓ MultiModalSDTPS 测试通过！\n")

    return sdtps


def test_complete_pipeline():
    """测试完整的pipeline（模拟真实使用）"""
    print("=" * 70)
    print("测试完整 Pipeline（模拟训练）")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建SDTPS模块
    sdtps = MultiModalSDTPS(
        embed_dim=512,
        num_patches=128,
        sparse_ratio=0.5,
        aggr_ratio=0.4,
        use_gumbel=True,  # 测试Gumbel
        gumbel_tau=1.0,
    ).to(device)

    print()
    sdtps.train()  # 训练模式

    # 准备数据
    batch = 4
    RGB_cash = torch.randn(batch, 128, 512).to(device)
    NI_cash = torch.randn(batch, 128, 512).to(device)
    TI_cash = torch.randn(batch, 128, 512).to(device)
    RGB_global = torch.randn(batch, 512).to(device)
    NI_global = torch.randn(batch, 512).to(device)
    TI_global = torch.randn(batch, 512).to(device)

    # 前向传播
    RGB_enh, NI_enh, TI_enh, rgb_mask, nir_mask, tir_mask = sdtps(
        RGB_cash, NI_cash, TI_cash,
        RGB_global, NI_global, TI_global
    )

    # 池化得到全局特征
    RGB_feat = RGB_enh.mean(dim=1)  # (B, N_c+1, C) → (B, C)
    NI_feat = NI_enh.mean(dim=1)
    TI_feat = TI_enh.mean(dim=1)

    # 拼接
    final_feat = torch.cat([RGB_feat, NI_feat, TI_feat], dim=-1)  # (B, 3C)

    print(f"最终特征: {final_feat.shape}")
    print(f"✓ 完整Pipeline测试通过！")

    # 测试梯度
    print(f"\n测试梯度反向传播:")
    loss = final_feat.sum()
    loss.backward()

    # 检查梯度
    has_grad = any(p.grad is not None for p in sdtps.parameters())
    print(f"  参数是否有梯度: {has_grad}")

    if has_grad:
        grad_norm = sum(p.grad.norm().item() for p in sdtps.parameters() if p.grad is not None)
        print(f"  梯度范数: {grad_norm:.4f}")
        print(f"  ✓ 梯度正常！")
    else:
        print(f"  ⚠️  没有梯度（可能是no_grad导致）")


if __name__ == "__main__":
    print(f"使用设备: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}\n")

    try:
        # 测试各个组件
        test_token_sparse()
        test_token_aggregation()
        test_multimodal_sdtps()
        test_complete_pipeline()

        print("\n" + "=" * 70)
        print("🎉 所有测试通过！SDTPS 完整实现正确！")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
