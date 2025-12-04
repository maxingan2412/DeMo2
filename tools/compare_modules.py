"""
模块对比工具：统计不同配置下的参数量和 FLOPs

对比三种配置：
1. Baseline: 不使用 HDM/ATM/SDTPS
2. Original: 使用 HDM + ATM (原始 DeMo)
3. SDTPS: 使用 SDTPS (新方法)
"""

import torch
import sys
sys.path.insert(0, '.')

from config import cfg as cfg_base
from modeling import make_model
from yacs.config import CfgNode
import copy


def count_parameters(model, verbose=False):
    """
    统计模型参数

    Args:
        model: PyTorch 模型
        verbose: 是否打印详细信息

    Returns:
        total_params: 总参数量
        trainable_params: 可训练参数量
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if verbose:
        print("\n参数详细统计:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"  {name:60s} {param.numel():>12,d}")

    return total_params, trainable_params


def count_module_parameters(model, module_names):
    """
    统计特定模块的参数量

    Args:
        model: PyTorch 模型
        module_names: 模块名称列表（如 ['sdtps', 'generalFusion']）

    Returns:
        dict: {module_name: param_count}
    """
    module_params = {}

    for module_name in module_names:
        if hasattr(model, module_name):
            module = getattr(model, module_name)
            params = sum(p.numel() for p in module.parameters())
            module_params[module_name] = params
        else:
            module_params[module_name] = 0

    return module_params


def compute_flops(model, cfg, skip_flops=False):
    """
    计算模型的 FLOPs

    Args:
        model: PyTorch 模型
        cfg: 配置对象
        skip_flops: 是否跳过 FLOPs 计算（省显存）

    Returns:
        flops: 浮点运算量
    """
    if skip_flops:
        print(f"  (跳过 FLOPs 计算以节省显存)")
        return None

    if hasattr(model, 'flops'):
        try:
            model.cuda()
            flops = model.flops()
            # 立即释放显存
            del model
            torch.cuda.empty_cache()
            return flops
        except Exception as e:
            print(f"  ⚠️  FLOPs 计算失败: {e}")
            return None
    else:
        print(f"  ⚠️  模型没有 flops() 方法")
        return None


def create_config(base_cfg, use_hdm=False, use_atm=False, use_sdtps=False):
    """
    创建配置

    Args:
        base_cfg: 基础配置
        use_hdm: 是否使用 HDM
        use_atm: 是否使用 ATM
        use_sdtps: 是否使用 SDTPS

    Returns:
        cfg: 新配置对象
    """
    cfg = copy.deepcopy(base_cfg)
    cfg.defrost()
    cfg.MODEL.HDM = use_hdm
    cfg.MODEL.ATM = use_atm
    cfg.MODEL.USE_SDTPS = use_sdtps
    cfg.freeze()
    return cfg


def compare_configurations():
    """
    对比不同配置的参数量和 FLOPs
    """
    print("=" * 80)
    print("DeMo 模块对比工具")
    print("=" * 80)

    # 加载基础配置
    cfg_base.merge_from_file("configs/RGBNT201/DeMo.yml")

    # 模型参数
    num_classes = 201
    camera_num = 15
    view_num = 1

    # ==================== 配置1: Baseline ====================
    print("\n[1/3] Baseline: 不使用 HDM/ATM/SDTPS")
    print("-" * 80)

    cfg_baseline = create_config(cfg_base, use_hdm=False, use_atm=False, use_sdtps=False)
    model_baseline = make_model(cfg_baseline, num_class=num_classes, camera_num=camera_num, view_num=view_num)

    total_baseline, trainable_baseline = count_parameters(model_baseline)
    print(f"  总参数量: {total_baseline:,} ({total_baseline/1e6:.2f}M)")
    print(f"  可训练参数: {trainable_baseline:,} ({trainable_baseline/1e6:.2f}M)")

    # 计算 FLOPs（会移到 cuda 并自动释放）
    flops_baseline = compute_flops(model_baseline, cfg_baseline, skip_flops=True)
    if flops_baseline:
        print(f"  FLOPs: {flops_baseline:,.0f} ({flops_baseline/1e9:.2f}G)")

    # ==================== 配置2: HDM + ATM ====================
    print("\n[2/3] Original DeMo: HDM + ATM")
    print("-" * 80)

    cfg_original = create_config(cfg_base, use_hdm=True, use_atm=True, use_sdtps=False)
    model_original = make_model(cfg_original, num_class=num_classes, camera_num=camera_num, view_num=view_num)

    total_original, trainable_original = count_parameters(model_original)
    print(f"  总参数量: {total_original:,} ({total_original/1e6:.2f}M)")
    print(f"  可训练参数: {trainable_original:,} ({trainable_original/1e6:.2f}M)")

    # 统计 HDM+ATM 模块的参数
    module_params_original = count_module_parameters(model_original, ['generalFusion', 'classifier_moe', 'bottleneck_moe'])
    print(f"\n  模块参数详情:")
    for name, params in module_params_original.items():
        print(f"    {name:20s}: {params:>12,d} ({params/1e6:.2f}M)")

    flops_original = compute_flops(model_original, cfg_original, skip_flops=True)
    if flops_original:
        print(f"\n  FLOPs: {flops_original:,.0f} ({flops_original/1e9:.2f}G)")

    # ==================== 配置3: SDTPS ====================
    print("\n[3/3] New Method: SDTPS")
    print("-" * 80)

    cfg_sdtps = cfg_base.clone()
    cfg_sdtps.defrost()
    cfg_sdtps.merge_from_file("configs/RGBNT201/DeMo_SDTPS.yml")
    cfg_sdtps.freeze()

    model_sdtps = make_model(cfg_sdtps, num_class=num_classes, camera_num=camera_num, view_num=view_num)

    total_sdtps, trainable_sdtps = count_parameters(model_sdtps)
    print(f"  总参数量: {total_sdtps:,} ({total_sdtps/1e6:.2f}M)")
    print(f"  可训练参数: {trainable_sdtps:,} ({trainable_sdtps/1e6:.2f}M)")

    # 统计 SDTPS 模块的参数
    module_params_sdtps = count_module_parameters(model_sdtps, ['sdtps', 'classifier_sdtps', 'bottleneck_sdtps'])
    print(f"\n  模块参数详情:")
    for name, params in module_params_sdtps.items():
        print(f"    {name:20s}: {params:>12,d} ({params/1e6:.2f}M)")

    # 进一步细分 SDTPS 内部
    if hasattr(model_sdtps, 'sdtps'):
        sdtps_module = model_sdtps.sdtps
        sdtps_submodules = {
            'rgb_sparse (MLP)': sum(p.numel() for p in sdtps_module.rgb_sparse.parameters()),
            'rgb_aggr (Aggregation)': sum(p.numel() for p in sdtps_module.rgb_aggr.parameters()),
            'nir_sparse': sum(p.numel() for p in sdtps_module.nir_sparse.parameters()),
            'nir_aggr': sum(p.numel() for p in sdtps_module.nir_aggr.parameters()),
            'tir_sparse': sum(p.numel() for p in sdtps_module.tir_sparse.parameters()),
            'tir_aggr': sum(p.numel() for p in sdtps_module.tir_aggr.parameters()),
        }
        print(f"\n  SDTPS 内部模块:")
        for name, params in sdtps_submodules.items():
            print(f"    {name:25s}: {params:>12,d} ({params/1e6:.3f}M)")

    flops_sdtps = compute_flops(model_sdtps, cfg_sdtps, skip_flops=True)
    if flops_sdtps:
        print(f"\n  FLOPs: {flops_sdtps:,.0f} ({flops_sdtps/1e9:.2f}G)")

    # ==================== 对比分析 ====================
    print("\n" + "=" * 80)
    print("对比分析")
    print("=" * 80)

    print("\n参数量对比:")
    print(f"{'配置':<20s} {'总参数':<15s} {'模块参数':<15s} {'vs Baseline':<15s}")
    print("-" * 80)
    print(f"{'Baseline':<20s} {total_baseline/1e6:>12.2f}M {'N/A':<15s} {'0.00M (0.0%)':<15s}")

    original_module_total = sum(module_params_original.values())
    original_increase = total_original - total_baseline
    original_increase_pct = (original_increase / total_baseline) * 100
    print(f"{'HDM+ATM':<20s} {total_original/1e6:>12.2f}M {original_module_total/1e6:>12.2f}M "
          f"+{original_increase/1e6:.2f}M (+{original_increase_pct:.1f}%)")

    sdtps_module_total = sum(module_params_sdtps.values())
    sdtps_increase = total_sdtps - total_baseline
    sdtps_increase_pct = (sdtps_increase / total_baseline) * 100
    print(f"{'SDTPS':<20s} {total_sdtps/1e6:>12.2f}M {sdtps_module_total/1e6:>12.2f}M "
          f"+{sdtps_increase/1e6:.2f}M (+{sdtps_increase_pct:.1f}%)")

    # HDM+ATM vs SDTPS 对比
    print(f"\n{'SDTPS vs HDM+ATM':<20s}")
    param_diff = total_sdtps - total_original
    param_diff_pct = (param_diff / total_original) * 100
    if param_diff > 0:
        print(f"  参数量: +{param_diff/1e6:.2f}M (+{param_diff_pct:.1f}%)")
    else:
        print(f"  参数量: {param_diff/1e6:.2f}M ({param_diff_pct:.1f}%)")

    if flops_baseline and flops_original and flops_sdtps:
        print(f"\nFLOPs 对比:")
        print(f"{'配置':<20s} {'FLOPs (GFLOPs)':<20s} {'vs Baseline':<20s}")
        print("-" * 80)
        print(f"{'Baseline':<20s} {flops_baseline/1e9:>15.2f}G {'0.00G (0.0%)':<20s}")

        original_flops_increase = flops_original - flops_baseline
        original_flops_pct = (original_flops_increase / flops_baseline) * 100
        print(f"{'HDM+ATM':<20s} {flops_original/1e9:>15.2f}G "
              f"+{original_flops_increase/1e9:.2f}G (+{original_flops_pct:.1f}%)")

        sdtps_flops_increase = flops_sdtps - flops_baseline
        sdtps_flops_pct = (sdtps_flops_increase / flops_baseline) * 100
        print(f"{'SDTPS':<20s} {flops_sdtps/1e9:>15.2f}G "
              f"+{sdtps_flops_increase/1e9:.2f}G (+{sdtps_flops_pct:.1f}%)")

        print(f"\n{'SDTPS vs HDM+ATM':<20s}")
        flops_diff = flops_sdtps - flops_original
        flops_diff_pct = (flops_diff / flops_original) * 100
        if flops_diff > 0:
            print(f"  FLOPs: +{flops_diff/1e9:.2f}G (+{flops_diff_pct:.1f}%)")
        else:
            print(f"  FLOPs: {flops_diff/1e9:.2f}G ({flops_diff_pct:.1f}%)")

    # ==================== 保存报告 ====================
    print("\n" + "=" * 80)
    print("保存对比报告")
    print("=" * 80)

    report_path = "results/module_comparison_report.txt"
    import os
    os.makedirs("results", exist_ok=True)

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DeMo 模块对比报告\n")
        f.write("=" * 80 + "\n\n")

        f.write("配置说明:\n")
        f.write("  1. Baseline: 仅使用基础 Backbone + 直接拼接\n")
        f.write("  2. HDM+ATM: 原始 DeMo 方法（层次解耦 + 注意力触发MoE）\n")
        f.write("  3. SDTPS: 新方法（Token 稀疏选择 + 聚合）\n\n")

        f.write("-" * 80 + "\n")
        f.write(f"{'配置':<20s} {'总参数(M)':<15s} {'模块参数(M)':<15s} {'vs Baseline':<20s}\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Baseline':<20s} {total_baseline/1e6:>12.2f}M {'N/A':<15s} {'0.00M (0.0%)':<20s}\n")
        f.write(f"{'HDM+ATM':<20s} {total_original/1e6:>12.2f}M {original_module_total/1e6:>12.2f}M "
                f"+{original_increase/1e6:.2f}M (+{original_increase_pct:.1f}%)\n")
        f.write(f"{'SDTPS':<20s} {total_sdtps/1e6:>12.2f}M {sdtps_module_total/1e6:>12.2f}M "
                f"+{sdtps_increase/1e6:.2f}M (+{sdtps_increase_pct:.1f}%)\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write(f"SDTPS vs HDM+ATM:\n")
        f.write(f"  参数差异: {param_diff/1e6:+.2f}M ({param_diff_pct:+.1f}%)\n")

        if flops_baseline and flops_original and flops_sdtps:
            f.write("\n" + "-" * 80 + "\n")
            f.write(f"{'配置':<20s} {'FLOPs (G)':<20s} {'vs Baseline':<20s}\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Baseline':<20s} {flops_baseline/1e9:>15.2f}G {'0.00G (0.0%)':<20s}\n")
            f.write(f"{'HDM+ATM':<20s} {flops_original/1e9:>15.2f}G "
                    f"+{original_flops_increase/1e9:.2f}G (+{original_flops_pct:.1f}%)\n")
            f.write(f"{'SDTPS':<20s} {flops_sdtps/1e9:>15.2f}G "
                    f"+{sdtps_flops_increase/1e9:.2f}G (+{sdtps_flops_pct:.1f}%)\n")

            f.write("\n" + "-" * 80 + "\n")
            f.write(f"SDTPS vs HDM+ATM:\n")
            f.write(f"  FLOPs 差异: {flops_diff/1e9:+.2f}G ({flops_diff_pct:+.1f}%)\n")

    print(f"\n✓ 报告已保存到: {report_path}")

    # ==================== 返回结果 ====================
    return {
        'baseline': {
            'params': total_baseline,
            'flops': flops_baseline,
        },
        'hdm_atm': {
            'params': total_original,
            'module_params': original_module_total,
            'flops': flops_original,
        },
        'sdtps': {
            'params': total_sdtps,
            'module_params': sdtps_module_total,
            'flops': flops_sdtps,
        },
    }


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')

    try:
        results = compare_configurations()

        print("\n" + "=" * 80)
        print("✓ 对比完成！")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ 对比失败: {e}")
        import traceback
        traceback.print_exc()
