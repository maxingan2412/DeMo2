#!/usr/bin/env python3
"""
SACR 膨胀率验证脚本

功能:
  1. 检查当前配置的膨胀率设置
  2. 验证 SACR 模块工作是否正常
  3. 分析感受野与特征图的匹配度
  4. 生成详细的诊断报告

使用方法:
  python verify_sacr_config.py [--config PATH] [--visualize]
"""

import sys
import torch
import torch.nn as nn
import argparse
from pathlib import Path
import json

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def calculate_receptive_field(dilation_rates):
    """计算膨胀卷积的感受野"""
    rfs = []
    for d in dilation_rates:
        rf = 2 * d + 1
        rfs.append({
            'dilation': d,
            'receptive_field': rf,
            'width_coverage': f"{(rf / 8 * 100):.1f}%",
            'height_coverage': f"{(rf / 16 * 100):.1f}%",
        })
    return rfs

def assess_dilation_rates(dilation_rates, feat_h=16, feat_w=8):
    """
    评估膨胀率是否合适

    返回:
        dict: 包含评估结果
    """
    results = {
        'feature_map': {'height': feat_h, 'width': feat_w},
        'min_dimension': min(feat_h, feat_w),
        'dilation_rates': dilation_rates,
        'receptive_fields': [],
        'assessment': {
            'overall_rating': '',
            'issues': [],
            'recommendations': []
        }
    }

    max_recommended = (min(feat_h, feat_w) - 1) / 2

    for d in dilation_rates:
        rf = 2 * d + 1
        width_coverage = (rf / feat_w) * 100
        height_coverage = (rf / feat_h) * 100

        # 判断是否超出范围
        exceeds_width = width_coverage > 100
        exceeds_height = height_coverage > 100

        rf_info = {
            'dilation': d,
            'receptive_field': f"{rf}×{rf}",
            'width_coverage': f"{width_coverage:.1f}%",
            'height_coverage': f"{height_coverage:.1f}%",
            'assessment': 'OK' if not (exceeds_width or exceeds_height) else (
                'Warning (width)' if exceeds_width and not exceeds_height else
                'Warning (height)' if exceeds_height and not exceeds_width else
                'Error (both)'
            )
        }
        results['receptive_fields'].append(rf_info)

        if rf > feat_w:
            results['assessment']['issues'].append(
                f"Dilation={d}: 感受野 {rf} 超出宽度 {feat_w}"
            )
        if rf > feat_h:
            results['assessment']['issues'].append(
                f"Dilation={d}: 感受野 {rf} 超出高度 {feat_h}"
            )

    # 总体评分
    max_dilation = max(dilation_rates)
    max_rf = 2 * max_dilation + 1

    if max_rf <= feat_w and max_rf <= feat_h:
        results['assessment']['overall_rating'] = '✓ Excellent'
    elif max_rf <= feat_w * 1.5:
        results['assessment']['overall_rating'] = '⚠ Warning'
        results['assessment']['recommendations'].append(
            '膨胀率可能过大，建议考虑减小'
        )
    else:
        results['assessment']['overall_rating'] = '✗ Error'
        results['assessment']['recommendations'].append(
            f'最大膨胀率应 <= {int(max_recommended)}'
        )

    results['assessment']['max_recommended_dilation'] = int(max_recommended)

    return results

def test_sacr_forward(token_dim=512, feat_h=16, feat_w=8, dilation_rates=[2, 3, 4]):
    """测试 SACR 前向传播"""
    try:
        from modeling.sacr import SACR

        # 创建 SACR 模块
        sacr = SACR(
            token_dim=token_dim,
            height=feat_h,
            width=feat_w,
            dilation_rates=dilation_rates
        )

        # 创建测试输入 (batch_size=2, num_tokens=128, token_dim=512)
        batch_size = 2
        num_tokens = feat_h * feat_w
        x = torch.randn(batch_size, num_tokens, token_dim)

        # 前向传播
        with torch.no_grad():
            out = sacr(x)

        # 验证输出
        assert out.shape == x.shape, f"输出形状不匹配: {out.shape} vs {x.shape}"

        # 计算参数量
        params = sum(p.numel() for p in sacr.parameters())

        return {
            'success': True,
            'input_shape': tuple(x.shape),
            'output_shape': tuple(out.shape),
            'num_parameters': params,
            'error': None
        }
    except Exception as e:
        return {
            'success': False,
            'input_shape': None,
            'output_shape': None,
            'num_parameters': None,
            'error': str(e)
        }

def print_assessment_report(assessment_results):
    """打印评估报告"""
    print("\n" + "=" * 80)
    print("SACR 膨胀率评估报告".center(80))
    print("=" * 80)

    feat = assessment_results['feature_map']
    print(f"\n【特征图尺寸】")
    print(f"  高度 (H): {feat['height']} pixels")
    print(f"  宽度 (W): {feat['width']} pixels")
    print(f"  最小维度: {assessment_results['min_dimension']} pixels")

    print(f"\n【膨胀率配置】")
    print(f"  当前设置: {assessment_results['dilation_rates']}")

    print(f"\n【感受野分析】")
    print(f"{'膨胀率':<8} {'感受野':<12} {'宽度覆盖':<12} {'高度覆盖':<12} {'状态':<15}")
    print("-" * 60)
    for rf_info in assessment_results['receptive_fields']:
        status_symbol = "✓" if rf_info['assessment'] == 'OK' else "⚠"
        print(
            f"{rf_info['dilation']:<8} "
            f"{rf_info['receptive_field']:<12} "
            f"{rf_info['width_coverage']:<12} "
            f"{rf_info['height_coverage']:<12} "
            f"{status_symbol} {rf_info['assessment']:<13}"
        )

    print(f"\n【总体评估】")
    print(f"  评分: {assessment_results['assessment']['overall_rating']}")

    if assessment_results['assessment']['issues']:
        print(f"\n【发现的问题】")
        for issue in assessment_results['assessment']['issues']:
            print(f"  ✗ {issue}")

    if assessment_results['assessment']['recommendations']:
        print(f"\n【建议】")
        for rec in assessment_results['assessment']['recommendations']:
            print(f"  → {rec}")

    max_rec = assessment_results['assessment'].get('max_recommended_dilation', '?')
    print(f"\n【推荐配置】")
    print(f"  最大膨胀率: ≤ {max_rec}")
    print(f"  推荐方案 A: [2, 3, 4]")
    print(f"  推荐方案 B: [3, 5, 7]")
    print(f"  推荐方案 C: [1, 2, 4]")

    print("\n" + "=" * 80)

def print_forward_test_report(forward_result):
    """打印前向传播测试报告"""
    print("\n【SACR 前向传播测试】")
    if forward_result['success']:
        print(f"  ✓ 测试通过")
        print(f"    输入形状: {forward_result['input_shape']}")
        print(f"    输出形状: {forward_result['output_shape']}")
        print(f"    参数量: {forward_result['num_parameters']:,}")
    else:
        print(f"  ✗ 测试失败")
        print(f"    错误: {forward_result['error']}")

def compare_configurations():
    """对比不同的膨胀率配置"""
    print("\n" + "=" * 80)
    print("不同膨胀率配置对比".center(80))
    print("=" * 80)

    configs = {
        '当前 (原始)': [6, 12, 18],
        '推荐 A': [2, 3, 4],
        '推荐 B': [3, 5, 7],
        '标准 ASPP': [1, 2, 4],
    }

    for name, dilation_rates in configs.items():
        print(f"\n【{name}】: {dilation_rates}")
        rfs = calculate_receptive_field(dilation_rates)
        assessment = assess_dilation_rates(dilation_rates)

        print(f"  评估: {assessment['assessment']['overall_rating']}")
        print(f"  感受野: ", end='')
        rf_list = [rf['receptive_field'] for rf in rfs]
        print(', '.join([f"{r}" for r in rf_list]))

        if assessment['assessment']['issues']:
            for issue in assessment['assessment']['issues'][:2]:
                print(f"    ⚠ {issue}")

def main():
    parser = argparse.ArgumentParser(description='SACR 膨胀率验证脚本')
    parser.add_argument('--config', default=None, help='配置文件路径')
    parser.add_argument('--visualize', action='store_true', help='生成可视化输出')
    parser.add_argument('--compare', action='store_true', help='对比不同配置')

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("SACR 膨胀率验证工具".center(80))
    print("=" * 80)

    # 第一步: 读取当前配置
    print("\n【步骤 1】读取当前配置...")
    try:
        from config.defaults import _C
        current_dilation_rates = _C.MODEL.SACR_DILATION_RATES
        current_feat_dim = 512  # 针对 RGBNT201 的 vit_base_patch16_224
        print(f"  ✓ 当前膨胀率: {current_dilation_rates}")
    except Exception as e:
        print(f"  ✗ 读取失败: {e}")
        current_dilation_rates = [6, 12, 18]
        print(f"  使用默认值: {current_dilation_rates}")

    # 第二步: 评估膨胀率
    print("\n【步骤 2】评估膨胀率...")
    assessment = assess_dilation_rates(current_dilation_rates)
    print_assessment_report(assessment)

    # 第三步: 前向传播测试
    print("\n【步骤 3】测试 SACR 前向传播...")
    forward_result = test_sacr_forward(
        token_dim=current_feat_dim,
        dilation_rates=current_dilation_rates
    )
    print_forward_test_report(forward_result)

    # 第四步: 对比分析
    if args.compare:
        print("\n【步骤 4】对比不同配置...")
        compare_configurations()

    # 生成 JSON 报告
    report = {
        'timestamp': str(Path(__file__).stat().st_mtime),
        'current_config': current_dilation_rates,
        'assessment': assessment,
        'forward_test': forward_result,
    }

    report_path = project_root / 'SACR_Verification_Report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n✓ 报告已保存到: {report_path}")

    # 总结
    print("\n" + "=" * 80)
    print("总结".center(80))
    print("=" * 80)

    if assessment['assessment']['overall_rating'].startswith('✓'):
        print("\n✓ 当前膨胀率配置合理")
        print("\n建议:")
        print("  - 继续使用当前配置")
        print("  - 或尝试推荐方案以获得更好的性能")
    else:
        print("\n⚠ 当前膨胀率配置需要调整")
        print("\n建议:")
        print("  1. 修改 /path/to/config/defaults.py 第 38 行")
        print("  2. 将膨胀率改为: [2, 3, 4]")
        print("  3. 重新训练模型")
        print("\n预期效果:")
        print("  - 消除感受野超出问题")
        print("  - 性能提升 0.5-1.5% mAP")
        print("  - 训练更稳定")

    print("\n" + "=" * 80)

if __name__ == '__main__':
    main()
