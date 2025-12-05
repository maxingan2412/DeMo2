"""检查 SDTPS 参数量占比"""
import torch
from config import cfg
from modeling import make_model

cfg.merge_from_file("configs/RGBNT201/DeMo_SDTPS.yml")
cfg.freeze()

model = make_model(cfg, num_class=201, camera_num=15, view_num=1)

print("="*80)
print("SDTPS 参数量分析")
print("="*80)

# 总参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"\n总参数量: {total_params:,} ({total_params/1e6:.2f}M)")

# Backbone 参数量
backbone_params = sum(p.numel() for p in model.BACKBONE.parameters())
print(f"Backbone: {backbone_params:,} ({backbone_params/1e6:.2f}M) - {backbone_params/total_params*100:.1f}%")

# SDTPS 参数量
if hasattr(model, 'sdtps'):
    sdtps_params = sum(p.numel() for p in model.sdtps.parameters())
    print(f"SDTPS: {sdtps_params:,} ({sdtps_params/1e6:.2f}M) - {sdtps_params/total_params*100:.2f}%")

    # 细分 SDTPS 内部
    print(f"\n  SDTPS 内部模块:")
    rgb_sparse_params = sum(p.numel() for p in model.sdtps.rgb_sparse.parameters())
    rgb_aggr_params = sum(p.numel() for p in model.sdtps.rgb_aggr.parameters())
    print(f"    rgb_sparse (MLP): {rgb_sparse_params:,} ({rgb_sparse_params/1e6:.3f}M)")
    print(f"    rgb_aggr (Aggr):  {rgb_aggr_params:,} ({rgb_aggr_params/1e6:.3f}M)")
    print(f"    × 3 模态 = {(rgb_sparse_params + rgb_aggr_params)*3:,} ({(rgb_sparse_params + rgb_aggr_params)*3/1e6:.3f}M)")

# 分类器参数量
classifier_sdtps_params = sum(p.numel() for p in model.classifier_sdtps.parameters())
classifier_ori_params = sum(p.numel() for p in model.classifier.parameters())
print(f"\nclassifier_sdtps: {classifier_sdtps_params:,} ({classifier_sdtps_params/1e6:.3f}M)")
print(f"classifier (ori): {classifier_ori_params:,} ({classifier_ori_params/1e6:.3f}M)")

# 原始分支的其他模块
if model.GLOBAL_LOCAL:
    reduce_params = sum(p.numel() for p in model.rgb_reduce.parameters())
    reduce_params += sum(p.numel() for p in model.nir_reduce.parameters())
    reduce_params += sum(p.numel() for p in model.tir_reduce.parameters())
    print(f"Global-Local reduce: {reduce_params:,} ({reduce_params/1e6:.3f}M)")

print("\n"+"="*80)
print("关键发现")
print("="*80)

sdtps_ratio = sdtps_params / total_params * 100
print(f"\nSDTPS 只占总参数的 {sdtps_ratio:.2f}%")
print(f"即使 SDTPS 完全学不到东西，模型还有 {100-sdtps_ratio:.2f}% 的参数在工作")
print(f"\n这可能就是为什么开关 SDTPS 影响不大的原因！")
