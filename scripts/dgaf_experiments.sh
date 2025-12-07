#!/bin/bash
# ============================================================================
# DGAF 实验脚本 - Dual-Gated Adaptive Fusion
# ============================================================================

CONFIG_BASE="configs/RGBNT201/DeMo_SACR_SDTPS_LIF.yml"

mkdir -p logs

echo "========== DGAF 实验 =========="

# ============================================================================
# 实验 1: DGAF only (无 SDTPS)
# 目的: 验证 DGAF 单独的效果
# ============================================================================
echo "[实验 1] DGAF only"
CUDA_VISIBLE_DEVICES=0 python train_net.py --config_file $CONFIG_BASE \
    --exp_name "dgaf_only" \
    MODEL.USE_SACR False MODEL.USE_SDTPS False MODEL.USE_LIF False MODEL.USE_DGAF True \
    MODEL.DGAF_TAU 1.0 MODEL.DGAF_INIT_ALPHA 0.5 MODEL.DGAF_USE_CROSS_ATTN True \
    > logs/dgaf_only.log 2>&1 &

# ============================================================================
# 实验 2: DGAF + SDTPS (推荐配置)
# 目的: DGAF 优化全局特征，SDTPS 优化局部 token
# ============================================================================
echo "[实验 2] DGAF + SDTPS"
CUDA_VISIBLE_DEVICES=1 python train_net.py --config_file $CONFIG_BASE \
    --exp_name "dgaf_sdtps" \
    MODEL.USE_SACR False MODEL.USE_SDTPS True MODEL.USE_LIF False MODEL.USE_DGAF True \
    MODEL.DGAF_TAU 1.0 MODEL.DGAF_INIT_ALPHA 0.5 MODEL.DGAF_USE_CROSS_ATTN True \
    MODEL.SDTPS_LOSS_WEIGHT 1.0 \
    > logs/dgaf_sdtps.log 2>&1 &

# ============================================================================
# 实验 3: DGAF + SDTPS + 不同 TAU
# 目的: 探索熵门控温度的影响
# ============================================================================
echo "[实验 3] DGAF + SDTPS (TAU=0.5)"
CUDA_VISIBLE_DEVICES=2 python train_net.py --config_file $CONFIG_BASE \
    --exp_name "dgaf_sdtps_tau0.5" \
    MODEL.USE_SACR False MODEL.USE_SDTPS True MODEL.USE_LIF False MODEL.USE_DGAF True \
    MODEL.DGAF_TAU 0.5 MODEL.DGAF_INIT_ALPHA 0.5 MODEL.DGAF_USE_CROSS_ATTN True \
    MODEL.SDTPS_LOSS_WEIGHT 1.0 \
    > logs/dgaf_sdtps_tau0.5.log 2>&1 &

# ============================================================================
# 实验 4: DGAF + SDTPS + 不同 ALPHA 初始值
# 目的: 探索两个门控的平衡
# ============================================================================
echo "[实验 4] DGAF + SDTPS (ALPHA=0.7)"
CUDA_VISIBLE_DEVICES=3 python train_net.py --config_file $CONFIG_BASE \
    --exp_name "dgaf_sdtps_alpha0.7" \
    MODEL.USE_SACR False MODEL.USE_SDTPS True MODEL.USE_LIF False MODEL.USE_DGAF True \
    MODEL.DGAF_TAU 1.0 MODEL.DGAF_INIT_ALPHA 0.7 MODEL.DGAF_USE_CROSS_ATTN True \
    MODEL.SDTPS_LOSS_WEIGHT 1.0 \
    > logs/dgaf_sdtps_alpha0.7.log 2>&1 &

echo "========== 4 个 DGAF 实验已启动 =========="
echo "监控命令:"
echo "  nvidia-smi"
echo "  tail -f logs/dgaf_*.log"

wait
echo "========== 所有 DGAF 实验完成 =========="
