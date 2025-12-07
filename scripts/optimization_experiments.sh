#!/bin/bash
# ============================================================================
# 优化实验脚本 - 基于消融实验结果的参数探索
# ============================================================================

CONFIG_BASE="configs/RGBNT201/DeMo_SACR_SDTPS_LIF.yml"

mkdir -p logs

echo "========== 优化实验 =========="

# ============================================================================
# 实验 1: SDTPS only + SDTPS_LOSS=1.0 (核心优化)
# 预期: 结合两个最佳设置
# ============================================================================
echo "[实验 1] SDTPS only + LOSS=1.0"
CUDA_VISIBLE_DEVICES=0 python train_net.py --config_file $CONFIG_BASE \
    --exp_name "opt_sdtps_loss1.0" \
    MODEL.USE_SACR False MODEL.USE_SDTPS True MODEL.USE_LIF False \
    MODEL.SDTPS_LOSS_WEIGHT 1.0 \
    > logs/opt_sdtps_loss1.0.log 2>&1 &

# ============================================================================
# 实验 2: SDTPS only + SPARSE=0.75 + LOSS=1.0
# 预期: 探索 0.7-0.8 之间的最优稀疏度
# ============================================================================
echo "[实验 2] SDTPS only + SPARSE=0.75 + LOSS=1.0"
CUDA_VISIBLE_DEVICES=1 python train_net.py --config_file $CONFIG_BASE \
    --exp_name "opt_sdtps_sparse0.75_loss1.0" \
    MODEL.USE_SACR False MODEL.USE_SDTPS True MODEL.USE_LIF False \
    MODEL.SDTPS_SPARSE_RATIO 0.75 MODEL.SDTPS_LOSS_WEIGHT 1.0 \
    > logs/opt_sdtps_sparse0.75_loss1.0.log 2>&1 &

# ============================================================================
# 实验 3: SDTPS only + SPARSE=0.8 + LOSS=1.0
# 预期: 结合最佳稀疏度和最佳损失权重
# ============================================================================
echo "[实验 3] SDTPS only + SPARSE=0.8 + LOSS=1.0"
CUDA_VISIBLE_DEVICES=2 python train_net.py --config_file $CONFIG_BASE \
    --exp_name "opt_sdtps_sparse0.8_loss1.0" \
    MODEL.USE_SACR False MODEL.USE_SDTPS True MODEL.USE_LIF False \
    MODEL.SDTPS_SPARSE_RATIO 0.8 MODEL.SDTPS_LOSS_WEIGHT 1.0 \
    > logs/opt_sdtps_sparse0.8_loss1.0.log 2>&1 &

# ============================================================================
# 实验 4: SDTPS only + BETA=0.3 + LOSS=1.0
# 预期: 增加跨模态注意力权重
# ============================================================================
echo "[实验 4] SDTPS only + BETA=0.3 + LOSS=1.0"
CUDA_VISIBLE_DEVICES=3 python train_net.py --config_file $CONFIG_BASE \
    --exp_name "opt_sdtps_beta0.3_loss1.0" \
    MODEL.USE_SACR False MODEL.USE_SDTPS True MODEL.USE_LIF False \
    MODEL.SDTPS_BETA 0.3 MODEL.SDTPS_LOSS_WEIGHT 1.0 \
    > logs/opt_sdtps_beta0.3_loss1.0.log 2>&1 &

echo "========== 4 个实验已启动 =========="
echo "监控命令:"
echo "  nvidia-smi"
echo "  tail -f logs/opt_*.log"

wait
echo "========== 所有实验完成 =========="
