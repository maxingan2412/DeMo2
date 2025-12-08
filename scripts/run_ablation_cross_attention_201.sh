#!/bin/bash
# ============================================================================
# Cross-Attention Ablation Experiments - RGBNT201 Only
# 测试 SDTPS_CROSS_ATTN_TYPE='attention' 模式
# 4个消融实验，4个GPU并行
# ============================================================================

# Config file
CONFIG_RGBNT201="configs/RGBNT201/DeMo_SDTPS_DGAF_ablation.yml"

# Get timestamp for folder name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create experiment folder
EXP_DIR_201="logs/RGBNT201_cross_attn_${TIMESTAMP}"
mkdir -p ${EXP_DIR_201}

echo "=============================================================================="
echo "Cross-Attention Ablation Experiments - RGBNT201"
echo "(SDTPS_CROSS_ATTN_TYPE='attention')"
echo "=============================================================================="
echo "RGBNT201 logs: ${EXP_DIR_201}"
echo "=============================================================================="

# ============================================================================
# RGBNT201 (4 GPUs in parallel)
# ============================================================================
echo ""
echo "Starting RGBNT201 experiments on 4 GPUs..."

# GPU 0: Baseline (no SDTPS, no DGAF)
CUDA_VISIBLE_DEVICES=0 nohup python train_net.py --config_file ${CONFIG_RGBNT201} \
    --exp_name "cross_attn_baseline" \
    MODEL.USE_SDTPS False MODEL.USE_DGAF False \
    > ${EXP_DIR_201}/baseline.log 2>&1 &
echo "  GPU 0: Baseline, PID: $!"

# GPU 1: SDTPS only (with Cross-Attention)
CUDA_VISIBLE_DEVICES=1 nohup python train_net.py --config_file ${CONFIG_RGBNT201} \
    --exp_name "cross_attn_SDTPS_only" \
    MODEL.USE_SDTPS True MODEL.USE_DGAF False \
    MODEL.SDTPS_CROSS_ATTN_TYPE attention MODEL.SDTPS_CROSS_ATTN_HEADS 4 \
    > ${EXP_DIR_201}/SDTPS_attn_only.log 2>&1 &
echo "  GPU 1: SDTPS (attention) only, PID: $!"

# GPU 2: DGAF V3 only
CUDA_VISIBLE_DEVICES=2 nohup python train_net.py --config_file ${CONFIG_RGBNT201} \
    --exp_name "cross_attn_DGAFv3_only" \
    MODEL.USE_SDTPS False MODEL.USE_DGAF True \
    > ${EXP_DIR_201}/DGAFv3_only.log 2>&1 &
echo "  GPU 2: DGAF V3 only, PID: $!"

# GPU 3: SDTPS (attention) + DGAF V3
CUDA_VISIBLE_DEVICES=3 nohup python train_net.py --config_file ${CONFIG_RGBNT201} \
    --exp_name "cross_attn_SDTPS_DGAFv3" \
    MODEL.USE_SDTPS True MODEL.USE_DGAF True \
    MODEL.SDTPS_CROSS_ATTN_TYPE attention MODEL.SDTPS_CROSS_ATTN_HEADS 4 \
    > ${EXP_DIR_201}/SDTPS_attn_DGAFv3.log 2>&1 &
echo "  GPU 3: SDTPS (attention) + DGAF V3, PID: $!"

echo ""
echo "All RGBNT201 experiments started!"
echo "Use 'tail -f ${EXP_DIR_201}/*.log' to monitor progress"
echo "=============================================================================="
