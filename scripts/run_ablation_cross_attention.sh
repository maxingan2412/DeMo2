#!/bin/bash
# ============================================================================
# Cross-Attention Ablation Experiments
# 测试 SDTPS_CROSS_ATTN_TYPE='attention' 模式
# 三个数据集 x 4个消融实验 = 12个实验
# 分三轮执行，每轮4个GPU并行
# ============================================================================

# Config files
CONFIG_RGBNT201="configs/RGBNT201/DeMo_SDTPS_DGAF_ablation.yml"
CONFIG_RGBNT100="configs/RGBNT100/DeMo_SDTPS_DGAF_ablation.yml"
CONFIG_MSVR310="configs/MSVR310/DeMo_SDTPS_DGAF_ablation.yml"

# Get timestamp for folder name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create experiment folders
EXP_DIR_201="logs/RGBNT201_cross_attn_${TIMESTAMP}"
EXP_DIR_100="logs/RGBNT100_cross_attn_${TIMESTAMP}"
EXP_DIR_310="logs/MSVR310_cross_attn_${TIMESTAMP}"
mkdir -p ${EXP_DIR_201}
mkdir -p ${EXP_DIR_100}
mkdir -p ${EXP_DIR_310}

echo "=============================================================================="
echo "Cross-Attention Ablation Experiments (SDTPS_CROSS_ATTN_TYPE='attention')"
echo "=============================================================================="
echo "RGBNT201 logs: ${EXP_DIR_201}"
echo "RGBNT100 logs: ${EXP_DIR_100}"
echo "MSVR310 logs:  ${EXP_DIR_310}"
echo "=============================================================================="

# ============================================================================
# Phase 1: RGBNT201 (4 GPUs in parallel)
# ============================================================================
echo ""
echo "[Phase 1/3] Starting RGBNT201 experiments on 4 GPUs..."

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
echo "Waiting for RGBNT201 to complete..."
wait
echo "RGBNT201 completed!"

# ============================================================================
# Phase 2: RGBNT100 (4 GPUs in parallel)
# ============================================================================
echo ""
echo "[Phase 2/3] Starting RGBNT100 experiments on 4 GPUs..."

# GPU 0: Baseline
CUDA_VISIBLE_DEVICES=0 nohup python train_net.py --config_file ${CONFIG_RGBNT100} \
    --exp_name "cross_attn_baseline" \
    MODEL.USE_SDTPS False MODEL.USE_DGAF False \
    > ${EXP_DIR_100}/baseline.log 2>&1 &
echo "  GPU 0: Baseline, PID: $!"

# GPU 1: SDTPS (attention) only
CUDA_VISIBLE_DEVICES=1 nohup python train_net.py --config_file ${CONFIG_RGBNT100} \
    --exp_name "cross_attn_SDTPS_only" \
    MODEL.USE_SDTPS True MODEL.USE_DGAF False \
    MODEL.SDTPS_CROSS_ATTN_TYPE attention MODEL.SDTPS_CROSS_ATTN_HEADS 4 \
    > ${EXP_DIR_100}/SDTPS_attn_only.log 2>&1 &
echo "  GPU 1: SDTPS (attention) only, PID: $!"

# GPU 2: DGAF V3 only
CUDA_VISIBLE_DEVICES=2 nohup python train_net.py --config_file ${CONFIG_RGBNT100} \
    --exp_name "cross_attn_DGAFv3_only" \
    MODEL.USE_SDTPS False MODEL.USE_DGAF True \
    > ${EXP_DIR_100}/DGAFv3_only.log 2>&1 &
echo "  GPU 2: DGAF V3 only, PID: $!"

# GPU 3: SDTPS (attention) + DGAF V3
CUDA_VISIBLE_DEVICES=3 nohup python train_net.py --config_file ${CONFIG_RGBNT100} \
    --exp_name "cross_attn_SDTPS_DGAFv3" \
    MODEL.USE_SDTPS True MODEL.USE_DGAF True \
    MODEL.SDTPS_CROSS_ATTN_TYPE attention MODEL.SDTPS_CROSS_ATTN_HEADS 4 \
    > ${EXP_DIR_100}/SDTPS_attn_DGAFv3.log 2>&1 &
echo "  GPU 3: SDTPS (attention) + DGAF V3, PID: $!"

echo ""
echo "Waiting for RGBNT100 to complete..."
wait
echo "RGBNT100 completed!"

# ============================================================================
# Phase 3: MSVR310 (4 GPUs in parallel)
# ============================================================================
echo ""
echo "[Phase 3/3] Starting MSVR310 experiments on 4 GPUs..."

# GPU 0: Baseline
CUDA_VISIBLE_DEVICES=0 nohup python train_net.py --config_file ${CONFIG_MSVR310} \
    --exp_name "cross_attn_baseline" \
    MODEL.USE_SDTPS False MODEL.USE_DGAF False \
    > ${EXP_DIR_310}/baseline.log 2>&1 &
echo "  GPU 0: Baseline, PID: $!"

# GPU 1: SDTPS (attention) only
CUDA_VISIBLE_DEVICES=1 nohup python train_net.py --config_file ${CONFIG_MSVR310} \
    --exp_name "cross_attn_SDTPS_only" \
    MODEL.USE_SDTPS True MODEL.USE_DGAF False \
    MODEL.SDTPS_CROSS_ATTN_TYPE attention MODEL.SDTPS_CROSS_ATTN_HEADS 4 \
    > ${EXP_DIR_310}/SDTPS_attn_only.log 2>&1 &
echo "  GPU 1: SDTPS (attention) only, PID: $!"

# GPU 2: DGAF V3 only
CUDA_VISIBLE_DEVICES=2 nohup python train_net.py --config_file ${CONFIG_MSVR310} \
    --exp_name "cross_attn_DGAFv3_only" \
    MODEL.USE_SDTPS False MODEL.USE_DGAF True \
    > ${EXP_DIR_310}/DGAFv3_only.log 2>&1 &
echo "  GPU 2: DGAF V3 only, PID: $!"

# GPU 3: SDTPS (attention) + DGAF V3
CUDA_VISIBLE_DEVICES=3 nohup python train_net.py --config_file ${CONFIG_MSVR310} \
    --exp_name "cross_attn_SDTPS_DGAFv3" \
    MODEL.USE_SDTPS True MODEL.USE_DGAF True \
    MODEL.SDTPS_CROSS_ATTN_TYPE attention MODEL.SDTPS_CROSS_ATTN_HEADS 4 \
    > ${EXP_DIR_310}/SDTPS_attn_DGAFv3.log 2>&1 &
echo "  GPU 3: SDTPS (attention) + DGAF V3, PID: $!"

echo ""
echo "Waiting for MSVR310 to complete..."
wait

echo ""
echo "=============================================================================="
echo "All Cross-Attention ablation experiments completed!"
echo "=============================================================================="
echo "Results:"
echo "  RGBNT201: ${EXP_DIR_201}"
echo "  RGBNT100: ${EXP_DIR_100}"
echo "  MSVR310:  ${EXP_DIR_310}"
echo "=============================================================================="
