#!/bin/bash
# Ablation Experiments for Vehicle Re-ID: MSVR310
# 4 GPUs run MSVR310 in parallel

# Config file
CONFIG_MSVR310="configs/MSVR310/DeMo_SDTPS_DGAF_ablation.yml"

# Get timestamp for folder name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create experiment folder
EXP_DIR_310="logs/MSVR310_ablation_SDTPS_DGAF_${TIMESTAMP}"
mkdir -p ${EXP_DIR_310}

echo "=============================================="
echo "Vehicle Re-ID Ablation Experiments - MSVR310"
echo "=============================================="
echo "MSVR310 logs: ${EXP_DIR_310}"
echo "=============================================="

# ============================================================================
# MSVR310 (4 GPUs in parallel)
# ============================================================================
echo ""
echo "Starting MSVR310 experiments on 4 GPUs..."

CUDA_VISIBLE_DEVICES=0 nohup python train_net.py --config_file ${CONFIG_MSVR310} \
    --exp_name "ablation_baseline" \
    MODEL.USE_SDTPS False MODEL.USE_DGAF False \
    > ${EXP_DIR_310}/baseline.log 2>&1 &
echo "  GPU 0: Baseline, PID: $!"

CUDA_VISIBLE_DEVICES=1 nohup python train_net.py --config_file ${CONFIG_MSVR310} \
    --exp_name "ablation_SDTPS_only" \
    MODEL.USE_SDTPS True MODEL.USE_DGAF False \
    > ${EXP_DIR_310}/SDTPS_only.log 2>&1 &
echo "  GPU 1: SDTPS only, PID: $!"

CUDA_VISIBLE_DEVICES=2 nohup python train_net.py --config_file ${CONFIG_MSVR310} \
    --exp_name "ablation_DGAFv3_only" \
    MODEL.USE_SDTPS False MODEL.USE_DGAF True \
    > ${EXP_DIR_310}/DGAFv3_only.log 2>&1 &
echo "  GPU 2: DGAF V3 only, PID: $!"

CUDA_VISIBLE_DEVICES=3 nohup python train_net.py --config_file ${CONFIG_MSVR310} \
    --exp_name "ablation_SDTPS_DGAFv3" \
    MODEL.USE_SDTPS True MODEL.USE_DGAF True \
    > ${EXP_DIR_310}/SDTPS_DGAFv3.log 2>&1 &
echo "  GPU 3: SDTPS + DGAF V3, PID: $!"

echo ""
echo "All MSVR310 experiments started!"
echo "Use 'tail -f ${EXP_DIR_310}/*.log' to monitor progress"
echo "=============================================="
