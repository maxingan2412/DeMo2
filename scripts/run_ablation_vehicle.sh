#!/bin/bash
# Ablation Experiments for Vehicle Re-ID: RGBNT100 and MSVR310
# Phase 1: 4 GPUs run RGBNT100 in parallel
# Phase 2: 4 GPUs run MSVR310 in parallel (after Phase 1 completes)

# Config files
CONFIG_RGBNT100="configs/RGBNT100/DeMo_SDTPS_DGAF_ablation.yml"
CONFIG_MSVR310="configs/MSVR310/DeMo_SDTPS_DGAF_ablation.yml"

# Get timestamp for folder name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create experiment folders
EXP_DIR_100="logs/RGBNT100_ablation_SDTPS_DGAF_${TIMESTAMP}"
EXP_DIR_310="logs/MSVR310_ablation_SDTPS_DGAF_${TIMESTAMP}"
mkdir -p ${EXP_DIR_100}
mkdir -p ${EXP_DIR_310}

echo "=============================================="
echo "Vehicle Re-ID Ablation Experiments"
echo "=============================================="
echo "RGBNT100 logs: ${EXP_DIR_100}"
echo "MSVR310 logs:  ${EXP_DIR_310}"
echo "=============================================="

# ============================================================================
# Phase 1: RGBNT100 (4 GPUs in parallel)
# ============================================================================
echo ""
echo "[Phase 1] Starting RGBNT100 experiments on 4 GPUs..."

CUDA_VISIBLE_DEVICES=0 nohup python train_net.py --config_file ${CONFIG_RGBNT100} \
    --exp_name "ablation_baseline" \
    MODEL.USE_SDTPS False MODEL.USE_DGAF False \
    > ${EXP_DIR_100}/baseline.log 2>&1 &
echo "  GPU 0: Baseline, PID: $!"

CUDA_VISIBLE_DEVICES=1 nohup python train_net.py --config_file ${CONFIG_RGBNT100} \
    --exp_name "ablation_SDTPS_only" \
    MODEL.USE_SDTPS True MODEL.USE_DGAF False \
    > ${EXP_DIR_100}/SDTPS_only.log 2>&1 &
echo "  GPU 1: SDTPS only, PID: $!"

CUDA_VISIBLE_DEVICES=2 nohup python train_net.py --config_file ${CONFIG_RGBNT100} \
    --exp_name "ablation_DGAFv3_only" \
    MODEL.USE_SDTPS False MODEL.USE_DGAF True \
    > ${EXP_DIR_100}/DGAFv3_only.log 2>&1 &
echo "  GPU 2: DGAF V3 only, PID: $!"

CUDA_VISIBLE_DEVICES=3 nohup python train_net.py --config_file ${CONFIG_RGBNT100} \
    --exp_name "ablation_SDTPS_DGAFv3" \
    MODEL.USE_SDTPS True MODEL.USE_DGAF True \
    > ${EXP_DIR_100}/SDTPS_DGAFv3.log 2>&1 &
echo "  GPU 3: SDTPS + DGAF V3, PID: $!"

echo ""
echo "Waiting for RGBNT100 to complete..."
wait
echo "RGBNT100 completed!"

# ============================================================================
# Phase 2: MSVR310 (4 GPUs in parallel)
# ============================================================================
echo ""
echo "[Phase 2] Starting MSVR310 experiments on 4 GPUs..."

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
echo "Waiting for MSVR310 to complete..."
wait

echo ""
echo "=============================================="
echo "All experiments completed!"
echo "=============================================="
