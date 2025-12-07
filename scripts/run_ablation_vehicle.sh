#!/bin/bash
# Ablation Experiments for Vehicle Re-ID: RGBNT100 and MSVR310
# Run 8 experiments on GPUs 0-3 in parallel
# Each GPU runs 2 experiments sequentially (RGBNT100 first, then MSVR310)
#
# RGBNT100: lr=0.00035, epochs=30, warmup=5
# MSVR310:  lr=0.00035, epochs=50, warmup=10
# TF32 enabled, Cosine scheduler

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
echo "Starting Vehicle Re-ID Ablation Experiments"
echo "=============================================="
echo "RGBNT100 config: ${CONFIG_RGBNT100}"
echo "MSVR310 config:  ${CONFIG_MSVR310}"
echo ""
echo "RGBNT100 logs: ${EXP_DIR_100}"
echo "MSVR310 logs:  ${EXP_DIR_310}"
echo ""
echo "GPU 0: Baseline (RGBNT100 -> MSVR310)"
echo "GPU 1: SDTPS only (RGBNT100 -> MSVR310)"
echo "GPU 2: DGAF V3 only (RGBNT100 -> MSVR310)"
echo "GPU 3: SDTPS + DGAF V3 (RGBNT100 -> MSVR310)"
echo "=============================================="

# ============================================================================
# GPU 0: Baseline experiments (no SDTPS, no DGAF)
# Run RGBNT100 first, then MSVR310 sequentially on the same GPU
# ============================================================================
echo "[GPU 0] Starting Baseline experiments..."
CUDA_VISIBLE_DEVICES=0 nohup bash -c "
    python train_net.py --config_file ${CONFIG_RGBNT100} \
        --exp_name 'ablation_baseline' \
        MODEL.USE_SDTPS False \
        MODEL.USE_DGAF False \
        > ${EXP_DIR_100}/baseline.log 2>&1 && \
    python train_net.py --config_file ${CONFIG_MSVR310} \
        --exp_name 'ablation_baseline' \
        MODEL.USE_SDTPS False \
        MODEL.USE_DGAF False \
        > ${EXP_DIR_310}/baseline.log 2>&1
" &
echo "Started on GPU 0, PID: $!"

# ============================================================================
# GPU 1: SDTPS only experiments
# ============================================================================
echo "[GPU 1] Starting SDTPS only experiments..."
CUDA_VISIBLE_DEVICES=1 nohup bash -c "
    python train_net.py --config_file ${CONFIG_RGBNT100} \
        --exp_name 'ablation_SDTPS_only' \
        MODEL.USE_SDTPS True \
        MODEL.USE_DGAF False \
        > ${EXP_DIR_100}/SDTPS_only.log 2>&1 && \
    python train_net.py --config_file ${CONFIG_MSVR310} \
        --exp_name 'ablation_SDTPS_only' \
        MODEL.USE_SDTPS True \
        MODEL.USE_DGAF False \
        > ${EXP_DIR_310}/SDTPS_only.log 2>&1
" &
echo "Started on GPU 1, PID: $!"

# ============================================================================
# GPU 2: DGAF V3 only experiments
# ============================================================================
echo "[GPU 2] Starting DGAF V3 only experiments..."
CUDA_VISIBLE_DEVICES=2 nohup bash -c "
    python train_net.py --config_file ${CONFIG_RGBNT100} \
        --exp_name 'ablation_DGAFv3_only' \
        MODEL.USE_SDTPS False \
        MODEL.USE_DGAF True \
        > ${EXP_DIR_100}/DGAFv3_only.log 2>&1 && \
    python train_net.py --config_file ${CONFIG_MSVR310} \
        --exp_name 'ablation_DGAFv3_only' \
        MODEL.USE_SDTPS False \
        MODEL.USE_DGAF True \
        > ${EXP_DIR_310}/DGAFv3_only.log 2>&1
" &
echo "Started on GPU 2, PID: $!"

# ============================================================================
# GPU 3: SDTPS + DGAF V3 experiments
# ============================================================================
echo "[GPU 3] Starting SDTPS + DGAF V3 experiments..."
CUDA_VISIBLE_DEVICES=3 nohup bash -c "
    python train_net.py --config_file ${CONFIG_RGBNT100} \
        --exp_name 'ablation_SDTPS_DGAFv3' \
        MODEL.USE_SDTPS True \
        MODEL.USE_DGAF True \
        > ${EXP_DIR_100}/SDTPS_DGAFv3.log 2>&1 && \
    python train_net.py --config_file ${CONFIG_MSVR310} \
        --exp_name 'ablation_SDTPS_DGAFv3' \
        MODEL.USE_SDTPS True \
        MODEL.USE_DGAF True \
        > ${EXP_DIR_310}/SDTPS_DGAFv3.log 2>&1
" &
echo "Started on GPU 3, PID: $!"

echo ""
echo "=============================================="
echo "All experiments started in background!"
echo "=============================================="
echo ""
echo "Monitor RGBNT100 progress:"
echo "  tail -f ${EXP_DIR_100}/baseline.log"
echo "  tail -f ${EXP_DIR_100}/SDTPS_only.log"
echo "  tail -f ${EXP_DIR_100}/DGAFv3_only.log"
echo "  tail -f ${EXP_DIR_100}/SDTPS_DGAFv3.log"
echo ""
echo "Monitor MSVR310 progress (after RGBNT100 finishes):"
echo "  tail -f ${EXP_DIR_310}/baseline.log"
echo "  tail -f ${EXP_DIR_310}/SDTPS_only.log"
echo "  tail -f ${EXP_DIR_310}/DGAFv3_only.log"
echo "  tail -f ${EXP_DIR_310}/SDTPS_DGAFv3.log"
echo ""
echo "Check GPU usage: watch -n 1 nvidia-smi"
