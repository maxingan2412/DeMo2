#!/bin/bash
# Ablation Experiments: Testing SDTPS and DGAF effectiveness
# Run 4 experiments on GPUs 0-3 in parallel
#
# Experiment setup:
#   GPU 0: Baseline (nothing added)
#   GPU 1: SDTPS only
#   GPU 2: DGAF V3 only (standalone)
#   GPU 3: SDTPS + DGAF V3

# Get timestamp for folder name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create experiment folder
EXP_DIR="logs/RGBNT201_ablation_SDTPS_DGAF_${TIMESTAMP}"
mkdir -p ${EXP_DIR}

echo "=============================================="
echo "Starting Ablation Experiments"
echo "=============================================="
echo "Experiment folder: ${EXP_DIR}"
echo ""
echo "GPU 0: Baseline"
echo "GPU 1: SDTPS only"
echo "GPU 2: DGAF V3 only"
echo "GPU 3: SDTPS + DGAF V3"
echo "=============================================="

# Experiment 1: Baseline on GPU 0
CUDA_VISIBLE_DEVICES=0 nohup python train_net.py \
    --config_file configs/RGBNT201/ablation_baseline.yml \
    --exp_name "ablation_baseline" \
    > ${EXP_DIR}/baseline.log 2>&1 &
echo "Started Experiment 1 (Baseline) on GPU 0, PID: $!"

# Experiment 2: SDTPS only on GPU 1
CUDA_VISIBLE_DEVICES=1 nohup python train_net.py \
    --config_file configs/RGBNT201/ablation_sdtps_only.yml \
    --exp_name "ablation_SDTPS_only" \
    > ${EXP_DIR}/SDTPS_only.log 2>&1 &
echo "Started Experiment 2 (SDTPS only) on GPU 1, PID: $!"

# Experiment 3: DGAF V3 only on GPU 2
CUDA_VISIBLE_DEVICES=2 nohup python train_net.py \
    --config_file configs/RGBNT201/ablation_dgaf_only.yml \
    --exp_name "ablation_DGAFv3_only" \
    > ${EXP_DIR}/DGAFv3_only.log 2>&1 &
echo "Started Experiment 3 (DGAF V3 only) on GPU 2, PID: $!"

# Experiment 4: SDTPS + DGAF V3 on GPU 3
CUDA_VISIBLE_DEVICES=3 nohup python train_net.py \
    --config_file configs/RGBNT201/ablation_sdtps_dgaf.yml \
    --exp_name "ablation_SDTPS_DGAFv3" \
    > ${EXP_DIR}/SDTPS_DGAFv3.log 2>&1 &
echo "Started Experiment 4 (SDTPS + DGAF V3) on GPU 3, PID: $!"

echo ""
echo "=============================================="
echo "All experiments started!"
echo "=============================================="
echo ""
echo "Monitor progress with:"
echo "  tail -f ${EXP_DIR}/baseline.log"
echo "  tail -f ${EXP_DIR}/SDTPS_only.log"
echo "  tail -f ${EXP_DIR}/DGAFv3_only.log"
echo "  tail -f ${EXP_DIR}/SDTPS_DGAFv3.log"
