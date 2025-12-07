#!/bin/bash
# Ablation Experiments: Testing SDTPS and DGAF effectiveness
# Run 4 experiments on GPUs 0-3 in parallel
#
# Base config: DeMo_SDTPS_DGAF_ablation.yml (SACR disabled)
# Training settings: lr=0.00035, epochs=50, warmup=10, TF32 enabled

# Base config file
CONFIG_BASE="configs/RGBNT201/DeMo_SDTPS_DGAF_ablation.yml"

# Get timestamp for folder name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create experiment folder
EXP_DIR="logs/RGBNT201_ablation_SDTPS_DGAF_normalsetting_cosine_${TIMESTAMP}"
mkdir -p ${EXP_DIR}

echo "=============================================="
echo "Starting Ablation Experiments"
echo "=============================================="
echo "Base config: ${CONFIG_BASE}"
echo "Experiment folder: ${EXP_DIR}"
echo ""
echo "GPU 0: Baseline (no SDTPS, no DGAF)"
echo "GPU 1: SDTPS only"
echo "GPU 2: DGAF V3 only"
echo "GPU 3: SDTPS + DGAF V3"
echo "=============================================="

# Experiment 1: Baseline on GPU 0 (no SDTPS, no DGAF)
echo "[Experiment 1] Baseline"
CUDA_VISIBLE_DEVICES=0 nohup python train_net.py --config_file ${CONFIG_BASE} \
    --exp_name "ablation_baseline" \
    MODEL.USE_SDTPS False \
    MODEL.USE_DGAF False \
    > ${EXP_DIR}/baseline.log 2>&1 &
echo "Started on GPU 0, PID: $!"

# Experiment 2: SDTPS only on GPU 1
echo "[Experiment 2] SDTPS only"
CUDA_VISIBLE_DEVICES=1 nohup python train_net.py --config_file ${CONFIG_BASE} \
    --exp_name "ablation_SDTPS_only" \
    MODEL.USE_SDTPS True \
    MODEL.USE_DGAF False \
    > ${EXP_DIR}/SDTPS_only.log 2>&1 &
echo "Started on GPU 1, PID: $!"

# Experiment 3: DGAF V3 only on GPU 2 (standalone, no SDTPS)
echo "[Experiment 3] DGAF V3 only"
CUDA_VISIBLE_DEVICES=2 nohup python train_net.py --config_file ${CONFIG_BASE} \
    --exp_name "ablation_DGAFv3_only" \
    MODEL.USE_SDTPS False \
    MODEL.USE_DGAF True \
    > ${EXP_DIR}/DGAFv3_only.log 2>&1 &
echo "Started on GPU 2, PID: $!"

# Experiment 4: SDTPS + DGAF V3 on GPU 3
echo "[Experiment 4] SDTPS + DGAF V3"
CUDA_VISIBLE_DEVICES=3 nohup python train_net.py --config_file ${CONFIG_BASE} \
    --exp_name "ablation_SDTPS_DGAFv3" \
    MODEL.USE_SDTPS True \
    MODEL.USE_DGAF True \
    > ${EXP_DIR}/SDTPS_DGAFv3.log 2>&1 &
echo "Started on GPU 3, PID: $!"

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
