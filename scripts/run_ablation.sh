#!/bin/bash
# Ablation Experiments: Testing SDTPS and DGAF effectiveness
# Run 4 experiments on GPUs 0-3 in parallel
#
# Experiment setup:
#   GPU 0: Baseline (nothing added)
#   GPU 1: SDTPS only
#   GPU 2: DGAF V3 only (standalone)
#   GPU 3: SDTPS + DGAF V3

# Create logs directory if not exists
mkdir -p logs

echo "=============================================="
echo "Starting Ablation Experiments"
echo "=============================================="
echo "GPU 0: Baseline"
echo "GPU 1: SDTPS only"
echo "GPU 2: DGAF V3 only"
echo "GPU 3: SDTPS + DGAF V3"
echo "=============================================="

# Get timestamp for log files
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Experiment 1: Baseline on GPU 0
CUDA_VISIBLE_DEVICES=0 nohup python train_net.py \
    --config_file configs/RGBNT201/ablation_baseline.yml \
    --exp_name "RGBNT201_ablation_baseline_${TIMESTAMP}" \
    > logs/RGBNT201_ablation_baseline_${TIMESTAMP}.log 2>&1 &
echo "Started Experiment 1 (Baseline) on GPU 0, PID: $!"

# Experiment 2: SDTPS only on GPU 1
CUDA_VISIBLE_DEVICES=1 nohup python train_net.py \
    --config_file configs/RGBNT201/ablation_sdtps_only.yml \
    --exp_name "RGBNT201_ablation_SDTPS_only_${TIMESTAMP}" \
    > logs/RGBNT201_ablation_SDTPS_only_${TIMESTAMP}.log 2>&1 &
echo "Started Experiment 2 (SDTPS only) on GPU 1, PID: $!"

# Experiment 3: DGAF V3 only on GPU 2
CUDA_VISIBLE_DEVICES=2 nohup python train_net.py \
    --config_file configs/RGBNT201/ablation_dgaf_only.yml \
    --exp_name "RGBNT201_ablation_DGAFv3_only_${TIMESTAMP}" \
    > logs/RGBNT201_ablation_DGAFv3_only_${TIMESTAMP}.log 2>&1 &
echo "Started Experiment 3 (DGAF V3 only) on GPU 2, PID: $!"

# Experiment 4: SDTPS + DGAF V3 on GPU 3
CUDA_VISIBLE_DEVICES=3 nohup python train_net.py \
    --config_file configs/RGBNT201/ablation_sdtps_dgaf.yml \
    --exp_name "RGBNT201_ablation_SDTPS_DGAFv3_${TIMESTAMP}" \
    > logs/RGBNT201_ablation_SDTPS_DGAFv3_${TIMESTAMP}.log 2>&1 &
echo "Started Experiment 4 (SDTPS + DGAF V3) on GPU 3, PID: $!"

echo ""
echo "=============================================="
echo "All experiments started!"
echo "=============================================="
echo ""
echo "Monitor progress with:"
echo "  tail -f logs/RGBNT201_ablation_baseline_${TIMESTAMP}.log"
echo "  tail -f logs/RGBNT201_ablation_SDTPS_only_${TIMESTAMP}.log"
echo "  tail -f logs/RGBNT201_ablation_DGAFv3_only_${TIMESTAMP}.log"
echo "  tail -f logs/RGBNT201_ablation_SDTPS_DGAFv3_${TIMESTAMP}.log"
