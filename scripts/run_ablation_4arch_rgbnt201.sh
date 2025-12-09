#!/bin/bash
# ============================================================================
# 4种架构消融实验 - 仅 RGBNT201 数据集
# 4种配置在4个GPU上并行执行
# ============================================================================
#
# 4种配置:
# 1. Baseline:        无 SDTPS, 无 DGAF, 无 GLOBAL_LOCAL → 只用 ori 损失
# 2. SDTPS only:      有 SDTPS, 无 DGAF, 无 GLOBAL_LOCAL → 只用 sdtps 损失
# 3. DGAF V3 only:    无 SDTPS, 有 DGAF V3, 无 GLOBAL_LOCAL → 只用 dgaf 损失
# 4. SDTPS+DGAF V1:   有 SDTPS, 有 DGAF V1, 有 GLOBAL_LOCAL → sdtps 损失 + dgaf 损失
#
# ============================================================================

# Get timestamp for folder name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create experiment folder
EXP_DIR_201="logs/RGBNT201_4arch_ablation_${TIMESTAMP}"
mkdir -p ${EXP_DIR_201}

echo "=============================================================================="
echo "4种架构消融实验 - RGBNT201"
echo "=============================================================================="
echo "配置1: Baseline        - 只用 ori 损失"
echo "配置2: SDTPS only      - 只用 sdtps 损失"
echo "配置3: DGAF V3 only    - 只用 dgaf 损失"
echo "配置4: SDTPS+DGAF V1   - sdtps 损失 + dgaf 损失 (需要 GLOBAL_LOCAL)"
echo "=============================================================================="
echo "RGBNT201 logs: ${EXP_DIR_201}"
echo "=============================================================================="

# ============================================================================
# RGBNT201 实验 (4 GPUs in parallel)
# ============================================================================
echo ""
echo "Starting RGBNT201 experiments on 4 GPUs..."

# GPU 0: Baseline
CUDA_VISIBLE_DEVICES=0 nohup python train_net.py --config_file configs/RGBNT201/DeMo_SDTPS_DGAF_ablation.yml MODEL.USE_SDTPS False MODEL.USE_DGAF False MODEL.GLOBAL_LOCAL False > ${EXP_DIR_201}/01_baseline.log 2>&1 &
PID_201_1=$!
echo "  GPU 0: Baseline, PID: ${PID_201_1}"

# GPU 1: SDTPS only (attention, no DGAF, no GLOBAL_LOCAL)
CUDA_VISIBLE_DEVICES=1 nohup python train_net.py --config_file configs/RGBNT201/DeMo_SDTPS_DGAF_ablation.yml MODEL.USE_SDTPS True MODEL.USE_DGAF False MODEL.GLOBAL_LOCAL False MODEL.SDTPS_CROSS_ATTN_TYPE attention MODEL.SDTPS_CROSS_ATTN_HEADS 4 > ${EXP_DIR_201}/02_sdtps_only.log 2>&1 &
PID_201_2=$!
echo "  GPU 1: SDTPS only, PID: ${PID_201_2}"

# GPU 2: DGAF V3 only
CUDA_VISIBLE_DEVICES=2 nohup python train_net.py --config_file configs/RGBNT201/DeMo_SDTPS_DGAF_ablation.yml MODEL.USE_SDTPS False MODEL.USE_DGAF True MODEL.DGAF_VERSION v3 MODEL.GLOBAL_LOCAL False > ${EXP_DIR_201}/03_dgaf_v3_only.log 2>&1 &
PID_201_3=$!
echo "  GPU 2: DGAF V3 only, PID: ${PID_201_3}"

# GPU 3: SDTPS + DGAF V1 (必须用 V1 + GLOBAL_LOCAL)
CUDA_VISIBLE_DEVICES=3 nohup python train_net.py --config_file configs/RGBNT201/DeMo_SDTPS_DGAF_ablation.yml MODEL.USE_SDTPS True MODEL.USE_DGAF True MODEL.DGAF_VERSION v1 MODEL.GLOBAL_LOCAL True MODEL.SDTPS_CROSS_ATTN_TYPE attention MODEL.SDTPS_CROSS_ATTN_HEADS 4 > ${EXP_DIR_201}/04_sdtps_dgaf_v1_gl.log 2>&1 &
PID_201_4=$!
echo "  GPU 3: SDTPS + DGAF V1 (GLOBAL_LOCAL), PID: ${PID_201_4}"

echo ""
echo "Waiting for all experiments to complete..."
wait ${PID_201_1} ${PID_201_2} ${PID_201_3} ${PID_201_4}
echo "RGBNT201 experiments completed!"

echo ""
echo "=============================================================================="
echo "实验完成!"
echo "=============================================================================="
echo "结果位置: ${EXP_DIR_201}"
echo "=============================================================================="
echo ""
echo "实验配置总结:"
echo "  1. Baseline:        只有 backbone, 只用 ori 损失"
echo "  2. SDTPS only:      SDTPS(attention), 只用 sdtps 损失"
echo "  3. DGAF V3 only:    DGAF V3, 只用 dgaf 损失"
echo "  4. SDTPS+DGAF V1:   SDTPS + DGAF V1 + GLOBAL_LOCAL, sdtps损失 + dgaf损失"
echo "=============================================================================="
