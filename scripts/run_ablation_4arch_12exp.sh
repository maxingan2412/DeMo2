#!/bin/bash
# ============================================================================
# 4种架构消融实验
# 3个数据集 × 4种配置 = 12个实验
# 分3轮执行，每轮4个GPU并行
# ============================================================================
#
# 4种配置:
# 1. Baseline: 无 SDTPS, 无 DGAF (只有 backbone)
# 2. SDTPS only: 只用 SDTPS
# 3. DGAF only: 只用 DGAF V3
# 4. SDTPS + DGAF V3: 两者都用
#
# ============================================================================

# Base config files (使用现有的 ablation 配置作为基础)
CONFIG_RGBNT201="configs/RGBNT201/DeMo_SDTPS_DGAF_ablation.yml"
CONFIG_RGBNT100="configs/RGBNT100/DeMo_SDTPS_DGAF_ablation.yml"
CONFIG_MSVR310="configs/MSVR310/DeMo_SDTPS_DGAF_ablation.yml"

# Get timestamp for folder name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create experiment folders
EXP_DIR_201="logs/RGBNT201_4arch_ablation_${TIMESTAMP}"
EXP_DIR_100="logs/RGBNT100_4arch_ablation_${TIMESTAMP}"
EXP_DIR_310="logs/MSVR310_4arch_ablation_${TIMESTAMP}"
mkdir -p ${EXP_DIR_201}
mkdir -p ${EXP_DIR_100}
mkdir -p ${EXP_DIR_310}

echo "=============================================================================="
echo "4种架构消融实验 (Baseline, SDTPS, DGAF, SDTPS+DGAF)"
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

# GPU 0: Baseline (无 SDTPS, 无 DGAF, 无 GLOBAL_LOCAL)
CUDA_VISIBLE_DEVICES=0 nohup python train_net.py --config_file ${CONFIG_RGBNT201} \
    MODEL.USE_SDTPS False \
    MODEL.USE_DGAF False \
    MODEL.GLOBAL_LOCAL False \
    > ${EXP_DIR_201}/01_baseline.log 2>&1 &
PID_201_1=$!
echo "  GPU 0: Baseline, PID: ${PID_201_1}"

# GPU 1: SDTPS only (不用 DGAF, 不用 GLOBAL_LOCAL)
CUDA_VISIBLE_DEVICES=1 nohup python train_net.py --config_file ${CONFIG_RGBNT201} \
    MODEL.USE_SDTPS True \
    MODEL.USE_DGAF False \
    MODEL.GLOBAL_LOCAL False \
    MODEL.SDTPS_CROSS_ATTN_TYPE attention \
    MODEL.SDTPS_CROSS_ATTN_HEADS 4 \
    > ${EXP_DIR_201}/02_sdtps_only.log 2>&1 &
PID_201_2=$!
echo "  GPU 1: SDTPS only, PID: ${PID_201_2}"

# GPU 2: DGAF V3 only (不用 SDTPS, 不用 GLOBAL_LOCAL)
CUDA_VISIBLE_DEVICES=2 nohup python train_net.py --config_file ${CONFIG_RGBNT201} \
    MODEL.USE_SDTPS False \
    MODEL.USE_DGAF True \
    MODEL.DGAF_VERSION v3 \
    MODEL.GLOBAL_LOCAL False \
    > ${EXP_DIR_201}/03_dgaf_v3_only.log 2>&1 &
PID_201_3=$!
echo "  GPU 2: DGAF V3 only, PID: ${PID_201_3}"

# GPU 3: SDTPS + DGAF V3 (两者都用, 不用 GLOBAL_LOCAL)
CUDA_VISIBLE_DEVICES=3 nohup python train_net.py --config_file ${CONFIG_RGBNT201} \
    MODEL.USE_SDTPS True \
    MODEL.USE_DGAF True \
    MODEL.DGAF_VERSION v3 \
    MODEL.GLOBAL_LOCAL False \
    MODEL.SDTPS_CROSS_ATTN_TYPE attention \
    MODEL.SDTPS_CROSS_ATTN_HEADS 4 \
    > ${EXP_DIR_201}/04_sdtps_dgaf_v3.log 2>&1 &
PID_201_4=$!
echo "  GPU 3: SDTPS + DGAF V3, PID: ${PID_201_4}"

echo ""
echo "Waiting for RGBNT201 to complete..."
wait ${PID_201_1} ${PID_201_2} ${PID_201_3} ${PID_201_4}
echo "RGBNT201 completed!"

# ============================================================================
# Phase 2: RGBNT100 (4 GPUs in parallel)
# ============================================================================
echo ""
echo "[Phase 2/3] Starting RGBNT100 experiments on 4 GPUs..."

# GPU 0: Baseline
CUDA_VISIBLE_DEVICES=0 nohup python train_net.py --config_file ${CONFIG_RGBNT100} \
    MODEL.USE_SDTPS False \
    MODEL.USE_DGAF False \
    MODEL.GLOBAL_LOCAL False \
    > ${EXP_DIR_100}/01_baseline.log 2>&1 &
PID_100_1=$!
echo "  GPU 0: Baseline, PID: ${PID_100_1}"

# GPU 1: SDTPS only
CUDA_VISIBLE_DEVICES=1 nohup python train_net.py --config_file ${CONFIG_RGBNT100} \
    MODEL.USE_SDTPS True \
    MODEL.USE_DGAF False \
    MODEL.GLOBAL_LOCAL False \
    MODEL.SDTPS_CROSS_ATTN_TYPE attention \
    MODEL.SDTPS_CROSS_ATTN_HEADS 4 \
    > ${EXP_DIR_100}/02_sdtps_only.log 2>&1 &
PID_100_2=$!
echo "  GPU 1: SDTPS only, PID: ${PID_100_2}"

# GPU 2: DGAF V3 only
CUDA_VISIBLE_DEVICES=2 nohup python train_net.py --config_file ${CONFIG_RGBNT100} \
    MODEL.USE_SDTPS False \
    MODEL.USE_DGAF True \
    MODEL.DGAF_VERSION v3 \
    MODEL.GLOBAL_LOCAL False \
    > ${EXP_DIR_100}/03_dgaf_v3_only.log 2>&1 &
PID_100_3=$!
echo "  GPU 2: DGAF V3 only, PID: ${PID_100_3}"

# GPU 3: SDTPS + DGAF V3
CUDA_VISIBLE_DEVICES=3 nohup python train_net.py --config_file ${CONFIG_RGBNT100} \
    MODEL.USE_SDTPS True \
    MODEL.USE_DGAF True \
    MODEL.DGAF_VERSION v3 \
    MODEL.GLOBAL_LOCAL False \
    MODEL.SDTPS_CROSS_ATTN_TYPE attention \
    MODEL.SDTPS_CROSS_ATTN_HEADS 4 \
    > ${EXP_DIR_100}/04_sdtps_dgaf_v3.log 2>&1 &
PID_100_4=$!
echo "  GPU 3: SDTPS + DGAF V3, PID: ${PID_100_4}"

echo ""
echo "Waiting for RGBNT100 to complete..."
wait ${PID_100_1} ${PID_100_2} ${PID_100_3} ${PID_100_4}
echo "RGBNT100 completed!"

# ============================================================================
# Phase 3: MSVR310 (4 GPUs in parallel)
# ============================================================================
echo ""
echo "[Phase 3/3] Starting MSVR310 experiments on 4 GPUs..."

# GPU 0: Baseline
CUDA_VISIBLE_DEVICES=0 nohup python train_net.py --config_file ${CONFIG_MSVR310} \
    MODEL.USE_SDTPS False \
    MODEL.USE_DGAF False \
    MODEL.GLOBAL_LOCAL False \
    > ${EXP_DIR_310}/01_baseline.log 2>&1 &
PID_310_1=$!
echo "  GPU 0: Baseline, PID: ${PID_310_1}"

# GPU 1: SDTPS only
CUDA_VISIBLE_DEVICES=1 nohup python train_net.py --config_file ${CONFIG_MSVR310} \
    MODEL.USE_SDTPS True \
    MODEL.USE_DGAF False \
    MODEL.GLOBAL_LOCAL False \
    MODEL.SDTPS_CROSS_ATTN_TYPE attention \
    MODEL.SDTPS_CROSS_ATTN_HEADS 4 \
    > ${EXP_DIR_310}/02_sdtps_only.log 2>&1 &
PID_310_2=$!
echo "  GPU 1: SDTPS only, PID: ${PID_310_2}"

# GPU 2: DGAF V3 only
CUDA_VISIBLE_DEVICES=2 nohup python train_net.py --config_file ${CONFIG_MSVR310} \
    MODEL.USE_SDTPS False \
    MODEL.USE_DGAF True \
    MODEL.DGAF_VERSION v3 \
    MODEL.GLOBAL_LOCAL False \
    > ${EXP_DIR_310}/03_dgaf_v3_only.log 2>&1 &
PID_310_3=$!
echo "  GPU 2: DGAF V3 only, PID: ${PID_310_3}"

# GPU 3: SDTPS + DGAF V3
CUDA_VISIBLE_DEVICES=3 nohup python train_net.py --config_file ${CONFIG_MSVR310} \
    MODEL.USE_SDTPS True \
    MODEL.USE_DGAF True \
    MODEL.DGAF_VERSION v3 \
    MODEL.GLOBAL_LOCAL False \
    MODEL.SDTPS_CROSS_ATTN_TYPE attention \
    MODEL.SDTPS_CROSS_ATTN_HEADS 4 \
    > ${EXP_DIR_310}/04_sdtps_dgaf_v3.log 2>&1 &
PID_310_4=$!
echo "  GPU 3: SDTPS + DGAF V3, PID: ${PID_310_4}"

echo ""
echo "Waiting for MSVR310 to complete..."
wait ${PID_310_1} ${PID_310_2} ${PID_310_3} ${PID_310_4}
echo "MSVR310 completed!"

echo ""
echo "=============================================================================="
echo "所有实验完成!"
echo "=============================================================================="
echo "结果位置:"
echo "  RGBNT201: ${EXP_DIR_201}"
echo "  RGBNT100: ${EXP_DIR_100}"
echo "  MSVR310:  ${EXP_DIR_310}"
echo "=============================================================================="
echo ""
echo "实验配置总结:"
echo "  1. Baseline:        无 SDTPS, 无 DGAF, 无 GLOBAL_LOCAL"
echo "  2. SDTPS only:      有 SDTPS(attention), 无 DGAF, 无 GLOBAL_LOCAL"
echo "  3. DGAF V3 only:    无 SDTPS, 有 DGAF V3, 无 GLOBAL_LOCAL"
echo "  4. SDTPS+DGAF V3:   有 SDTPS(attention), 有 DGAF V3, 无 GLOBAL_LOCAL"
echo "=============================================================================="
