#!/bin/bash
# ============================================================================
# FRCA 模块消融实验 - 仅 RGBNT201 数据集
# 4种配置在4个GPU上并行执行（测试 FRCA 替代 SDTPS 的效果）
# ============================================================================
# 用法:
#   bash run_ablation_frca_201.sh [实验标识]
#
# 示例:
#   bash run_ablation_frca_201.sh frca_test1
#   bash run_ablation_frca_201.sh  # 默认无标识
# ============================================================================
#
# 4种配置:
# 1. Baseline:        无 FRCA, 无 DGAF → 只用 ori 损失
# 2. FRCA only:       有 FRCA, 无 DGAF → 只用 frca 损失
# 3. DGAF V3 only:    无 FRCA, 有 DGAF V3 → 只用 dgaf 损失
# 4. FRCA+DGAF V3:    有 FRCA, 有 DGAF V3 → 只用 dgaf 损失
#
# 对比 SDTPS: 用 FRCA 替换 SDTPS，其余设置相同
# ============================================================================

# 获取命令行参数（可选的实验标识）
EXP_TAG="${1:-}"

# Get timestamp for folder name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 构建实验目录名（包含可选标识）
if [ -n "$EXP_TAG" ]; then
    EXP_DIR_201="logs/RGBNT201_frca_ablation_${EXP_TAG}_${TIMESTAMP}"
else
    EXP_DIR_201="logs/RGBNT201_frca_ablation_${TIMESTAMP}"
fi

mkdir -p ${EXP_DIR_201}

echo "=============================================================================="
echo "FRCA 模块消融实验 - RGBNT201"
if [ -n "$EXP_TAG" ]; then
    echo "实验标识: ${EXP_TAG}"
fi
echo "=============================================================================="
echo "配置1: Baseline      - 只用 ori 损失"
echo "配置2: FRCA only     - 只用 frca 损失 (替代 SDTPS)"
echo "配置3: DGAF V3 only  - 只用 dgaf 损失"
echo "配置4: FRCA+DGAF V3  - 只用 dgaf 损失 (FRCA 替代 SDTPS)"
echo "=============================================================================="
echo "结果目录: ${EXP_DIR_201}"
echo "=============================================================================="

# 基础配置文件与通用开关（确保不混用 SDTPS）
BASE_CFG="configs/RGBNT201/DeMo_SDTPS_DGAF_ablation.yml"
COMMON_OFF="MODEL.USE_SDTPS False MODEL.GLOBAL_LOCAL False"

# 构建 log 文件名（包含可选标识）
if [ -n "$EXP_TAG" ]; then
    LOG_SUFFIX="_${EXP_TAG}"
else
    LOG_SUFFIX=""
fi

# ============================================================================
# RGBNT201 实验 (4 GPUs in parallel)
# ============================================================================
echo ""
echo "Starting RGBNT201 FRCA experiments on 4 GPUs..."

# GPU 0: Baseline
CUDA_VISIBLE_DEVICES=0 nohup python train_net.py --config_file ${BASE_CFG} ${COMMON_OFF} MODEL.USE_FRCA False MODEL.USE_DGAF False > ${EXP_DIR_201}/01_baseline${LOG_SUFFIX}.log 2>&1 &
PID_1=$!
echo "  GPU 0: Baseline, PID: ${PID_1}"

# GPU 1: FRCA only (no DGAF, no GLOBAL_LOCAL)
CUDA_VISIBLE_DEVICES=1 nohup python train_net.py --config_file ${BASE_CFG} ${COMMON_OFF} MODEL.USE_FRCA True MODEL.USE_DGAF False > ${EXP_DIR_201}/02_frca_only${LOG_SUFFIX}.log 2>&1 &
PID_2=$!
echo "  GPU 1: FRCA only, PID: ${PID_2}"

# GPU 2: DGAF V3 only
CUDA_VISIBLE_DEVICES=2 nohup python train_net.py --config_file ${BASE_CFG} ${COMMON_OFF} MODEL.USE_FRCA False MODEL.USE_DGAF True MODEL.DGAF_VERSION v3 > ${EXP_DIR_201}/03_dgaf_v3_only${LOG_SUFFIX}.log 2>&1 &
PID_3=$!
echo "  GPU 2: DGAF V3 only, PID: ${PID_3}"

# GPU 3: FRCA + DGAF V3
CUDA_VISIBLE_DEVICES=3 nohup python train_net.py --config_file ${BASE_CFG} ${COMMON_OFF} MODEL.USE_FRCA True MODEL.USE_DGAF True MODEL.DGAF_VERSION v3 > ${EXP_DIR_201}/04_frca_dgaf_v3${LOG_SUFFIX}.log 2>&1 &
PID_4=$!
echo "  GPU 3: FRCA + DGAF V3, PID: ${PID_4}"

echo ""
echo "Waiting for all experiments to complete..."
wait ${PID_1} ${PID_2} ${PID_3} ${PID_4}
echo "FRCA experiments completed!"

echo ""
echo "=============================================================================="
echo "实验完成!"
echo "=============================================================================="
echo "结果位置: ${EXP_DIR_201}"
echo "=============================================================================="
echo ""
echo "实验配置总结:"
echo "  1. Baseline:     只有 backbone, 只用 ori 损失"
echo "  2. FRCA only:    FRCA(Fourier), 只用 frca 损失"
echo "  3. DGAF V3 only: DGAF V3, 只用 dgaf 损失"
echo "  4. FRCA+DGAF V3: FRCA + DGAF V3, 只用 dgaf 损失"
echo ""
echo "对比分析:"
echo "  将此实验结果与 SDTPS 实验对比，评估 FRCA vs SDTPS 的效果"
echo "=============================================================================="
