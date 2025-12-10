#!/bin/bash
# ============================================================================
# DeMo_Parallel 消融实验 - RGBNT201 数据集
# 7种配置在多个GPU上执行（分两轮或按需选择）
# ============================================================================
# 用法:
#   bash run_ablation_parallel_201.sh [实验标识]
#
# 示例:
#   bash run_ablation_parallel_201.sh parallel_test1
#   bash run_ablation_parallel_201.sh 9heads_full
#   bash run_ablation_parallel_201.sh  # 默认无标识
# ============================================================================
#
# 7种配置（通过损失权重控制分支）:
# 1. Fused only:       只用 Fused 分支（3个头）
# 2. SDTPS only:       只用 SDTPS 分支（3个头）
# 3. DGAF only:        只用 DGAF 分支（3个头）
# 4. SDTPS + DGAF:     SDTPS + DGAF（6个头）
# 5. SDTPS + Fused:    SDTPS + Fused（6个头）
# 6. DGAF + Fused:     DGAF + Fused（6个头）
# 7. Full (all 3):     所有3个分支（9个头）
#
# ============================================================================

# 获取命令行参数（可选的实验标识）
EXP_TAG="${1:-}"

# Get timestamp for folder name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 构建实验目录名（包含可选标识）
if [ -n "$EXP_TAG" ]; then
    EXP_DIR="logs/RGBNT201_parallel_ablation_${EXP_TAG}_${TIMESTAMP}"
else
    EXP_DIR="logs/RGBNT201_parallel_ablation_${TIMESTAMP}"
fi

mkdir -p ${EXP_DIR}

echo "=============================================================================="
echo "DeMo_Parallel 消融实验 - RGBNT201"
if [ -n "$EXP_TAG" ]; then
    echo "实验标识: ${EXP_TAG}"
fi
echo "=============================================================================="
echo "配置说明:"
echo "  1. Fused only:    只用 Fused 分支（3个分类头）"
echo "  2. SDTPS only:    只用 SDTPS 分支（3个分类头）"
echo "  3. DGAF only:     只用 DGAF 分支（3个分类头）"
echo "  4. SDTPS+DGAF:    SDTPS + DGAF（6个分类头）"
echo "  5. SDTPS+Fused:   SDTPS + Fused（6个分类头）"
echo "  6. DGAF+Fused:    DGAF + Fused（6个分类头）"
echo "  7. Full (9头):    所有3个分支（9个分类头）"
echo "=============================================================================="
echo "结果目录: ${EXP_DIR}"
echo "=============================================================================="

# 构建 log 文件名后缀
if [ -n "$EXP_TAG" ]; then
    LOG_SUFFIX="_${EXP_TAG}"
else
    LOG_SUFFIX=""
fi

# 配置文件
CONFIG="configs/RGBNT201/DeMo_Parallel.yml"

# ============================================================================
# 第一轮: 4个单分支和双分支实验 (4 GPUs)
# ============================================================================
echo ""
echo "[第一轮] 运行 4个基础实验（4 GPUs 并行）..."

# GPU 0: Fused only (权重: 0, 0, 1)
CUDA_VISIBLE_DEVICES=0 nohup python train_net.py --config_file ${CONFIG} \
    MODEL.SDTPS_LOSS_WEIGHT 0.0 MODEL.DGAF_LOSS_WEIGHT 0.0 MODEL.FUSED_LOSS_WEIGHT 1.0 \
    > ${EXP_DIR}/01_fused_only${LOG_SUFFIX}.log 2>&1 &
PID_1=$!
echo "  GPU 0: Fused only (3头), PID: ${PID_1}"

# GPU 1: SDTPS only (权重: 1, 0, 0)
CUDA_VISIBLE_DEVICES=1 nohup python train_net.py --config_file ${CONFIG} \
    MODEL.SDTPS_LOSS_WEIGHT 1.0 MODEL.DGAF_LOSS_WEIGHT 0.0 MODEL.FUSED_LOSS_WEIGHT 0.0 \
    > ${EXP_DIR}/02_sdtps_only${LOG_SUFFIX}.log 2>&1 &
PID_2=$!
echo "  GPU 1: SDTPS only (3头), PID: ${PID_2}"

# GPU 2: DGAF only (权重: 0, 1, 0)
CUDA_VISIBLE_DEVICES=2 nohup python train_net.py --config_file ${CONFIG} \
    MODEL.SDTPS_LOSS_WEIGHT 0.0 MODEL.DGAF_LOSS_WEIGHT 1.0 MODEL.FUSED_LOSS_WEIGHT 0.0 \
    > ${EXP_DIR}/03_dgaf_only${LOG_SUFFIX}.log 2>&1 &
PID_3=$!
echo "  GPU 2: DGAF only (3头), PID: ${PID_3}"

# GPU 3: SDTPS + DGAF (权重: 1, 1, 0)
CUDA_VISIBLE_DEVICES=3 nohup python train_net.py --config_file ${CONFIG} \
    MODEL.SDTPS_LOSS_WEIGHT 1.0 MODEL.DGAF_LOSS_WEIGHT 1.0 MODEL.FUSED_LOSS_WEIGHT 0.0 \
    > ${EXP_DIR}/04_sdtps_dgaf${LOG_SUFFIX}.log 2>&1 &
PID_4=$!
echo "  GPU 3: SDTPS + DGAF (6头), PID: ${PID_4}"

echo ""
echo "等待第一轮实验完成..."
wait ${PID_1} ${PID_2} ${PID_3} ${PID_4}
echo "第一轮完成!"

# ============================================================================
# 第二轮: 3个组合实验 (3 GPUs)
# ============================================================================
echo ""
echo "[第二轮] 运行 3个组合实验（3 GPUs 并行）..."

# GPU 0: SDTPS + Fused (权重: 1, 0, 1)
CUDA_VISIBLE_DEVICES=0 nohup python train_net.py --config_file ${CONFIG} \
    MODEL.SDTPS_LOSS_WEIGHT 1.0 MODEL.DGAF_LOSS_WEIGHT 0.0 MODEL.FUSED_LOSS_WEIGHT 1.0 \
    > ${EXP_DIR}/05_sdtps_fused${LOG_SUFFIX}.log 2>&1 &
PID_5=$!
echo "  GPU 0: SDTPS + Fused (6头), PID: ${PID_5}"

# GPU 1: DGAF + Fused (权重: 0, 1, 1)
CUDA_VISIBLE_DEVICES=1 nohup python train_net.py --config_file ${CONFIG} \
    MODEL.SDTPS_LOSS_WEIGHT 0.0 MODEL.DGAF_LOSS_WEIGHT 1.0 MODEL.FUSED_LOSS_WEIGHT 1.0 \
    > ${EXP_DIR}/06_dgaf_fused${LOG_SUFFIX}.log 2>&1 &
PID_6=$!
echo "  GPU 1: DGAF + Fused (6头), PID: ${PID_6}"

# GPU 2: Full - 所有3个分支 (权重: 1, 1, 1)
CUDA_VISIBLE_DEVICES=2 nohup python train_net.py --config_file ${CONFIG} \
    MODEL.SDTPS_LOSS_WEIGHT 1.0 MODEL.DGAF_LOSS_WEIGHT 1.0 MODEL.FUSED_LOSS_WEIGHT 1.0 \
    > ${EXP_DIR}/07_full_9heads${LOG_SUFFIX}.log 2>&1 &
PID_7=$!
echo "  GPU 2: Full - 所有分支 (9头), PID: ${PID_7}"

echo ""
echo "等待第二轮实验完成..."
wait ${PID_5} ${PID_6} ${PID_7}
echo "第二轮完成!"

echo ""
echo "=============================================================================="
echo "所有实验完成!"
echo "=============================================================================="
echo "结果位置: ${EXP_DIR}"
echo "=============================================================================="
echo ""
echo "实验总结:"
echo "  第一轮（4个基础实验）:"
echo "    1. Fused only    - 3个头 (global-local fusion)"
echo "    2. SDTPS only    - 3个头 (token selection)"
echo "    3. DGAF only     - 3个头 (adaptive fusion)"
echo "    4. SDTPS+DGAF    - 6个头 (组合1)"
echo ""
echo "  第二轮（3个组合实验）:"
echo "    5. SDTPS+Fused   - 6个头 (组合2)"
echo "    6. DGAF+Fused    - 6个头 (组合3)"
echo "    7. Full          - 9个头 (完整架构)"
echo ""
echo "分析建议:"
echo "  - 对比单分支性能（1-3）识别最强分支"
echo "  - 对比双分支组合（4-6）分析互补性"
echo "  - 观察9头是否过拟合（7 vs 最佳双分支）"
echo "=============================================================================="
