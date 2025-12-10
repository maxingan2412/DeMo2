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
# 4. SDTPS+DGAF V3:   有 SDTPS, 有 DGAF V3, 无 GLOBAL_LOCAL → 只用 dgaf 损失 (推荐组合)
#     注：之前用 V1+GLOBAL_LOCAL 效果差（双重压缩），现改用 V3
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
echo "配置4: SDTPS+DGAF V3   - 只用 dgaf 损失 (V3 优化：避免双重压缩)"
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

# GPU 3: SDTPS + DGAF V3 (优化：改用 V3，避免双重压缩)
CUDA_VISIBLE_DEVICES=3 nohup python train_net.py --config_file configs/RGBNT201/DeMo_SDTPS_DGAF_ablation.yml MODEL.USE_SDTPS True MODEL.USE_DGAF True MODEL.DGAF_VERSION v3 MODEL.GLOBAL_LOCAL False MODEL.SDTPS_CROSS_ATTN_TYPE attention MODEL.SDTPS_CROSS_ATTN_HEADS 4 > ${EXP_DIR_201}/04_sdtps_dgaf_v3.log 2>&1 &
PID_201_4=$!
echo "  GPU 3: SDTPS + DGAF V3 (避免双重压缩), PID: ${PID_201_4}"

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
echo "  4. SDTPS+DGAF V3:   SDTPS(selection) + DGAF V3(fusion), 只用 dgaf 损失"
echo ""
echo "优化说明: 第4个实验改用 V3（原用 V1+GLOBAL_LOCAL 效果差）"
echo "  问题: SDTPS masking + pool + reduce = 双重信息损失"
echo "  解决: V3 直接处理 masked tokens，保留局部细节"
echo "=============================================================================="
