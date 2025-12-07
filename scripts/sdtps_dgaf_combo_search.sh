#!/bin/bash
# ============================================================================
# SDTPS + DGAF 最佳参数组合搜索
# 在完成第一轮消融实验后，基于最佳参数进行组合搜索
#
# 使用方法：
#   1. 先运行 sdtps_dgaf_ablation.sh 完成第一轮消融
#   2. 分析结果，修改下方的 BEST_* 变量
#   3. 运行本脚本进行组合搜索
#
# 任务分配：4 块 GPU，每块 4 个任务，间隔 1 小时
# 总计：16 个实验，约 4 小时/GPU
# ============================================================================

CONFIG="configs/RGBNT201/DeMo_DGAF.yml"
SLEEP_TIME=3600  # 1小时 = 3600秒

# ============================================================================
# 第一轮消融实验的最佳参数（根据实际结果修改）
# ============================================================================
BEST_TAU=1.0           # 从 TAU 消融中选出
BEST_ALPHA=0.5         # 从 ALPHA 消融中选出
BEST_SPARSE=0.7        # 从 SPARSE_RATIO 消融中选出
BEST_AGGR=0.5          # 从 AGGR_RATIO 消融中选出
BEST_BETA=0.25         # 从 BETA 消融中选出
BEST_LOSS=1.0          # 从 LOSS_WEIGHT 消融中选出

# 创建 logs 目录
mkdir -p logs/sdtps_dgaf_combo

echo "============================================================"
echo "        SDTPS + DGAF 最佳参数组合搜索"
echo "============================================================"
echo "基准参数:"
echo "  DGAF_TAU: $BEST_TAU"
echo "  DGAF_INIT_ALPHA: $BEST_ALPHA"
echo "  SDTPS_SPARSE_RATIO: $BEST_SPARSE"
echo "  SDTPS_AGGR_RATIO: $BEST_AGGR"
echo "  SDTPS_BETA: $BEST_BETA"
echo "  SDTPS_LOSS_WEIGHT: $BEST_LOSS"
echo "============================================================"

# ============================================================================
# GPU 0: DGAF 参数组合
# ============================================================================
(
    echo "[GPU 0] 开始 DGAF 参数组合..."

    # 组合 1: 最佳 TAU + 不同 ALPHA
    echo "[GPU 0] 任务 1/4: dgaf_tau${BEST_TAU}_alpha0.4"
    CUDA_VISIBLE_DEVICES=0 python train_net.py --config_file $CONFIG \
        --exp_name "dgaf_tau${BEST_TAU}_alpha0.4" \
        MODEL.USE_DGAF True MODEL.DGAF_TAU $BEST_TAU MODEL.DGAF_INIT_ALPHA 0.4 \
        > logs/sdtps_dgaf_combo/dgaf_tau${BEST_TAU}_alpha0.4.log 2>&1
    echo "[GPU 0] 任务 1 完成，等待 1 小时..."
    sleep $SLEEP_TIME

    echo "[GPU 0] 任务 2/4: dgaf_tau${BEST_TAU}_alpha0.6"
    CUDA_VISIBLE_DEVICES=0 python train_net.py --config_file $CONFIG \
        --exp_name "dgaf_tau${BEST_TAU}_alpha0.6" \
        MODEL.USE_DGAF True MODEL.DGAF_TAU $BEST_TAU MODEL.DGAF_INIT_ALPHA 0.6 \
        > logs/sdtps_dgaf_combo/dgaf_tau${BEST_TAU}_alpha0.6.log 2>&1
    echo "[GPU 0] 任务 2 完成，等待 1 小时..."
    sleep $SLEEP_TIME

    # 组合 2: 最佳 ALPHA + 不同 TAU
    echo "[GPU 0] 任务 3/4: dgaf_tau0.7_alpha${BEST_ALPHA}"
    CUDA_VISIBLE_DEVICES=0 python train_net.py --config_file $CONFIG \
        --exp_name "dgaf_tau0.7_alpha${BEST_ALPHA}" \
        MODEL.USE_DGAF True MODEL.DGAF_TAU 0.7 MODEL.DGAF_INIT_ALPHA $BEST_ALPHA \
        > logs/sdtps_dgaf_combo/dgaf_tau0.7_alpha${BEST_ALPHA}.log 2>&1
    echo "[GPU 0] 任务 3 完成，等待 1 小时..."
    sleep $SLEEP_TIME

    echo "[GPU 0] 任务 4/4: dgaf_tau1.5_alpha${BEST_ALPHA}"
    CUDA_VISIBLE_DEVICES=0 python train_net.py --config_file $CONFIG \
        --exp_name "dgaf_tau1.5_alpha${BEST_ALPHA}" \
        MODEL.USE_DGAF True MODEL.DGAF_TAU 1.5 MODEL.DGAF_INIT_ALPHA $BEST_ALPHA \
        > logs/sdtps_dgaf_combo/dgaf_tau1.5_alpha${BEST_ALPHA}.log 2>&1

    echo "[GPU 0] 所有任务完成!"
) &

# ============================================================================
# GPU 1: SDTPS SPARSE + AGGR 组合
# ============================================================================
(
    echo "[GPU 1] 开始 SDTPS SPARSE+AGGR 组合..."

    # 组合 1: 不同 SPARSE + AGGR 组合
    echo "[GPU 1] 任务 1/4: sparse0.6_aggr0.4"
    CUDA_VISIBLE_DEVICES=1 python train_net.py --config_file $CONFIG \
        --exp_name "sparse0.6_aggr0.4" \
        MODEL.USE_DGAF True MODEL.SDTPS_SPARSE_RATIO 0.6 MODEL.SDTPS_AGGR_RATIO 0.4 \
        > logs/sdtps_dgaf_combo/sparse0.6_aggr0.4.log 2>&1
    echo "[GPU 1] 任务 1 完成，等待 1 小时..."
    sleep $SLEEP_TIME

    echo "[GPU 1] 任务 2/4: sparse0.6_aggr0.5"
    CUDA_VISIBLE_DEVICES=1 python train_net.py --config_file $CONFIG \
        --exp_name "sparse0.6_aggr0.5" \
        MODEL.USE_DGAF True MODEL.SDTPS_SPARSE_RATIO 0.6 MODEL.SDTPS_AGGR_RATIO 0.5 \
        > logs/sdtps_dgaf_combo/sparse0.6_aggr0.5.log 2>&1
    echo "[GPU 1] 任务 2 完成，等待 1 小时..."
    sleep $SLEEP_TIME

    echo "[GPU 1] 任务 3/4: sparse0.7_aggr0.4"
    CUDA_VISIBLE_DEVICES=1 python train_net.py --config_file $CONFIG \
        --exp_name "sparse0.7_aggr0.4" \
        MODEL.USE_DGAF True MODEL.SDTPS_SPARSE_RATIO 0.7 MODEL.SDTPS_AGGR_RATIO 0.4 \
        > logs/sdtps_dgaf_combo/sparse0.7_aggr0.4.log 2>&1
    echo "[GPU 1] 任务 3 完成，等待 1 小时..."
    sleep $SLEEP_TIME

    echo "[GPU 1] 任务 4/4: sparse0.8_aggr0.4"
    CUDA_VISIBLE_DEVICES=1 python train_net.py --config_file $CONFIG \
        --exp_name "sparse0.8_aggr0.4" \
        MODEL.USE_DGAF True MODEL.SDTPS_SPARSE_RATIO 0.8 MODEL.SDTPS_AGGR_RATIO 0.4 \
        > logs/sdtps_dgaf_combo/sparse0.8_aggr0.4.log 2>&1

    echo "[GPU 1] 所有任务完成!"
) &

# ============================================================================
# GPU 2: SDTPS BETA + LOSS 组合
# ============================================================================
(
    echo "[GPU 2] 开始 SDTPS BETA+LOSS 组合..."

    echo "[GPU 2] 任务 1/4: beta0.2_loss1.0"
    CUDA_VISIBLE_DEVICES=2 python train_net.py --config_file $CONFIG \
        --exp_name "beta0.2_loss1.0" \
        MODEL.USE_DGAF True MODEL.SDTPS_BETA 0.2 MODEL.SDTPS_LOSS_WEIGHT 1.0 \
        > logs/sdtps_dgaf_combo/beta0.2_loss1.0.log 2>&1
    echo "[GPU 2] 任务 1 完成，等待 1 小时..."
    sleep $SLEEP_TIME

    echo "[GPU 2] 任务 2/4: beta0.3_loss1.0"
    CUDA_VISIBLE_DEVICES=2 python train_net.py --config_file $CONFIG \
        --exp_name "beta0.3_loss1.0" \
        MODEL.USE_DGAF True MODEL.SDTPS_BETA 0.3 MODEL.SDTPS_LOSS_WEIGHT 1.0 \
        > logs/sdtps_dgaf_combo/beta0.3_loss1.0.log 2>&1
    echo "[GPU 2] 任务 2 完成，等待 1 小时..."
    sleep $SLEEP_TIME

    echo "[GPU 2] 任务 3/4: beta0.25_loss1.2"
    CUDA_VISIBLE_DEVICES=2 python train_net.py --config_file $CONFIG \
        --exp_name "beta0.25_loss1.2" \
        MODEL.USE_DGAF True MODEL.SDTPS_BETA 0.25 MODEL.SDTPS_LOSS_WEIGHT 1.2 \
        > logs/sdtps_dgaf_combo/beta0.25_loss1.2.log 2>&1
    echo "[GPU 2] 任务 3 完成，等待 1 小时..."
    sleep $SLEEP_TIME

    echo "[GPU 2] 任务 4/4: beta0.25_loss0.8"
    CUDA_VISIBLE_DEVICES=2 python train_net.py --config_file $CONFIG \
        --exp_name "beta0.25_loss0.8" \
        MODEL.USE_DGAF True MODEL.SDTPS_BETA 0.25 MODEL.SDTPS_LOSS_WEIGHT 0.8 \
        > logs/sdtps_dgaf_combo/beta0.25_loss0.8.log 2>&1

    echo "[GPU 2] 所有任务完成!"
) &

# ============================================================================
# GPU 3: 全参数最佳组合候选
# ============================================================================
(
    echo "[GPU 3] 开始全参数最佳组合候选..."

    # 候选 1: 保守配置
    echo "[GPU 3] 任务 1/4: best_combo_v1"
    CUDA_VISIBLE_DEVICES=3 python train_net.py --config_file $CONFIG \
        --exp_name "best_combo_v1" \
        MODEL.USE_DGAF True \
        MODEL.DGAF_TAU 1.0 MODEL.DGAF_INIT_ALPHA 0.5 \
        MODEL.SDTPS_SPARSE_RATIO 0.7 MODEL.SDTPS_AGGR_RATIO 0.5 \
        MODEL.SDTPS_BETA 0.25 MODEL.SDTPS_LOSS_WEIGHT 1.0 \
        > logs/sdtps_dgaf_combo/best_combo_v1.log 2>&1
    echo "[GPU 3] 任务 1 完成，等待 1 小时..."
    sleep $SLEEP_TIME

    # 候选 2: 偏向熵门控
    echo "[GPU 3] 任务 2/4: best_combo_v2_ieg"
    CUDA_VISIBLE_DEVICES=3 python train_net.py --config_file $CONFIG \
        --exp_name "best_combo_v2_ieg" \
        MODEL.USE_DGAF True \
        MODEL.DGAF_TAU 0.7 MODEL.DGAF_INIT_ALPHA 0.6 \
        MODEL.SDTPS_SPARSE_RATIO 0.7 MODEL.SDTPS_AGGR_RATIO 0.5 \
        MODEL.SDTPS_BETA 0.25 MODEL.SDTPS_LOSS_WEIGHT 1.0 \
        > logs/sdtps_dgaf_combo/best_combo_v2_ieg.log 2>&1
    echo "[GPU 3] 任务 2 完成，等待 1 小时..."
    sleep $SLEEP_TIME

    # 候选 3: 更多 tokens
    echo "[GPU 3] 任务 3/4: best_combo_v3_tokens"
    CUDA_VISIBLE_DEVICES=3 python train_net.py --config_file $CONFIG \
        --exp_name "best_combo_v3_tokens" \
        MODEL.USE_DGAF True \
        MODEL.DGAF_TAU 1.0 MODEL.DGAF_INIT_ALPHA 0.5 \
        MODEL.SDTPS_SPARSE_RATIO 0.8 MODEL.SDTPS_AGGR_RATIO 0.4 \
        MODEL.SDTPS_BETA 0.25 MODEL.SDTPS_LOSS_WEIGHT 1.0 \
        > logs/sdtps_dgaf_combo/best_combo_v3_tokens.log 2>&1
    echo "[GPU 3] 任务 3 完成，等待 1 小时..."
    sleep $SLEEP_TIME

    # 候选 4: 更高 loss weight
    echo "[GPU 3] 任务 4/4: best_combo_v4_loss"
    CUDA_VISIBLE_DEVICES=3 python train_net.py --config_file $CONFIG \
        --exp_name "best_combo_v4_loss" \
        MODEL.USE_DGAF True \
        MODEL.DGAF_TAU 1.0 MODEL.DGAF_INIT_ALPHA 0.5 \
        MODEL.SDTPS_SPARSE_RATIO 0.7 MODEL.SDTPS_AGGR_RATIO 0.5 \
        MODEL.SDTPS_BETA 0.25 MODEL.SDTPS_LOSS_WEIGHT 1.5 \
        > logs/sdtps_dgaf_combo/best_combo_v4_loss.log 2>&1

    echo "[GPU 3] 所有任务完成!"
) &

echo ""
echo "============================================================"
echo "所有 GPU 任务已启动!"
echo "============================================================"
echo ""
echo "监控命令:"
echo "  nvidia-smi"
echo "  tail -f logs/sdtps_dgaf_combo/*.log"
echo ""

wait
echo "============================================================"
echo "所有组合实验已完成!"
echo "============================================================"
echo ""
echo "结果汇总:"
echo "  grep -r 'Best mAP' logs/sdtps_dgaf_combo/*.log | sort -t: -k2 -rn"
