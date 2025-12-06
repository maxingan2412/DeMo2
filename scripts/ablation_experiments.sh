#!/bin/bash
# ============================================================================
# 消融实验脚本
# 基于 configs/RGBNT201/DeMo_SACR_SDTPS_LIF.yml
# 分散到 GPU 0-3，每块卡顺序执行任务，任务间隔 2 小时
# ============================================================================

CONFIG="configs/RGBNT201/DeMo_SACR_SDTPS_LIF.yml"
SLEEP_TIME=7200  # 2小时 = 7200秒

# 创建 logs 目录
mkdir -p logs

echo "========== 启动消融实验 =========="
echo "每块 GPU 的任务将顺序执行，任务间隔 2 小时"

# ============================================================================
# GPU 0: Baseline + SACR_only + LIF_BETA消融
# ============================================================================
(
    echo "[GPU 0] 开始任务序列..."

    # 任务 1: Baseline (无任何模块)
    echo "[GPU 0] 任务 1/4: ablation_baseline"
    CUDA_VISIBLE_DEVICES=0 python train_net.py --config_file $CONFIG \
        --exp_name "ablation_baseline" \
        MODEL.USE_SACR False MODEL.USE_SDTPS False MODEL.USE_LIF False \
        > logs/ablation_baseline.log 2>&1

    echo "[GPU 0] 任务 1 完成，等待 2 小时..."
    sleep $SLEEP_TIME

    # 任务 2: +SACR
    echo "[GPU 0] 任务 2/4: ablation_SACR_only"
    CUDA_VISIBLE_DEVICES=0 python train_net.py --config_file $CONFIG \
        --exp_name "ablation_SACR_only" \
        MODEL.USE_SACR True MODEL.USE_SDTPS False MODEL.USE_LIF False \
        > logs/ablation_SACR_only.log 2>&1

    echo "[GPU 0] 任务 2 完成，等待 2 小时..."
    sleep $SLEEP_TIME

    # 任务 3: LIF_BETA 0.2
    echo "[GPU 0] 任务 3/4: ablation_LIF_BETA_0.2"
    CUDA_VISIBLE_DEVICES=0 python train_net.py --config_file $CONFIG \
        --exp_name "ablation_LIF_BETA_0.2" \
        MODEL.LIF_BETA 0.2 \
        > logs/ablation_LIF_BETA_0.2.log 2>&1

    echo "[GPU 0] 任务 3 完成，等待 2 小时..."
    sleep $SLEEP_TIME

    # 任务 4: LIF_BETA 0.6
    echo "[GPU 0] 任务 4/4: ablation_LIF_BETA_0.6"
    CUDA_VISIBLE_DEVICES=0 python train_net.py --config_file $CONFIG \
        --exp_name "ablation_LIF_BETA_0.6" \
        MODEL.LIF_BETA 0.6 \
        > logs/ablation_LIF_BETA_0.6.log 2>&1

    echo "[GPU 0] 所有任务完成!"
) &

# ============================================================================
# GPU 1: SDTPS_only + LIF_only + LIF_LOSS_WEIGHT消融
# ============================================================================
(
    echo "[GPU 1] 开始任务序列..."

    # 任务 1: +SDTPS
    echo "[GPU 1] 任务 1/4: ablation_SDTPS_only"
    CUDA_VISIBLE_DEVICES=1 python train_net.py --config_file $CONFIG \
        --exp_name "ablation_SDTPS_only" \
        MODEL.USE_SACR False MODEL.USE_SDTPS True MODEL.USE_LIF False \
        > logs/ablation_SDTPS_only.log 2>&1

    echo "[GPU 1] 任务 1 完成，等待 2 小时..."
    sleep $SLEEP_TIME

    # 任务 2: +LIF
    echo "[GPU 1] 任务 2/4: ablation_LIF_only"
    CUDA_VISIBLE_DEVICES=1 python train_net.py --config_file $CONFIG \
        --exp_name "ablation_LIF_only" \
        MODEL.USE_SACR False MODEL.USE_SDTPS False MODEL.USE_LIF True \
        > logs/ablation_LIF_only.log 2>&1

    echo "[GPU 1] 任务 2 完成，等待 2 小时..."
    sleep $SLEEP_TIME

    # 任务 3: LIF_LOSS_WEIGHT 0.1
    echo "[GPU 1] 任务 3/4: ablation_LIF_LOSS_WEIGHT_0.1"
    CUDA_VISIBLE_DEVICES=1 python train_net.py --config_file $CONFIG \
        --exp_name "ablation_LIF_LOSS_WEIGHT_0.1" \
        MODEL.LIF_LOSS_WEIGHT 0.1 \
        > logs/ablation_LIF_LOSS_WEIGHT_0.1.log 2>&1

    echo "[GPU 1] 任务 3 完成，等待 2 小时..."
    sleep $SLEEP_TIME

    # 任务 4: LIF_LOSS_WEIGHT 0.3
    echo "[GPU 1] 任务 4/4: ablation_LIF_LOSS_WEIGHT_0.3"
    CUDA_VISIBLE_DEVICES=1 python train_net.py --config_file $CONFIG \
        --exp_name "ablation_LIF_LOSS_WEIGHT_0.3" \
        MODEL.LIF_LOSS_WEIGHT 0.3 \
        > logs/ablation_LIF_LOSS_WEIGHT_0.3.log 2>&1

    echo "[GPU 1] 所有任务完成!"
) &

# ============================================================================
# GPU 2: SACR+SDTPS + SACR+LIF + SDTPS_SPARSE_RATIO消融
# ============================================================================
(
    echo "[GPU 2] 开始任务序列..."

    # 任务 1: +SACR+SDTPS
    echo "[GPU 2] 任务 1/4: ablation_SACR_SDTPS"
    CUDA_VISIBLE_DEVICES=2 python train_net.py --config_file $CONFIG \
        --exp_name "ablation_SACR_SDTPS" \
        MODEL.USE_SACR True MODEL.USE_SDTPS True MODEL.USE_LIF False \
        > logs/ablation_SACR_SDTPS.log 2>&1

    echo "[GPU 2] 任务 1 完成，等待 2 小时..."
    sleep $SLEEP_TIME

    # 任务 2: +SACR+LIF
    echo "[GPU 2] 任务 2/4: ablation_SACR_LIF"
    CUDA_VISIBLE_DEVICES=2 python train_net.py --config_file $CONFIG \
        --exp_name "ablation_SACR_LIF" \
        MODEL.USE_SACR True MODEL.USE_SDTPS False MODEL.USE_LIF True \
        > logs/ablation_SACR_LIF.log 2>&1

    echo "[GPU 2] 任务 2 完成，等待 2 小时..."
    sleep $SLEEP_TIME

    # 任务 3: SDTPS_SPARSE_RATIO 0.5
    echo "[GPU 2] 任务 3/4: ablation_SDTPS_SPARSE_0.5"
    CUDA_VISIBLE_DEVICES=2 python train_net.py --config_file $CONFIG \
        --exp_name "ablation_SDTPS_SPARSE_0.5" \
        MODEL.SDTPS_SPARSE_RATIO 0.5 \
        > logs/ablation_SDTPS_SPARSE_0.5.log 2>&1

    echo "[GPU 2] 任务 3 完成，等待 2 小时..."
    sleep $SLEEP_TIME

    # 任务 4: SDTPS_SPARSE_RATIO 0.8
    echo "[GPU 2] 任务 4/4: ablation_SDTPS_SPARSE_0.8"
    CUDA_VISIBLE_DEVICES=2 python train_net.py --config_file $CONFIG \
        --exp_name "ablation_SDTPS_SPARSE_0.8" \
        MODEL.SDTPS_SPARSE_RATIO 0.8 \
        > logs/ablation_SDTPS_SPARSE_0.8.log 2>&1

    echo "[GPU 2] 所有任务完成!"
) &

# ============================================================================
# GPU 3: SDTPS+LIF + Full + SDTPS_LOSS_WEIGHT消融
# ============================================================================
(
    echo "[GPU 3] 开始任务序列..."

    # 任务 1: +SDTPS+LIF
    echo "[GPU 3] 任务 1/4: ablation_SDTPS_LIF"
    CUDA_VISIBLE_DEVICES=3 python train_net.py --config_file $CONFIG \
        --exp_name "ablation_SDTPS_LIF" \
        MODEL.USE_SACR False MODEL.USE_SDTPS True MODEL.USE_LIF True \
        > logs/ablation_SDTPS_LIF.log 2>&1

    echo "[GPU 3] 任务 1 完成，等待 2 小时..."
    sleep $SLEEP_TIME

    # 任务 2: Full (SACR+SDTPS+LIF)
    echo "[GPU 3] 任务 2/4: ablation_full_SACR_SDTPS_LIF"
    CUDA_VISIBLE_DEVICES=3 python train_net.py --config_file $CONFIG \
        --exp_name "ablation_full_SACR_SDTPS_LIF" \
        MODEL.USE_SACR True MODEL.USE_SDTPS True MODEL.USE_LIF True \
        > logs/ablation_full_SACR_SDTPS_LIF.log 2>&1

    echo "[GPU 3] 任务 2 完成，等待 2 小时..."
    sleep $SLEEP_TIME

    # 任务 3: SDTPS_LOSS_WEIGHT 1.0
    echo "[GPU 3] 任务 3/4: ablation_SDTPS_LOSS_1.0"
    CUDA_VISIBLE_DEVICES=3 python train_net.py --config_file $CONFIG \
        --exp_name "ablation_SDTPS_LOSS_1.0" \
        MODEL.SDTPS_LOSS_WEIGHT 1.0 \
        > logs/ablation_SDTPS_LOSS_1.0.log 2>&1

    echo "[GPU 3] 任务 3 完成，等待 2 小时..."
    sleep $SLEEP_TIME

    # 任务 4: SDTPS_LOSS_WEIGHT 3.0
    echo "[GPU 3] 任务 4/4: ablation_SDTPS_LOSS_3.0"
    CUDA_VISIBLE_DEVICES=3 python train_net.py --config_file $CONFIG \
        --exp_name "ablation_SDTPS_LOSS_3.0" \
        MODEL.SDTPS_LOSS_WEIGHT 3.0 \
        > logs/ablation_SDTPS_LOSS_3.0.log 2>&1

    echo "[GPU 3] 所有任务完成!"
) &

echo "========== 所有 GPU 任务已启动 =========="
echo "每块 GPU 运行 4 个任务，总计 16 个实验"
echo "预计总时间: 约 8 小时（每 GPU 4 个任务 × 2 小时）"
echo ""
echo "查看运行状态:"
echo "  nvidia-smi"
echo "  ps aux | grep train_net"
echo ""
echo "查看日志:"
echo "  tail -f logs/ablation_*.log"

# 等待所有后台任务完成
wait
echo "========== 所有消融实验已完成 =========="
