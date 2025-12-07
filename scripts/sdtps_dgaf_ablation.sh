#!/bin/bash
# ============================================================================
# SDTPS + DGAF 消融实验脚本
# 目标：找出最佳的 SDTPS 和 DGAF 参数组合
#
# 实验设计：
#   1. DGAF 参数消融 (TAU, ALPHA)
#   2. SDTPS 参数消融 (SPARSE_RATIO, AGGR_RATIO, BETA, LOSS_WEIGHT)
#   3. 最佳参数组合验证
#
# 任务分配：4 块 GPU，每块 5 个任务，间隔 1 小时
# 总计：20 个实验，约 5 小时/GPU
# ============================================================================

CONFIG="configs/RGBNT201/DeMo_DGAF.yml"
SLEEP_TIME=3600  # 1小时 = 3600秒

# 创建 logs 目录
mkdir -p logs/sdtps_dgaf

echo "============================================================"
echo "        SDTPS + DGAF 消融实验"
echo "============================================================"
echo "配置文件: $CONFIG"
echo "任务间隔: 1 小时"
echo "总计: 20 个实验 (4 GPU × 5 任务)"
echo "预计总时间: 约 5 小时"
echo "============================================================"

# ============================================================================
# GPU 0: Baseline 对比 + DGAF TAU 消融
# ============================================================================
(
    echo "[GPU 0] 开始任务序列..."

    # 任务 1: SDTPS only (无 DGAF，作为 baseline)
    echo "[GPU 0] 任务 1/5: sdtps_only_baseline"
    CUDA_VISIBLE_DEVICES=0 python train_net.py --config_file $CONFIG \
        --exp_name "sdtps_only_baseline" \
        MODEL.USE_DGAF False \
        > logs/sdtps_dgaf/sdtps_only_baseline.log 2>&1
    echo "[GPU 0] 任务 1 完成，等待 1 小时..."
    sleep $SLEEP_TIME

    # 任务 2: SDTPS + DGAF (默认参数)
    echo "[GPU 0] 任务 2/5: sdtps_dgaf_default"
    CUDA_VISIBLE_DEVICES=0 python train_net.py --config_file $CONFIG \
        --exp_name "sdtps_dgaf_default" \
        MODEL.USE_DGAF True MODEL.DGAF_TAU 1.0 MODEL.DGAF_INIT_ALPHA 0.5 \
        > logs/sdtps_dgaf/sdtps_dgaf_default.log 2>&1
    echo "[GPU 0] 任务 2 完成，等待 1 小时..."
    sleep $SLEEP_TIME

    # 任务 3: DGAF TAU=0.5 (更尖锐的熵权重)
    echo "[GPU 0] 任务 3/5: dgaf_tau_0.5"
    CUDA_VISIBLE_DEVICES=0 python train_net.py --config_file $CONFIG \
        --exp_name "dgaf_tau_0.5" \
        MODEL.USE_DGAF True MODEL.DGAF_TAU 0.5 \
        > logs/sdtps_dgaf/dgaf_tau_0.5.log 2>&1
    echo "[GPU 0] 任务 3 完成，等待 1 小时..."
    sleep $SLEEP_TIME

    # 任务 4: DGAF TAU=2.0 (更平滑的熵权重)
    echo "[GPU 0] 任务 4/5: dgaf_tau_2.0"
    CUDA_VISIBLE_DEVICES=0 python train_net.py --config_file $CONFIG \
        --exp_name "dgaf_tau_2.0" \
        MODEL.USE_DGAF True MODEL.DGAF_TAU 2.0 \
        > logs/sdtps_dgaf/dgaf_tau_2.0.log 2>&1
    echo "[GPU 0] 任务 4 完成，等待 1 小时..."
    sleep $SLEEP_TIME

    # 任务 5: DGAF TAU=0.3 (非常尖锐)
    echo "[GPU 0] 任务 5/5: dgaf_tau_0.3"
    CUDA_VISIBLE_DEVICES=0 python train_net.py --config_file $CONFIG \
        --exp_name "dgaf_tau_0.3" \
        MODEL.USE_DGAF True MODEL.DGAF_TAU 0.3 \
        > logs/sdtps_dgaf/dgaf_tau_0.3.log 2>&1

    echo "[GPU 0] 所有任务完成!"
) &

# ============================================================================
# GPU 1: DGAF ALPHA 消融 + SDTPS SPARSE_RATIO 消融
# ============================================================================
(
    echo "[GPU 1] 开始任务序列..."

    # 任务 1: DGAF ALPHA=0.3 (偏向 MIG)
    echo "[GPU 1] 任务 1/5: dgaf_alpha_0.3"
    CUDA_VISIBLE_DEVICES=1 python train_net.py --config_file $CONFIG \
        --exp_name "dgaf_alpha_0.3" \
        MODEL.USE_DGAF True MODEL.DGAF_INIT_ALPHA 0.3 \
        > logs/sdtps_dgaf/dgaf_alpha_0.3.log 2>&1
    echo "[GPU 1] 任务 1 完成，等待 1 小时..."
    sleep $SLEEP_TIME

    # 任务 2: DGAF ALPHA=0.7 (偏向 IEG)
    echo "[GPU 1] 任务 2/5: dgaf_alpha_0.7"
    CUDA_VISIBLE_DEVICES=1 python train_net.py --config_file $CONFIG \
        --exp_name "dgaf_alpha_0.7" \
        MODEL.USE_DGAF True MODEL.DGAF_INIT_ALPHA 0.7 \
        > logs/sdtps_dgaf/dgaf_alpha_0.7.log 2>&1
    echo "[GPU 1] 任务 2 完成，等待 1 小时..."
    sleep $SLEEP_TIME

    # 任务 3: SDTPS SPARSE_RATIO=0.5
    echo "[GPU 1] 任务 3/5: sdtps_sparse_0.5"
    CUDA_VISIBLE_DEVICES=1 python train_net.py --config_file $CONFIG \
        --exp_name "sdtps_sparse_0.5" \
        MODEL.USE_DGAF True MODEL.SDTPS_SPARSE_RATIO 0.5 \
        > logs/sdtps_dgaf/sdtps_sparse_0.5.log 2>&1
    echo "[GPU 1] 任务 3 完成，等待 1 小时..."
    sleep $SLEEP_TIME

    # 任务 4: SDTPS SPARSE_RATIO=0.6
    echo "[GPU 1] 任务 4/5: sdtps_sparse_0.6"
    CUDA_VISIBLE_DEVICES=1 python train_net.py --config_file $CONFIG \
        --exp_name "sdtps_sparse_0.6" \
        MODEL.USE_DGAF True MODEL.SDTPS_SPARSE_RATIO 0.6 \
        > logs/sdtps_dgaf/sdtps_sparse_0.6.log 2>&1
    echo "[GPU 1] 任务 4 完成，等待 1 小时..."
    sleep $SLEEP_TIME

    # 任务 5: SDTPS SPARSE_RATIO=0.8
    echo "[GPU 1] 任务 5/5: sdtps_sparse_0.8"
    CUDA_VISIBLE_DEVICES=1 python train_net.py --config_file $CONFIG \
        --exp_name "sdtps_sparse_0.8" \
        MODEL.USE_DGAF True MODEL.SDTPS_SPARSE_RATIO 0.8 \
        > logs/sdtps_dgaf/sdtps_sparse_0.8.log 2>&1

    echo "[GPU 1] 所有任务完成!"
) &

# ============================================================================
# GPU 2: SDTPS AGGR_RATIO + BETA 消融
# ============================================================================
(
    echo "[GPU 2] 开始任务序列..."

    # 任务 1: SDTPS AGGR_RATIO=0.3
    echo "[GPU 2] 任务 1/5: sdtps_aggr_0.3"
    CUDA_VISIBLE_DEVICES=2 python train_net.py --config_file $CONFIG \
        --exp_name "sdtps_aggr_0.3" \
        MODEL.USE_DGAF True MODEL.SDTPS_AGGR_RATIO 0.3 \
        > logs/sdtps_dgaf/sdtps_aggr_0.3.log 2>&1
    echo "[GPU 2] 任务 1 完成，等待 1 小时..."
    sleep $SLEEP_TIME

    # 任务 2: SDTPS AGGR_RATIO=0.6
    echo "[GPU 2] 任务 2/5: sdtps_aggr_0.6"
    CUDA_VISIBLE_DEVICES=2 python train_net.py --config_file $CONFIG \
        --exp_name "sdtps_aggr_0.6" \
        MODEL.USE_DGAF True MODEL.SDTPS_AGGR_RATIO 0.6 \
        > logs/sdtps_dgaf/sdtps_aggr_0.6.log 2>&1
    echo "[GPU 2] 任务 2 完成，等待 1 小时..."
    sleep $SLEEP_TIME

    # 任务 3: SDTPS BETA=0.15
    echo "[GPU 2] 任务 3/5: sdtps_beta_0.15"
    CUDA_VISIBLE_DEVICES=2 python train_net.py --config_file $CONFIG \
        --exp_name "sdtps_beta_0.15" \
        MODEL.USE_DGAF True MODEL.SDTPS_BETA 0.15 \
        > logs/sdtps_dgaf/sdtps_beta_0.15.log 2>&1
    echo "[GPU 2] 任务 3 完成，等待 1 小时..."
    sleep $SLEEP_TIME

    # 任务 4: SDTPS BETA=0.35
    echo "[GPU 2] 任务 4/5: sdtps_beta_0.35"
    CUDA_VISIBLE_DEVICES=2 python train_net.py --config_file $CONFIG \
        --exp_name "sdtps_beta_0.35" \
        MODEL.USE_DGAF True MODEL.SDTPS_BETA 0.35 \
        > logs/sdtps_dgaf/sdtps_beta_0.35.log 2>&1
    echo "[GPU 2] 任务 4 完成，等待 1 小时..."
    sleep $SLEEP_TIME

    # 任务 5: SDTPS BETA=0.4
    echo "[GPU 2] 任务 5/5: sdtps_beta_0.4"
    CUDA_VISIBLE_DEVICES=2 python train_net.py --config_file $CONFIG \
        --exp_name "sdtps_beta_0.4" \
        MODEL.USE_DGAF True MODEL.SDTPS_BETA 0.4 \
        > logs/sdtps_dgaf/sdtps_beta_0.4.log 2>&1

    echo "[GPU 2] 所有任务完成!"
) &

# ============================================================================
# GPU 3: SDTPS LOSS_WEIGHT + 组合实验
# ============================================================================
(
    echo "[GPU 3] 开始任务序列..."

    # 任务 1: SDTPS_LOSS_WEIGHT=0.5
    echo "[GPU 3] 任务 1/5: sdtps_loss_0.5"
    CUDA_VISIBLE_DEVICES=3 python train_net.py --config_file $CONFIG \
        --exp_name "sdtps_loss_0.5" \
        MODEL.USE_DGAF True MODEL.SDTPS_LOSS_WEIGHT 0.5 \
        > logs/sdtps_dgaf/sdtps_loss_0.5.log 2>&1
    echo "[GPU 3] 任务 1 完成，等待 1 小时..."
    sleep $SLEEP_TIME

    # 任务 2: SDTPS_LOSS_WEIGHT=1.5
    echo "[GPU 3] 任务 2/5: sdtps_loss_1.5"
    CUDA_VISIBLE_DEVICES=3 python train_net.py --config_file $CONFIG \
        --exp_name "sdtps_loss_1.5" \
        MODEL.USE_DGAF True MODEL.SDTPS_LOSS_WEIGHT 1.5 \
        > logs/sdtps_dgaf/sdtps_loss_1.5.log 2>&1
    echo "[GPU 3] 任务 2 完成，等待 1 小时..."
    sleep $SLEEP_TIME

    # 任务 3: SDTPS_LOSS_WEIGHT=2.0
    echo "[GPU 3] 任务 3/5: sdtps_loss_2.0"
    CUDA_VISIBLE_DEVICES=3 python train_net.py --config_file $CONFIG \
        --exp_name "sdtps_loss_2.0" \
        MODEL.USE_DGAF True MODEL.SDTPS_LOSS_WEIGHT 2.0 \
        > logs/sdtps_dgaf/sdtps_loss_2.0.log 2>&1
    echo "[GPU 3] 任务 3 完成，等待 1 小时..."
    sleep $SLEEP_TIME

    # 任务 4: 组合1 - 较大稀疏比例 + 较小TAU
    echo "[GPU 3] 任务 4/5: combo_sparse0.8_tau0.5"
    CUDA_VISIBLE_DEVICES=3 python train_net.py --config_file $CONFIG \
        --exp_name "combo_sparse0.8_tau0.5" \
        MODEL.USE_DGAF True MODEL.SDTPS_SPARSE_RATIO 0.8 MODEL.DGAF_TAU 0.5 \
        > logs/sdtps_dgaf/combo_sparse0.8_tau0.5.log 2>&1
    echo "[GPU 3] 任务 4 完成，等待 1 小时..."
    sleep $SLEEP_TIME

    # 任务 5: 组合2 - 较小稀疏比例 + 偏向IEG
    echo "[GPU 3] 任务 5/5: combo_sparse0.6_alpha0.7"
    CUDA_VISIBLE_DEVICES=3 python train_net.py --config_file $CONFIG \
        --exp_name "combo_sparse0.6_alpha0.7" \
        MODEL.USE_DGAF True MODEL.SDTPS_SPARSE_RATIO 0.6 MODEL.DGAF_INIT_ALPHA 0.7 \
        > logs/sdtps_dgaf/combo_sparse0.6_alpha0.7.log 2>&1

    echo "[GPU 3] 所有任务完成!"
) &

echo ""
echo "============================================================"
echo "所有 GPU 任务已启动!"
echo "============================================================"
echo ""
echo "实验列表："
echo "  GPU 0: baseline对比 + DGAF TAU消融 (5个)"
echo "  GPU 1: DGAF ALPHA消融 + SDTPS SPARSE消融 (5个)"
echo "  GPU 2: SDTPS AGGR + BETA消融 (5个)"
echo "  GPU 3: SDTPS LOSS消融 + 组合实验 (5个)"
echo ""
echo "监控命令:"
echo "  nvidia-smi"
echo "  ps aux | grep train_net"
echo "  tail -f logs/sdtps_dgaf/*.log"
echo ""
echo "查看特定日志:"
echo "  tail -f logs/sdtps_dgaf/sdtps_dgaf_default.log"
echo ""

# 等待所有后台任务完成
wait
echo "============================================================"
echo "所有消融实验已完成!"
echo "============================================================"
echo ""
echo "结果汇总命令:"
echo "  grep -r 'Best mAP' logs/sdtps_dgaf/*.log"
echo "  grep -r 'Best Rank-1' logs/sdtps_dgaf/*.log"
