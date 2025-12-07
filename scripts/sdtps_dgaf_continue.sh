#!/bin/bash
# ============================================================================
# SDTPS + DGAF 消融实验脚本 (续跑版 - 从任务3开始)
# 已完成: 每GPU各2个任务，共8个
# 剩余: 每GPU各4个任务，共16个
# ============================================================================

CONFIG="configs/RGBNT201/DeMo_DGAF.yml"

# 使用已有的实验目录
EXP_DIR="experiments/sdtps_dgaf_20251207_041659"

echo "============================================================"
echo "        SDTPS + DGAF 消融实验 (续跑)"
echo "============================================================"
echo "实验目录: $EXP_DIR"
echo "从任务 3 开始，无等待时间"
echo "============================================================"

# ============================================================================
# GPU 0: 继续 DGAF TAU 消融 (从任务3开始)
# 已完成: 01_sdtps_baseline, 02_dgaf_tau_0.3
# ============================================================================
(
    GPU_LOG_DIR="$EXP_DIR/gpu0_dgaf_tau"
    echo "[GPU 0] 继续执行..."

    # 任务 3: DGAF TAU=0.5
    echo "[GPU 0] 3/6: dgaf_tau_0.5"
    CUDA_VISIBLE_DEVICES=0 python train_net.py --config_file $CONFIG \
        --exp_name "dgaf_tau_0.5" \
        MODEL.USE_DGAF True MODEL.DGAF_TAU 0.5 \
        > $GPU_LOG_DIR/03_dgaf_tau_0.5.log 2>&1

    # 任务 4: DGAF TAU=1.0 (默认)
    echo "[GPU 0] 4/6: dgaf_tau_1.0"
    CUDA_VISIBLE_DEVICES=0 python train_net.py --config_file $CONFIG \
        --exp_name "dgaf_tau_1.0" \
        MODEL.USE_DGAF True MODEL.DGAF_TAU 1.0 \
        > $GPU_LOG_DIR/04_dgaf_tau_1.0.log 2>&1

    # 任务 5: DGAF TAU=1.5
    echo "[GPU 0] 5/6: dgaf_tau_1.5"
    CUDA_VISIBLE_DEVICES=0 python train_net.py --config_file $CONFIG \
        --exp_name "dgaf_tau_1.5" \
        MODEL.USE_DGAF True MODEL.DGAF_TAU 1.5 \
        > $GPU_LOG_DIR/05_dgaf_tau_1.5.log 2>&1

    # 任务 6: DGAF TAU=2.0
    echo "[GPU 0] 6/6: dgaf_tau_2.0"
    CUDA_VISIBLE_DEVICES=0 python train_net.py --config_file $CONFIG \
        --exp_name "dgaf_tau_2.0" \
        MODEL.USE_DGAF True MODEL.DGAF_TAU 2.0 \
        > $GPU_LOG_DIR/06_dgaf_tau_2.0.log 2>&1

    echo "[GPU 0] 完成!" > $GPU_LOG_DIR/DONE
) &

# ============================================================================
# GPU 1: 继续 SDTPS SPARSE_RATIO 消融 (从任务3开始)
# 已完成: 01_dgaf_alpha_0.3, 02_dgaf_alpha_0.7
# ============================================================================
(
    GPU_LOG_DIR="$EXP_DIR/gpu1_alpha_sparse"
    echo "[GPU 1] 继续执行..."

    # 任务 3: SDTPS SPARSE_RATIO=0.5
    echo "[GPU 1] 3/6: sdtps_sparse_0.5"
    CUDA_VISIBLE_DEVICES=1 python train_net.py --config_file $CONFIG \
        --exp_name "sdtps_sparse_0.5" \
        MODEL.USE_DGAF True MODEL.SDTPS_SPARSE_RATIO 0.5 \
        > $GPU_LOG_DIR/03_sdtps_sparse_0.5.log 2>&1

    # 任务 4: SDTPS SPARSE_RATIO=0.6
    echo "[GPU 1] 4/6: sdtps_sparse_0.6"
    CUDA_VISIBLE_DEVICES=1 python train_net.py --config_file $CONFIG \
        --exp_name "sdtps_sparse_0.6" \
        MODEL.USE_DGAF True MODEL.SDTPS_SPARSE_RATIO 0.6 \
        > $GPU_LOG_DIR/04_sdtps_sparse_0.6.log 2>&1

    # 任务 5: SDTPS SPARSE_RATIO=0.8
    echo "[GPU 1] 5/6: sdtps_sparse_0.8"
    CUDA_VISIBLE_DEVICES=1 python train_net.py --config_file $CONFIG \
        --exp_name "sdtps_sparse_0.8" \
        MODEL.USE_DGAF True MODEL.SDTPS_SPARSE_RATIO 0.8 \
        > $GPU_LOG_DIR/05_sdtps_sparse_0.8.log 2>&1

    # 任务 6: SDTPS SPARSE_RATIO=0.9
    echo "[GPU 1] 6/6: sdtps_sparse_0.9"
    CUDA_VISIBLE_DEVICES=1 python train_net.py --config_file $CONFIG \
        --exp_name "sdtps_sparse_0.9" \
        MODEL.USE_DGAF True MODEL.SDTPS_SPARSE_RATIO 0.9 \
        > $GPU_LOG_DIR/06_sdtps_sparse_0.9.log 2>&1

    echo "[GPU 1] 完成!" > $GPU_LOG_DIR/DONE
) &

# ============================================================================
# GPU 2: 继续 SDTPS BETA 消融 (从任务3开始)
# 已完成: 01_sdtps_aggr_0.3, 02_sdtps_aggr_0.6
# ============================================================================
(
    GPU_LOG_DIR="$EXP_DIR/gpu2_aggr_beta"
    echo "[GPU 2] 继续执行..."

    # 任务 3: SDTPS BETA=0.15
    echo "[GPU 2] 3/6: sdtps_beta_0.15"
    CUDA_VISIBLE_DEVICES=2 python train_net.py --config_file $CONFIG \
        --exp_name "sdtps_beta_0.15" \
        MODEL.USE_DGAF True MODEL.SDTPS_BETA 0.15 \
        > $GPU_LOG_DIR/03_sdtps_beta_0.15.log 2>&1

    # 任务 4: SDTPS BETA=0.2
    echo "[GPU 2] 4/6: sdtps_beta_0.2"
    CUDA_VISIBLE_DEVICES=2 python train_net.py --config_file $CONFIG \
        --exp_name "sdtps_beta_0.2" \
        MODEL.USE_DGAF True MODEL.SDTPS_BETA 0.2 \
        > $GPU_LOG_DIR/04_sdtps_beta_0.2.log 2>&1

    # 任务 5: SDTPS BETA=0.3
    echo "[GPU 2] 5/6: sdtps_beta_0.3"
    CUDA_VISIBLE_DEVICES=2 python train_net.py --config_file $CONFIG \
        --exp_name "sdtps_beta_0.3" \
        MODEL.USE_DGAF True MODEL.SDTPS_BETA 0.3 \
        > $GPU_LOG_DIR/05_sdtps_beta_0.3.log 2>&1

    # 任务 6: SDTPS BETA=0.35
    echo "[GPU 2] 6/6: sdtps_beta_0.35"
    CUDA_VISIBLE_DEVICES=2 python train_net.py --config_file $CONFIG \
        --exp_name "sdtps_beta_0.35" \
        MODEL.USE_DGAF True MODEL.SDTPS_BETA 0.35 \
        > $GPU_LOG_DIR/06_sdtps_beta_0.35.log 2>&1

    echo "[GPU 2] 完成!" > $GPU_LOG_DIR/DONE
) &

# ============================================================================
# GPU 3: 继续 LOSS_WEIGHT + 组合实验 (从任务3开始)
# 已完成: 01_sdtps_loss_0.5, 02_sdtps_loss_1.5
# ============================================================================
(
    GPU_LOG_DIR="$EXP_DIR/gpu3_loss_combo"
    echo "[GPU 3] 继续执行..."

    # 任务 3: SDTPS_LOSS_WEIGHT=2.0
    echo "[GPU 3] 3/6: sdtps_loss_2.0"
    CUDA_VISIBLE_DEVICES=3 python train_net.py --config_file $CONFIG \
        --exp_name "sdtps_loss_2.0" \
        MODEL.USE_DGAF True MODEL.SDTPS_LOSS_WEIGHT 2.0 \
        > $GPU_LOG_DIR/03_sdtps_loss_2.0.log 2>&1

    # 任务 4: 组合实验 - balanced
    echo "[GPU 3] 4/6: combo_v1 (balanced)"
    CUDA_VISIBLE_DEVICES=3 python train_net.py --config_file $CONFIG \
        --exp_name "combo_v1_balanced" \
        MODEL.USE_DGAF True \
        MODEL.DGAF_TAU 1.0 MODEL.DGAF_INIT_ALPHA 0.5 \
        MODEL.SDTPS_SPARSE_RATIO 0.7 MODEL.SDTPS_AGGR_RATIO 0.5 \
        MODEL.SDTPS_BETA 0.25 MODEL.SDTPS_LOSS_WEIGHT 1.0 \
        > $GPU_LOG_DIR/04_combo_v1_balanced.log 2>&1

    # 任务 5: 组合实验 - more_tokens
    echo "[GPU 3] 5/6: combo_v2 (more_tokens)"
    CUDA_VISIBLE_DEVICES=3 python train_net.py --config_file $CONFIG \
        --exp_name "combo_v2_more_tokens" \
        MODEL.USE_DGAF True \
        MODEL.DGAF_TAU 0.7 MODEL.DGAF_INIT_ALPHA 0.5 \
        MODEL.SDTPS_SPARSE_RATIO 0.8 MODEL.SDTPS_AGGR_RATIO 0.4 \
        MODEL.SDTPS_BETA 0.25 MODEL.SDTPS_LOSS_WEIGHT 1.0 \
        > $GPU_LOG_DIR/05_combo_v2_more_tokens.log 2>&1

    # 任务 6: 组合实验 - ieg_focus
    echo "[GPU 3] 6/6: combo_v3 (ieg_focus)"
    CUDA_VISIBLE_DEVICES=3 python train_net.py --config_file $CONFIG \
        --exp_name "combo_v3_ieg_focus" \
        MODEL.USE_DGAF True \
        MODEL.DGAF_TAU 0.5 MODEL.DGAF_INIT_ALPHA 0.6 \
        MODEL.SDTPS_SPARSE_RATIO 0.7 MODEL.SDTPS_AGGR_RATIO 0.5 \
        MODEL.SDTPS_BETA 0.25 MODEL.SDTPS_LOSS_WEIGHT 1.0 \
        > $GPU_LOG_DIR/06_combo_v3_ieg_focus.log 2>&1

    echo "[GPU 3] 完成!" > $GPU_LOG_DIR/DONE
) &

echo ""
echo "============================================================"
echo "续跑任务已启动! (无等待时间)"
echo "============================================================"
echo "剩余: 16 个实验 (每 GPU 4 个)"
echo ""
echo "监控命令:"
echo "  nvidia-smi"
echo "  tail -f $EXP_DIR/*/*.log"
echo ""

wait

echo "============================================================"
echo "所有消融实验已完成!"
echo "============================================================"

# 生成结果汇总
echo "生成结果汇总..."
SUMMARY_FILE="$EXP_DIR/results_summary.txt"
echo "SDTPS + DGAF 消融实验结果汇总" > $SUMMARY_FILE
echo "完成时间: $(date)" >> $SUMMARY_FILE
echo "============================================================" >> $SUMMARY_FILE
grep -h "Best mAP\|Best Rank-1" $EXP_DIR/*/*.log 2>/dev/null | tail -48 >> $SUMMARY_FILE
echo ""
echo "结果已保存到: $SUMMARY_FILE"
