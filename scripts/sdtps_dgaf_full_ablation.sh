#!/bin/bash
# ============================================================================
# SDTPS + DGAF 完整消融实验脚本 (终极版)
#
# 日志保存位置: experiments/sdtps_dgaf_ablation_$(date)
#
# 实验分为两个阶段：
#   阶段1: 单参数消融 (找出每个参数的最佳值)
#   阶段2: 参数组合搜索 (组合最佳参数)
#
# 任务分配：4 块 GPU，间隔 1 小时
# ============================================================================

# 实验配置
CONFIG="configs/RGBNT201/DeMo_DGAF.yml"
SLEEP_TIME=3600  # 1小时

# 创建带时间戳的实验目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_DIR="experiments/sdtps_dgaf_${TIMESTAMP}"
mkdir -p $EXP_DIR

# 保存实验配置
cat > $EXP_DIR/experiment_config.txt << EOF
实验时间: $(date)
配置文件: $CONFIG
任务间隔: 1 小时
GPU数量: 4
总实验数: 24

实验设计:
=========
阶段1 - 单参数消融:
  GPU 0: Baseline + DGAF TAU 消融 (0.3, 0.5, 1.0, 2.0)
  GPU 1: DGAF ALPHA 消融 (0.3, 0.5, 0.7) + SDTPS SPARSE 消融
  GPU 2: SDTPS AGGR + BETA 消融
  GPU 3: SDTPS LOSS_WEIGHT 消融 + 组合实验

实验参数范围:
  DGAF_TAU: [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
  DGAF_INIT_ALPHA: [0.3, 0.4, 0.5, 0.6, 0.7]
  SDTPS_SPARSE_RATIO: [0.5, 0.6, 0.7, 0.8]
  SDTPS_AGGR_RATIO: [0.3, 0.4, 0.5, 0.6]
  SDTPS_BETA: [0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
  SDTPS_LOSS_WEIGHT: [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
EOF

echo "============================================================"
echo "        SDTPS + DGAF 完整消融实验"
echo "============================================================"
echo "实验目录: $EXP_DIR"
echo "配置文件: $CONFIG"
echo "任务间隔: 1 小时"
echo "============================================================"

# ============================================================================
# GPU 0: Baseline 对比 + DGAF TAU 消融
# ============================================================================
(
    GPU_LOG_DIR="$EXP_DIR/gpu0_dgaf_tau"
    mkdir -p $GPU_LOG_DIR
    echo "[GPU 0] 日志目录: $GPU_LOG_DIR"

    # 任务 1: SDTPS only (baseline)
    echo "[GPU 0] 1/6: sdtps_baseline (无DGAF)"
    CUDA_VISIBLE_DEVICES=0 python train_net.py --config_file $CONFIG \
        --exp_name "sdtps_baseline" \
        MODEL.USE_DGAF False \
        > $GPU_LOG_DIR/01_sdtps_baseline.log 2>&1
    sleep $SLEEP_TIME

    # 任务 2: DGAF TAU=0.3
    echo "[GPU 0] 2/6: dgaf_tau_0.3"
    CUDA_VISIBLE_DEVICES=0 python train_net.py --config_file $CONFIG \
        --exp_name "dgaf_tau_0.3" \
        MODEL.USE_DGAF True MODEL.DGAF_TAU 0.3 \
        > $GPU_LOG_DIR/02_dgaf_tau_0.3.log 2>&1
    sleep $SLEEP_TIME

    # 任务 3: DGAF TAU=0.5
    echo "[GPU 0] 3/6: dgaf_tau_0.5"
    CUDA_VISIBLE_DEVICES=0 python train_net.py --config_file $CONFIG \
        --exp_name "dgaf_tau_0.5" \
        MODEL.USE_DGAF True MODEL.DGAF_TAU 0.5 \
        > $GPU_LOG_DIR/03_dgaf_tau_0.5.log 2>&1
    sleep $SLEEP_TIME

    # 任务 4: DGAF TAU=1.0 (默认)
    echo "[GPU 0] 4/6: dgaf_tau_1.0"
    CUDA_VISIBLE_DEVICES=0 python train_net.py --config_file $CONFIG \
        --exp_name "dgaf_tau_1.0" \
        MODEL.USE_DGAF True MODEL.DGAF_TAU 1.0 \
        > $GPU_LOG_DIR/04_dgaf_tau_1.0.log 2>&1
    sleep $SLEEP_TIME

    # 任务 5: DGAF TAU=1.5
    echo "[GPU 0] 5/6: dgaf_tau_1.5"
    CUDA_VISIBLE_DEVICES=0 python train_net.py --config_file $CONFIG \
        --exp_name "dgaf_tau_1.5" \
        MODEL.USE_DGAF True MODEL.DGAF_TAU 1.5 \
        > $GPU_LOG_DIR/05_dgaf_tau_1.5.log 2>&1
    sleep $SLEEP_TIME

    # 任务 6: DGAF TAU=2.0
    echo "[GPU 0] 6/6: dgaf_tau_2.0"
    CUDA_VISIBLE_DEVICES=0 python train_net.py --config_file $CONFIG \
        --exp_name "dgaf_tau_2.0" \
        MODEL.USE_DGAF True MODEL.DGAF_TAU 2.0 \
        > $GPU_LOG_DIR/06_dgaf_tau_2.0.log 2>&1

    echo "[GPU 0] 完成!" > $GPU_LOG_DIR/DONE
) &

# ============================================================================
# GPU 1: DGAF ALPHA 消融 + SDTPS SPARSE_RATIO 消融
# ============================================================================
(
    GPU_LOG_DIR="$EXP_DIR/gpu1_alpha_sparse"
    mkdir -p $GPU_LOG_DIR
    echo "[GPU 1] 日志目录: $GPU_LOG_DIR"

    # DGAF ALPHA 消融
    echo "[GPU 1] 1/6: dgaf_alpha_0.3"
    CUDA_VISIBLE_DEVICES=1 python train_net.py --config_file $CONFIG \
        --exp_name "dgaf_alpha_0.3" \
        MODEL.USE_DGAF True MODEL.DGAF_INIT_ALPHA 0.3 \
        > $GPU_LOG_DIR/01_dgaf_alpha_0.3.log 2>&1
    sleep $SLEEP_TIME

    echo "[GPU 1] 2/6: dgaf_alpha_0.7"
    CUDA_VISIBLE_DEVICES=1 python train_net.py --config_file $CONFIG \
        --exp_name "dgaf_alpha_0.7" \
        MODEL.USE_DGAF True MODEL.DGAF_INIT_ALPHA 0.7 \
        > $GPU_LOG_DIR/02_dgaf_alpha_0.7.log 2>&1
    sleep $SLEEP_TIME

    # SDTPS SPARSE_RATIO 消融
    echo "[GPU 1] 3/6: sdtps_sparse_0.5"
    CUDA_VISIBLE_DEVICES=1 python train_net.py --config_file $CONFIG \
        --exp_name "sdtps_sparse_0.5" \
        MODEL.USE_DGAF True MODEL.SDTPS_SPARSE_RATIO 0.5 \
        > $GPU_LOG_DIR/03_sdtps_sparse_0.5.log 2>&1
    sleep $SLEEP_TIME

    echo "[GPU 1] 4/6: sdtps_sparse_0.6"
    CUDA_VISIBLE_DEVICES=1 python train_net.py --config_file $CONFIG \
        --exp_name "sdtps_sparse_0.6" \
        MODEL.USE_DGAF True MODEL.SDTPS_SPARSE_RATIO 0.6 \
        > $GPU_LOG_DIR/04_sdtps_sparse_0.6.log 2>&1
    sleep $SLEEP_TIME

    echo "[GPU 1] 5/6: sdtps_sparse_0.8"
    CUDA_VISIBLE_DEVICES=1 python train_net.py --config_file $CONFIG \
        --exp_name "sdtps_sparse_0.8" \
        MODEL.USE_DGAF True MODEL.SDTPS_SPARSE_RATIO 0.8 \
        > $GPU_LOG_DIR/05_sdtps_sparse_0.8.log 2>&1
    sleep $SLEEP_TIME

    echo "[GPU 1] 6/6: sdtps_sparse_0.9"
    CUDA_VISIBLE_DEVICES=1 python train_net.py --config_file $CONFIG \
        --exp_name "sdtps_sparse_0.9" \
        MODEL.USE_DGAF True MODEL.SDTPS_SPARSE_RATIO 0.9 \
        > $GPU_LOG_DIR/06_sdtps_sparse_0.9.log 2>&1

    echo "[GPU 1] 完成!" > $GPU_LOG_DIR/DONE
) &

# ============================================================================
# GPU 2: SDTPS AGGR_RATIO + BETA 消融
# ============================================================================
(
    GPU_LOG_DIR="$EXP_DIR/gpu2_aggr_beta"
    mkdir -p $GPU_LOG_DIR
    echo "[GPU 2] 日志目录: $GPU_LOG_DIR"

    # SDTPS AGGR_RATIO 消融
    echo "[GPU 2] 1/6: sdtps_aggr_0.3"
    CUDA_VISIBLE_DEVICES=2 python train_net.py --config_file $CONFIG \
        --exp_name "sdtps_aggr_0.3" \
        MODEL.USE_DGAF True MODEL.SDTPS_AGGR_RATIO 0.3 \
        > $GPU_LOG_DIR/01_sdtps_aggr_0.3.log 2>&1
    sleep $SLEEP_TIME

    echo "[GPU 2] 2/6: sdtps_aggr_0.6"
    CUDA_VISIBLE_DEVICES=2 python train_net.py --config_file $CONFIG \
        --exp_name "sdtps_aggr_0.6" \
        MODEL.USE_DGAF True MODEL.SDTPS_AGGR_RATIO 0.6 \
        > $GPU_LOG_DIR/02_sdtps_aggr_0.6.log 2>&1
    sleep $SLEEP_TIME

    # SDTPS BETA 消融
    echo "[GPU 2] 3/6: sdtps_beta_0.15"
    CUDA_VISIBLE_DEVICES=2 python train_net.py --config_file $CONFIG \
        --exp_name "sdtps_beta_0.15" \
        MODEL.USE_DGAF True MODEL.SDTPS_BETA 0.15 \
        > $GPU_LOG_DIR/03_sdtps_beta_0.15.log 2>&1
    sleep $SLEEP_TIME

    echo "[GPU 2] 4/6: sdtps_beta_0.2"
    CUDA_VISIBLE_DEVICES=2 python train_net.py --config_file $CONFIG \
        --exp_name "sdtps_beta_0.2" \
        MODEL.USE_DGAF True MODEL.SDTPS_BETA 0.2 \
        > $GPU_LOG_DIR/04_sdtps_beta_0.2.log 2>&1
    sleep $SLEEP_TIME

    echo "[GPU 2] 5/6: sdtps_beta_0.3"
    CUDA_VISIBLE_DEVICES=2 python train_net.py --config_file $CONFIG \
        --exp_name "sdtps_beta_0.3" \
        MODEL.USE_DGAF True MODEL.SDTPS_BETA 0.3 \
        > $GPU_LOG_DIR/05_sdtps_beta_0.3.log 2>&1
    sleep $SLEEP_TIME

    echo "[GPU 2] 6/6: sdtps_beta_0.35"
    CUDA_VISIBLE_DEVICES=2 python train_net.py --config_file $CONFIG \
        --exp_name "sdtps_beta_0.35" \
        MODEL.USE_DGAF True MODEL.SDTPS_BETA 0.35 \
        > $GPU_LOG_DIR/06_sdtps_beta_0.35.log 2>&1

    echo "[GPU 2] 完成!" > $GPU_LOG_DIR/DONE
) &

# ============================================================================
# GPU 3: SDTPS LOSS_WEIGHT + 组合实验
# ============================================================================
(
    GPU_LOG_DIR="$EXP_DIR/gpu3_loss_combo"
    mkdir -p $GPU_LOG_DIR
    echo "[GPU 3] 日志目录: $GPU_LOG_DIR"

    # SDTPS LOSS_WEIGHT 消融
    echo "[GPU 3] 1/6: sdtps_loss_0.5"
    CUDA_VISIBLE_DEVICES=3 python train_net.py --config_file $CONFIG \
        --exp_name "sdtps_loss_0.5" \
        MODEL.USE_DGAF True MODEL.SDTPS_LOSS_WEIGHT 0.5 \
        > $GPU_LOG_DIR/01_sdtps_loss_0.5.log 2>&1
    sleep $SLEEP_TIME

    echo "[GPU 3] 2/6: sdtps_loss_1.5"
    CUDA_VISIBLE_DEVICES=3 python train_net.py --config_file $CONFIG \
        --exp_name "sdtps_loss_1.5" \
        MODEL.USE_DGAF True MODEL.SDTPS_LOSS_WEIGHT 1.5 \
        > $GPU_LOG_DIR/02_sdtps_loss_1.5.log 2>&1
    sleep $SLEEP_TIME

    echo "[GPU 3] 3/6: sdtps_loss_2.0"
    CUDA_VISIBLE_DEVICES=3 python train_net.py --config_file $CONFIG \
        --exp_name "sdtps_loss_2.0" \
        MODEL.USE_DGAF True MODEL.SDTPS_LOSS_WEIGHT 2.0 \
        > $GPU_LOG_DIR/03_sdtps_loss_2.0.log 2>&1
    sleep $SLEEP_TIME

    # 组合实验：基于经验的最佳候选
    echo "[GPU 3] 4/6: combo_v1 (balanced)"
    CUDA_VISIBLE_DEVICES=3 python train_net.py --config_file $CONFIG \
        --exp_name "combo_v1_balanced" \
        MODEL.USE_DGAF True \
        MODEL.DGAF_TAU 1.0 MODEL.DGAF_INIT_ALPHA 0.5 \
        MODEL.SDTPS_SPARSE_RATIO 0.7 MODEL.SDTPS_AGGR_RATIO 0.5 \
        MODEL.SDTPS_BETA 0.25 MODEL.SDTPS_LOSS_WEIGHT 1.0 \
        > $GPU_LOG_DIR/04_combo_v1_balanced.log 2>&1
    sleep $SLEEP_TIME

    echo "[GPU 3] 5/6: combo_v2 (more_tokens)"
    CUDA_VISIBLE_DEVICES=3 python train_net.py --config_file $CONFIG \
        --exp_name "combo_v2_more_tokens" \
        MODEL.USE_DGAF True \
        MODEL.DGAF_TAU 0.7 MODEL.DGAF_INIT_ALPHA 0.5 \
        MODEL.SDTPS_SPARSE_RATIO 0.8 MODEL.SDTPS_AGGR_RATIO 0.4 \
        MODEL.SDTPS_BETA 0.25 MODEL.SDTPS_LOSS_WEIGHT 1.0 \
        > $GPU_LOG_DIR/05_combo_v2_more_tokens.log 2>&1
    sleep $SLEEP_TIME

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
echo "所有 GPU 任务已启动!"
echo "============================================================"
echo ""
echo "实验目录: $EXP_DIR"
echo ""
echo "监控命令:"
echo "  nvidia-smi"
echo "  watch -n 60 'ls $EXP_DIR/*/DONE 2>/dev/null | wc -l'"
echo "  tail -f $EXP_DIR/*/*.log"
echo ""

# 等待所有任务完成
wait

echo "============================================================"
echo "所有消融实验已完成!"
echo "============================================================"

# 生成结果汇总
echo ""
echo "生成结果汇总..."
SUMMARY_FILE="$EXP_DIR/results_summary.txt"

echo "SDTPS + DGAF 消融实验结果汇总" > $SUMMARY_FILE
echo "实验时间: $(date)" >> $SUMMARY_FILE
echo "============================================================" >> $SUMMARY_FILE
echo "" >> $SUMMARY_FILE

# 提取所有实验的 Best mAP 和 Rank-1
echo "实验结果 (按 mAP 排序):" >> $SUMMARY_FILE
echo "----------------------------------------" >> $SUMMARY_FILE
grep -r "Best mAP" $EXP_DIR/*/*.log 2>/dev/null | \
    sed 's/.*\/\([^\/]*\)\.log:.*Best mAP: \([0-9.]*\).*/\2 \1/' | \
    sort -rn >> $SUMMARY_FILE

echo "" >> $SUMMARY_FILE
echo "详细结果:" >> $SUMMARY_FILE
echo "----------------------------------------" >> $SUMMARY_FILE
for log in $EXP_DIR/*/*.log; do
    exp_name=$(basename $log .log)
    map=$(grep "Best mAP" $log 2>/dev/null | tail -1 | grep -oP "Best mAP: \K[0-9.]+")
    rank1=$(grep "Best Rank-1" $log 2>/dev/null | tail -1 | grep -oP "Best Rank-1: \K[0-9.]+")
    if [ ! -z "$map" ]; then
        echo "$exp_name: mAP=$map, Rank-1=$rank1" >> $SUMMARY_FILE
    fi
done

echo ""
echo "结果汇总已保存到: $SUMMARY_FILE"
cat $SUMMARY_FILE
