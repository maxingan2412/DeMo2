#!/bin/bash
# ============================================================================
# 消融实验脚本
# 基于 configs/RGBNT201/DeMo_SACR_SDTPS_LIF.yml
# 分散到 GPU 0-3
# ============================================================================

CONFIG="configs/RGBNT201/DeMo_SACR_SDTPS_LIF.yml"

# ============================================================================
# 第一组：模块消融实验 (Module Ablation)
# ============================================================================

echo "========== 模块消融实验 =========="

# GPU 0: Baseline (无任何模块) 和 +SACR
CUDA_VISIBLE_DEVICES=0 nohup python train_net.py --config_file $CONFIG \
    --exp_name "ablation_baseline" \
    MODEL.USE_SACR False MODEL.USE_SDTPS False MODEL.USE_LIF False \
    > logs/ablation_baseline.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python train_net.py --config_file $CONFIG \
    --exp_name "ablation_SACR_only" \
    MODEL.USE_SACR True MODEL.USE_SDTPS False MODEL.USE_LIF False \
    > logs/ablation_SACR_only.log 2>&1 &

# GPU 1: +SDTPS 和 +LIF
CUDA_VISIBLE_DEVICES=1 nohup python train_net.py --config_file $CONFIG \
    --exp_name "ablation_SDTPS_only" \
    MODEL.USE_SACR False MODEL.USE_SDTPS True MODEL.USE_LIF False \
    > logs/ablation_SDTPS_only.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python train_net.py --config_file $CONFIG \
    --exp_name "ablation_LIF_only" \
    MODEL.USE_SACR False MODEL.USE_SDTPS False MODEL.USE_LIF True \
    > logs/ablation_LIF_only.log 2>&1 &

# GPU 2: +SACR+SDTPS 和 +SACR+LIF
CUDA_VISIBLE_DEVICES=2 nohup python train_net.py --config_file $CONFIG \
    --exp_name "ablation_SACR_SDTPS" \
    MODEL.USE_SACR True MODEL.USE_SDTPS True MODEL.USE_LIF False \
    > logs/ablation_SACR_SDTPS.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python train_net.py --config_file $CONFIG \
    --exp_name "ablation_SACR_LIF" \
    MODEL.USE_SACR True MODEL.USE_SDTPS False MODEL.USE_LIF True \
    > logs/ablation_SACR_LIF.log 2>&1 &

# GPU 3: +SDTPS+LIF 和 Full (SACR+SDTPS+LIF)
CUDA_VISIBLE_DEVICES=3 nohup python train_net.py --config_file $CONFIG \
    --exp_name "ablation_SDTPS_LIF" \
    MODEL.USE_SACR False MODEL.USE_SDTPS True MODEL.USE_LIF True \
    > logs/ablation_SDTPS_LIF.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python train_net.py --config_file $CONFIG \
    --exp_name "ablation_full_SACR_SDTPS_LIF" \
    MODEL.USE_SACR True MODEL.USE_SDTPS True MODEL.USE_LIF True \
    > logs/ablation_full_SACR_SDTPS_LIF.log 2>&1 &

echo "模块消融实验已启动，查看 logs/ 目录"

# ============================================================================
# 第二组：LIF 超参数消融
# ============================================================================

echo "========== LIF 超参数消融 =========="

# GPU 0: LIF_BETA 消融
CUDA_VISIBLE_DEVICES=0 nohup python train_net.py --config_file $CONFIG \
    --exp_name "ablation_LIF_BETA_0.2" \
    MODEL.LIF_BETA 0.2 \
    > logs/ablation_LIF_BETA_0.2.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python train_net.py --config_file $CONFIG \
    --exp_name "ablation_LIF_BETA_0.6" \
    MODEL.LIF_BETA 0.6 \
    > logs/ablation_LIF_BETA_0.6.log 2>&1 &

# GPU 1: LIF_LOSS_WEIGHT 消融
CUDA_VISIBLE_DEVICES=1 nohup python train_net.py --config_file $CONFIG \
    --exp_name "ablation_LIF_LOSS_WEIGHT_0.1" \
    MODEL.LIF_LOSS_WEIGHT 0.1 \
    > logs/ablation_LIF_LOSS_WEIGHT_0.1.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python train_net.py --config_file $CONFIG \
    --exp_name "ablation_LIF_LOSS_WEIGHT_0.3" \
    MODEL.LIF_LOSS_WEIGHT 0.3 \
    > logs/ablation_LIF_LOSS_WEIGHT_0.3.log 2>&1 &

# ============================================================================
# 第三组：SDTPS 超参数消融
# ============================================================================

echo "========== SDTPS 超参数消融 =========="

# GPU 2: SDTPS_SPARSE_RATIO 消融
CUDA_VISIBLE_DEVICES=2 nohup python train_net.py --config_file $CONFIG \
    --exp_name "ablation_SDTPS_SPARSE_0.5" \
    MODEL.SDTPS_SPARSE_RATIO 0.5 \
    > logs/ablation_SDTPS_SPARSE_0.5.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python train_net.py --config_file $CONFIG \
    --exp_name "ablation_SDTPS_SPARSE_0.8" \
    MODEL.SDTPS_SPARSE_RATIO 0.8 \
    > logs/ablation_SDTPS_SPARSE_0.8.log 2>&1 &

# GPU 3: SDTPS_LOSS_WEIGHT 消融
CUDA_VISIBLE_DEVICES=3 nohup python train_net.py --config_file $CONFIG \
    --exp_name "ablation_SDTPS_LOSS_1.0" \
    MODEL.SDTPS_LOSS_WEIGHT 1.0 \
    > logs/ablation_SDTPS_LOSS_1.0.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python train_net.py --config_file $CONFIG \
    --exp_name "ablation_SDTPS_LOSS_3.0" \
    MODEL.SDTPS_LOSS_WEIGHT 3.0 \
    > logs/ablation_SDTPS_LOSS_3.0.log 2>&1 &

echo "超参数消融实验已启动，查看 logs/ 目录"
echo "使用 'nvidia-smi' 或 'ps aux | grep train_net' 查看运行状态"
