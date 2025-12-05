#!/bin/bash

# 顺序训练脚本 - 自动运行多个实验配置
# 用法: bash run_sequential_experiments.sh

# ============================================
# 配置区域 - 在这里添加要运行的实验
# ============================================

# 实验列表（按顺序执行）
# 格式: "实验名称:配置文件路径"
experiments=(
    "SACR_SDTPS:configs/RGBNT201/DeMo_SACR_SDTPS.yml"
    "SDTPS_only:configs/RGBNT201/DeMo_SDTPS.yml"
    "Original_DeMo:configs/RGBNT201/DeMo.yml"
)

# ============================================
# 全局设置
# ============================================
LOG_DIR="experiment_logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="${LOG_DIR}/sequential_run_${TIMESTAMP}.log"

# 创建日志目录
mkdir -p ${LOG_DIR}

# ============================================
# 辅助函数
# ============================================

log_message() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a ${MAIN_LOG}
}

run_experiment() {
    local exp_name=$1
    local config_file=$2
    local exp_log="${LOG_DIR}/${exp_name}_${TIMESTAMP}.log"

    log_message "========================================"
    log_message "开始实验: ${exp_name}"
    log_message "配置文件: ${config_file}"
    log_message "日志文件: ${exp_log}"
    log_message "========================================"

    # 记录开始时间
    local start_time=$(date +%s)

    # 运行训练
    python train_net.py --config_file ${config_file} 2>&1 | tee ${exp_log}

    # 检查退出状态
    local exit_code=$?
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    local seconds=$((duration % 60))

    log_message "----------------------------------------"
    if [ ${exit_code} -eq 0 ]; then
        log_message "✓ 实验 ${exp_name} 完成"
        log_message "  耗时: ${hours}h ${minutes}m ${seconds}s"
        log_message "  日志: ${exp_log}"
    else
        log_message "✗ 实验 ${exp_name} 失败 (退出码: ${exit_code})"
        log_message "  耗时: ${hours}h ${minutes}m ${seconds}s"
        log_message "  日志: ${exp_log}"

        # 询问是否继续
        read -p "实验失败，是否继续下一个实验？(y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_message "用户选择终止所有实验"
            exit 1
        fi
    fi
    log_message "========================================\n"

    # 清理 GPU 显存
    sleep 5
}

# ============================================
# 主流程
# ============================================

log_message "========================================"
log_message "顺序训练脚本启动"
log_message "========================================"
log_message "总共 ${#experiments[@]} 个实验"
log_message "主日志: ${MAIN_LOG}"
log_message ""

# 显示实验列表
log_message "实验列表:"
for i in "${!experiments[@]}"; do
    IFS=':' read -r exp_name config_file <<< "${experiments[$i]}"
    log_message "  $((i+1)). ${exp_name} - ${config_file}"
done
log_message ""

# 询问确认
read -p "确认开始所有实验？(y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    log_message "用户取消"
    exit 0
fi

# 记录总开始时间
total_start_time=$(date +%s)

# 顺序执行所有实验
for i in "${!experiments[@]}"; do
    IFS=':' read -r exp_name config_file <<< "${experiments[$i]}"

    log_message "\n进度: $((i+1))/${#experiments[@]}"
    run_experiment "${exp_name}" "${config_file}"

    # 如果不是最后一个实验，等待几秒
    if [ $((i+1)) -lt ${#experiments[@]} ]; then
        log_message "等待 10 秒后开始下一个实验..."
        sleep 10
    fi
done

# 总结
total_end_time=$(date +%s)
total_duration=$((total_end_time - total_start_time))
total_hours=$((total_duration / 3600))
total_minutes=$(((total_duration % 3600) / 60))

log_message "\n========================================"
log_message "所有实验完成！"
log_message "========================================"
log_message "总耗时: ${total_hours}h ${total_minutes}m"
log_message "主日志: ${MAIN_LOG}"
log_message "========================================"
