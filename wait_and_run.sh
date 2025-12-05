#!/bin/bash

# 等待当前训练完成，然后自动运行下一个配置
# 用法: bash wait_and_run.sh <配置文件>
# 示例: bash wait_and_run.sh configs/RGBNT201/DeMo_SDTPS.yml

if [ $# -eq 0 ]; then
    echo "用法: bash wait_and_run.sh <配置文件>"
    echo "示例: bash wait_and_run.sh configs/RGBNT201/DeMo_SDTPS.yml"
    exit 1
fi

CONFIG_FILE=$1

echo "========================================"
echo "等待并运行脚本"
echo "========================================"
echo "下一个配置: ${CONFIG_FILE}"
echo ""

# 检查配置文件是否存在
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "✗ 配置文件不存在: ${CONFIG_FILE}"
    exit 1
fi

# 等待当前的训练进程结束
echo "正在等待当前训练进程结束..."
echo "（检测 train_net.py 进程）"
echo ""

while pgrep -f "train_net.py" > /dev/null; do
    # 显示当前运行的进程
    CURRENT_PROCESS=$(ps aux | grep "train_net.py" | grep -v grep | head -1)
    echo -ne "\r等待中... $(date '+%H:%M:%S')  "
    sleep 10
done

echo ""
echo "✓ 当前训练已完成"
echo ""

# 等待几秒，确保 GPU 显存释放
echo "等待 10 秒，释放 GPU 显存..."
sleep 10

# 开始新的训练
echo "========================================"
echo "开始新的训练"
echo "========================================"
echo "配置文件: ${CONFIG_FILE}"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 创建日志文件
LOG_FILE="experiment_logs/$(basename ${CONFIG_FILE} .yml)_$(date +%Y%m%d_%H%M%S).log"
mkdir -p experiment_logs

echo "日志文件: ${LOG_FILE}"
echo ""

# 运行训练
python train_net.py --config_file ${CONFIG_FILE} 2>&1 | tee ${LOG_FILE}

# 记录完成时间
echo ""
echo "========================================"
echo "训练完成"
echo "========================================"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "日志文件: ${LOG_FILE}"
