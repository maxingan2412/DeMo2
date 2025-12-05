"""
顺序训练脚本（Python 版本）

用法:
    python run_sequential_experiments.py

特性:
    - 按顺序自动运行多个训练配置
    - 等待前一个训练完成后再开始下一个
    - 记录每个实验的时间和结果
    - 支持训练失败时的处理（继续/停止）
    - 生成汇总报告
"""

import os
import sys
import time
import subprocess
from datetime import datetime

# ============================================
# 配置区域 - 在这里添加要运行的实验
# ============================================

EXPERIMENTS = [
    {
        'name': 'SACR_SDTPS',
        'config': 'configs/RGBNT201/DeMo_SACR_SDTPS.yml',
        'description': 'SACR + SDTPS 完整版'
    },
    {
        'name': 'SDTPS_only',
        'config': 'configs/RGBNT201/DeMo_SDTPS.yml',
        'description': '只用 SDTPS'
    },
    {
        'name': 'Original_DeMo',
        'config': 'configs/RGBNT201/DeMo.yml',
        'description': '原始 DeMo (HDM+ATM)'
    },
]

# ============================================
# 全局设置
# ============================================

LOG_DIR = "experiment_logs"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
MAIN_LOG = os.path.join(LOG_DIR, f"sequential_run_{TIMESTAMP}.log")


# ============================================
# 辅助函数
# ============================================

def log_message(msg, print_only=False):
    """记录日志消息"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{timestamp}] {msg}"
    print(formatted_msg)

    if not print_only:
        with open(MAIN_LOG, 'a') as f:
            f.write(formatted_msg + '\n')


def format_time(seconds):
    """格式化时间"""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours}h {minutes}m {secs}s"


def run_experiment(exp_name, config_file, description):
    """运行单个实验"""
    exp_log = os.path.join(LOG_DIR, f"{exp_name}_{TIMESTAMP}.log")

    log_message("=" * 80)
    log_message(f"开始实验: {exp_name}")
    log_message(f"描述: {description}")
    log_message(f"配置文件: {config_file}")
    log_message(f"日志文件: {exp_log}")
    log_message("=" * 80)

    # 检查配置文件是否存在
    if not os.path.exists(config_file):
        log_message(f"✗ 配置文件不存在: {config_file}")
        return False

    # 记录开始时间
    start_time = time.time()

    # 构建训练命令
    cmd = [
        sys.executable,  # python
        'train_net.py',
        '--config_file', config_file
    ]

    # 运行训练
    try:
        with open(exp_log, 'w') as f:
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                buffersize=1
            )

            # 实时显示进度（每30秒检查一次）
            while True:
                return_code = process.poll()
                if return_code is not None:
                    break
                time.sleep(30)
                elapsed = int(time.time() - start_time)
                log_message(f"  进行中... 已运行 {format_time(elapsed)}", print_only=True)

        exit_code = process.returncode

    except KeyboardInterrupt:
        log_message("\n用户中断训练")
        process.terminate()
        process.wait()
        return False
    except Exception as e:
        log_message(f"✗ 运行失败: {e}")
        return False

    # 计算耗时
    end_time = time.time()
    duration = int(end_time - start_time)

    log_message("-" * 80)
    if exit_code == 0:
        log_message(f"✓ 实验 {exp_name} 完成")
        log_message(f"  耗时: {format_time(duration)}")
        log_message(f"  日志: {exp_log}")
        success = True
    else:
        log_message(f"✗ 实验 {exp_name} 失败 (退出码: {exit_code})")
        log_message(f"  耗时: {format_time(duration)}")
        log_message(f"  日志: {exp_log}")

        # 询问是否继续
        while True:
            response = input("\n实验失败，是否继续下一个实验？(y/n): ").strip().lower()
            if response in ['y', 'n']:
                break
            print("请输入 y 或 n")

        if response != 'y':
            log_message("用户选择终止所有实验")
            return False
        success = True

    log_message("=" * 80 + "\n")
    return success


def main():
    """主流程"""
    # 创建日志目录
    os.makedirs(LOG_DIR, exist_ok=True)

    log_message("=" * 80)
    log_message("顺序训练脚本启动")
    log_message("=" * 80)
    log_message(f"总共 {len(EXPERIMENTS)} 个实验")
    log_message(f"主日志: {MAIN_LOG}\n")

    # 显示实验列表
    log_message("实验列表:")
    for i, exp in enumerate(EXPERIMENTS):
        log_message(f"  {i+1}. {exp['name']}")
        log_message(f"     描述: {exp['description']}")
        log_message(f"     配置: {exp['config']}")
    log_message("")

    # 询问确认
    response = input("确认开始所有实验？(y/n): ").strip().lower()
    if response != 'y':
        log_message("用户取消")
        return

    # 记录总开始时间
    total_start_time = time.time()
    completed_experiments = []
    failed_experiments = []

    # 顺序执行所有实验
    for i, exp in enumerate(EXPERIMENTS):
        log_message(f"\n进度: {i+1}/{len(EXPERIMENTS)}")

        success = run_experiment(
            exp_name=exp['name'],
            config_file=exp['config'],
            description=exp['description']
        )

        if success:
            completed_experiments.append(exp['name'])
        else:
            failed_experiments.append(exp['name'])
            break  # 用户选择停止

        # 如果不是最后一个实验，等待几秒
        if i + 1 < len(EXPERIMENTS):
            log_message("等待 10 秒后开始下一个实验...")
            time.sleep(10)

    # 生成总结报告
    total_duration = int(time.time() - total_start_time)

    log_message("\n" + "=" * 80)
    log_message("所有实验完成！")
    log_message("=" * 80)
    log_message(f"总耗时: {format_time(total_duration)}")
    log_message(f"完成实验: {len(completed_experiments)}/{len(EXPERIMENTS)}")

    if completed_experiments:
        log_message("\n成功完成的实验:")
        for exp_name in completed_experiments:
            log_message(f"  ✓ {exp_name}")

    if failed_experiments:
        log_message("\n失败/中断的实验:")
        for exp_name in failed_experiments:
            log_message(f"  ✗ {exp_name}")

    log_message(f"\n主日志: {MAIN_LOG}")
    log_message("=" * 80)

    # 生成结果汇总文件
    summary_file = os.path.join(LOG_DIR, f"summary_{TIMESTAMP}.txt")
    with open(summary_file, 'w') as f:
        f.write("实验汇总报告\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总耗时: {format_time(total_duration)}\n")
        f.write(f"完成: {len(completed_experiments)}/{len(EXPERIMENTS)}\n\n")

        f.write("-" * 80 + "\n")
        f.write("实验结果:\n")
        f.write("-" * 80 + "\n")
        for exp in EXPERIMENTS:
            status = "✓ 完成" if exp['name'] in completed_experiments else "✗ 失败/未运行"
            f.write(f"{status:12s} {exp['name']:20s} - {exp['description']}\n")
            f.write(f"{'':12s} {exp['config']}\n\n")

        f.write("-" * 80 + "\n")
        f.write("详细日志:\n")
        f.write("-" * 80 + "\n")
        for exp in EXPERIMENTS:
            if exp['name'] in completed_experiments:
                exp_log = os.path.join(LOG_DIR, f"{exp['name']}_{TIMESTAMP}.log")
                f.write(f"{exp['name']}: {exp_log}\n")

    log_message(f"汇总报告: {summary_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
