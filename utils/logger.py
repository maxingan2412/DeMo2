import logging
import os
import sys
import re
import os.path as osp
from datetime import datetime


def setup_logger(name, save_dir, if_train):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        if not osp.exists(save_dir):
            os.makedirs(save_dir)

        # 生成带时间戳和完整命令的日志文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 获取完整命令行参数: train_net.py --config_file configs/RGBNT201/DeMo.yml
        cmd_args = " ".join(sys.argv)
        # 替换特殊字符为下划线，确保文件名有效
        cmd_args_safe = re.sub(r'[/\\:*?"<>|\s]+', '_', cmd_args)
        # 限制长度，避免文件名过长
        if len(cmd_args_safe) > 100:
            cmd_args_safe = cmd_args_safe[:100]

        if if_train:
            log_file = os.path.join(save_dir, f"train_log_{timestamp}_{cmd_args_safe}.txt")
        else:
            log_file = os.path.join(save_dir, f"test_log_{timestamp}_{cmd_args_safe}.txt")

        fh = logging.FileHandler(log_file, mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
