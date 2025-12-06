import logging
import os
import sys
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

        # 生成带时间戳和脚本名的日志文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 获取执行的脚本名（不带路径和扩展名）
        script_name = osp.splitext(osp.basename(sys.argv[0]))[0]  # e.g., "train_net"

        if if_train:
            log_file = os.path.join(save_dir, f"train_log_{timestamp}_{script_name}.txt")
        else:
            log_file = os.path.join(save_dir, f"test_log_{timestamp}_{script_name}.txt")

        fh = logging.FileHandler(log_file, mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
