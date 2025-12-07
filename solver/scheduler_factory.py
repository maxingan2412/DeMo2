""" Scheduler Factory
Hacked together by / Copyright 2020 Ross Wightman
"""
from .cosine_lr import CosineLRScheduler
from .lr_scheduler import WarmupMultiStepLR


def create_scheduler(cfg, optimizer):
    """
    创建学习率调度器
    使用 WarmupMultiStepLR (线性衰减) 替代 CosineLRScheduler
    """
    lr_scheduler = WarmupMultiStepLR(
        optimizer,
        milestones=cfg.SOLVER.STEPS,
        gamma=cfg.SOLVER.GAMMA,
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
        warmup_iters=cfg.SOLVER.WARMUP_ITERS,
        warmup_method=cfg.SOLVER.WARMUP_METHOD
    )

    return lr_scheduler


# 保留原始 Cosine 调度器作为备用
# def create_scheduler_cosine(cfg, optimizer):
#     num_epochs = cfg.SOLVER.MAX_EPOCHS
#     lr_min = 0.001 * cfg.SOLVER.BASE_LR
#     warmup_lr_init = 0.1 * cfg.SOLVER.BASE_LR
#
#     warmup_t = cfg.SOLVER.WARMUP_ITERS
#     noise_range = (0, num_epochs)
#
#
#     lr_scheduler = CosineLRScheduler(
#         optimizer,
#         t_initial=num_epochs,
#         lr_min=lr_min,
#         t_mul=1.,
#         decay_rate=0.1,
#         warmup_lr_init=warmup_lr_init,
#         warmup_t=warmup_t,
#         cycle_limit=1,
#         t_in_epochs=True,
#         noise_range_t=noise_range,
#         noise_pct=0.67,
#         noise_std=1.,
#         noise_seed=42,
#     )
#
#     return lr_scheduler
