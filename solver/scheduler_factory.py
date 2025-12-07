""" Scheduler Factory
Hacked together by / Copyright 2020 Ross Wightman
"""
from .cosine_lr import CosineLRScheduler
from .lr_scheduler import WarmupMultiStepLR


def create_scheduler(cfg, optimizer):
    """
    根据配置创建学习率调度器

    Args:
        cfg: 配置对象，需要包含 SOLVER.LR_SCHEDULER
        optimizer: 优化器

    Returns:
        lr_scheduler: 学习率调度器

    配置选项 (cfg.SOLVER.LR_SCHEDULER):
        - 'cosine': CosineLRScheduler (余弦退火，默认)
        - 'multistep': WarmupMultiStepLR (线性衰减)
    """
    scheduler_type = cfg.SOLVER.LR_SCHEDULER.lower()

    if scheduler_type == 'cosine':
        return _create_cosine_scheduler(cfg, optimizer)
    elif scheduler_type == 'multistep':
        return _create_multistep_scheduler(cfg, optimizer)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}. "
                        f"Supported: 'cosine', 'multistep'")


def _create_multistep_scheduler(cfg, optimizer):
    """创建 WarmupMultiStepLR 调度器 (线性衰减)"""
    lr_scheduler = WarmupMultiStepLR(
        optimizer,
        milestones=cfg.SOLVER.STEPS,
        gamma=cfg.SOLVER.GAMMA,
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
        warmup_iters=cfg.SOLVER.WARMUP_ITERS,
        warmup_method=cfg.SOLVER.WARMUP_METHOD
    )
    return lr_scheduler


def _create_cosine_scheduler(cfg, optimizer):
    """创建 CosineLRScheduler 调度器 (余弦退火)"""
    num_epochs = cfg.SOLVER.MAX_EPOCHS
    lr_min = 0.001 * cfg.SOLVER.BASE_LR
    warmup_lr_init = 0.1 * cfg.SOLVER.BASE_LR

    warmup_t = cfg.SOLVER.WARMUP_ITERS
    noise_range = (0, num_epochs)

    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=num_epochs,
        lr_min=lr_min,
        t_mul=1.,
        decay_rate=0.1,
        warmup_lr_init=warmup_lr_init,
        warmup_t=warmup_t,
        cycle_limit=1,
        t_in_epochs=True,
        noise_range_t=noise_range,
        noise_pct=0.67,
        noise_std=1.,
        noise_seed=42,
    )
    return lr_scheduler


# 保留独立函数以便直接调用
def create_scheduler_cosine(cfg, optimizer):
    """直接创建 CosineLRScheduler (余弦退火)"""
    return _create_cosine_scheduler(cfg, optimizer)


def create_scheduler_multistep(cfg, optimizer):
    """直接创建 WarmupMultiStepLR (线性衰减)"""
    return _create_multistep_scheduler(cfg, optimizer)
